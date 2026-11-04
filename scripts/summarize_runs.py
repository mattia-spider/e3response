"""Summarise training runs from their *resolved* configs (tasks E1-E4 of REPORT_verifica.md).

Replacement for the override-string based ``summarize_grid.py`` (which is not in this repo).
Every column comes from the ``config.yaml`` that Hydra/tensorial writes into the run directory
(overrides already applied, ``${oc.select:...}`` and ``from_data`` already resolved), so no column
depends on how the run was launched.

Columns
    model            direct | deriv1st | deriv2nd | (+ "_tp" if the NodewiseFieldTensorProduct is present)
    r_max, ell_max, mul, num_layers    from the Nequip stack / data config
    w_energy, w_nmr, w_induced_magnetic_field   loss weights (``True`` -> 1.0, absent/False -> 0)
    n_elements, has_Z0                 from_data type map; ``has_Z0`` flags the padding species 0
                                       that leaked into old runs (the source of ``test_Z0``)
    status           complete | incomplete (running/killed/crashed): no test metrics and no
                     "Best ckpt path" line in train.log.  Incomplete runs are listed separately
                     and are NOT part of the comparison table.
    split_hash       md5 of the sorted train/val/test indices that ``Qm9NmrDataModule.load_split``
                     reconstructs from (dataset, limit, fractions); runs that must be compared
                     need identical hashes.  Needs the data archive (``--data-dir``); skipped with
                     ``--no-split-hash``.
    test_*           every ``test*`` column found in the run's metrics.csv

Usage::

    python scripts/summarize_runs.py logs/train/multiruns/2026-09-22_15-28-39 logs/train/runs \
        --out summary.csv [--data-dir data/qm9_nmr] [--no-split-hash]
"""

import argparse
import csv
import hashlib
import pathlib
import re

import yaml

HARTREE_TO_KCAL = 627.5094740631


def load_yaml(path):
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:  # pylint: disable=broad-except
        return None


def find_runs(roots):
    seen = set()
    for root in roots:
        root = pathlib.Path(root)
        for cfg in sorted(root.rglob("config.yaml")):
            if ".hydra" in cfg.parts:
                continue
            if cfg.parent not in seen:
                seen.add(cfg.parent)
                yield cfg.parent


def _irreps_stats(hidden):
    """(mul, ell_max) from a resolved irreps string or from a make_irreps dict."""
    if isinstance(hidden, dict):
        return hidden.get("mul"), hidden.get("ell_max", hidden.get("lmax"))
    if isinstance(hidden, str):
        terms = re.findall(r"(\d+)x(\d+)([eo])", hidden.replace(" ", ""))
        if terms:
            return max(int(m) for m, _, _ in terms), max(int(l) for _, l, _ in terms)
    return None, None


def _weight(v):
    if v is None or v is False:
        return 0.0
    return 1.0 if v is True else float(v)


def model_info(cfg):
    layers = (cfg.get("model", {}).get("model", {}) or {}).get("layers", []) or []
    targets = [str(layer.get("_target_", "")) for layer in layers]
    has = lambda s: any(s in t for t in targets)  # noqa: E731
    if has("InducedMagneticField"):
        kind = "deriv2nd"
    elif has("MagneticShieldingTensor"):
        kind = "deriv1st"
    else:
        kind = "direct"
    if has("NodewiseFieldTensorProduct"):
        kind += "_tp"
    mul = ell = num_layers = None
    for layer in layers:
        t = str(layer.get("_target_", ""))
        if t.endswith(".Nequip"):
            mul, ell = _irreps_stats(layer.get("hidden_irreps"))
            num_layers = layer.get("num_layers")
        elif t.endswith("NequipLayer") or t.endswith("build_nequip_backbone"):
            if t.endswith("build_nequip_backbone"):
                mul, ell, num_layers = layer.get("mul"), layer.get("lmax"), layer.get("num_layers")
            else:
                num_layers = (num_layers or 0) + 1
                mul, ell = _irreps_stats(layer.get("irreps_out"))
    return kind, mul, ell, num_layers


def read_metrics(run):
    out = {}
    for f in sorted((run / "csv").glob("*/metrics.csv")):
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for k, v in row.items():
                    if k and k.lower().startswith("test") and v not in ("", None):
                        try:
                            out[k] = float(v)
                        except ValueError:
                            pass
    return out


def run_status(run, test_metrics):
    log = run / "train.log"
    if test_metrics:
        return "complete"
    if log.exists():
        tail = log.read_text(errors="ignore")[-4000:]
        if "Best ckpt path" in tail and "Traceback" not in tail:
            return "complete"
    return "incomplete"


_SPLIT_CACHE = {}


def split_hash(cfg, data_dir_override):
    d = cfg.get("data", {})
    if "Qm9Nmr" not in str(d.get("_target_", "")):
        return ""
    import reax  # pylint: disable=import-outside-toplevel
    from flax import nnx  # pylint: disable=import-outside-toplevel

    from e3response.data.qm9_nmr import Qm9NmrDataset  # pylint: disable=import-outside-toplevel

    data_dir = data_dir_override or d.get("data_dir")
    dataset = d.get("dataset", "gasphase")
    limit = d.get("limit")
    frac = tuple(d.get("train_val_test_split", (0.85, 0.05, 0.1)))
    key = (str(data_dir), str(dataset), str(limit), frac)
    if key not in _SPLIT_CACHE:
        n = Qm9NmrDataset.count(data_dir, dataset, limit)
        parts = reax.data.random_split(nnx.Rngs(0), dataset=range(n), lengths=frac)
        h = hashlib.md5()
        for name, p in zip(("train", "val", "test"), parts):
            h.update(name.encode())
            h.update(",".join(map(str, sorted(p.indices))).encode())
        _SPLIT_CACHE[key] = f"{n}:{h.hexdigest()[:10]}"
    return _SPLIT_CACHE[key]


def summarise(run, args):
    cfg = load_yaml(run / "config.yaml")
    if not cfg:
        return None
    kind, mul, ell, nl = model_info(cfg)
    loss = cfg.get("model", {}).get("loss_fn", {}) or {}
    fd = cfg.get("from_data", {}) or {}
    zs = fd.get("atomic_numbers") or []
    test = read_metrics(run)
    row = dict(
        run=str(run),
        data=str(cfg.get("data", {}).get("_target_", "")).split(".")[-1],
        model=kind,
        r_max=cfg.get("data", {}).get("r_max", cfg.get("r_max")),
        ell_max=ell, mul=mul, num_layers=nl,
        w_energy=_weight(loss.get("energy")), w_nmr=_weight(loss.get("nmr_tensors")),
        w_induced_magnetic_field=_weight(loss.get("induced_magnetic_field")),
        n_elements=fd.get("n_elements"), has_Z0=(0 in zs),
        limit=cfg.get("data", {}).get("limit"), batch_size=cfg.get("data", {}).get("batch_size"),
        status=run_status(run, test),
    )
    if args.no_split_hash:
        row["split_hash"] = ""
    else:
        try:
            row["split_hash"] = split_hash(cfg, args.data_dir)
        except Exception as exc:  # pylint: disable=broad-except
            row["split_hash"] = f"ERR:{type(exc).__name__}"
    row.update(test)
    if "test/energy_rmse" in test:  # Hartree -> kcal/mol
        row["test_energy_rmse_kcal"] = test["test/energy_rmse"] * HARTREE_TO_KCAL
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", default="summary.csv")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--no-split-hash", action="store_true")
    args = ap.parse_args()

    rows = [r for r in (summarise(run, args) for run in find_runs(args.roots)) if r]
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    out = pathlib.Path(args.out)
    complete = [r for r in rows if r["status"] == "complete"]
    incomplete = [r for r in rows if r["status"] != "complete"]
    for name, part in ((out, complete), (out.with_name(out.stem + "_incomplete.csv"), incomplete)):
        with open(name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, restval="")
            w.writeheader()
            w.writerows(part)
    empty = [c for c in cols if all(r.get(c) in (None, "") for r in complete)] if complete else []
    print(f"{len(complete)} complete -> {out}; {len(incomplete)} incomplete -> *_incomplete.csv")
    if empty:
        print("columns empty in all complete runs:", empty)
    hashes = {r["split_hash"] for r in complete if r.get("split_hash")}
    print(f"distinct split hashes among complete runs: {len(hashes)} {sorted(hashes)[:5]}")


if __name__ == "__main__":
    main()
