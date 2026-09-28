#!/usr/bin/env python
"""Find the largest batch size a model fits into GPU memory on this machine.

Memory use is dominated by the derivative tape, so the differentiated NMR models need far
smaller batches than the direct one (roughly 10x the memory at the same batch size on QM9), and
how small depends on the card.  Run this once on a new machine before queueing anything, rather
than discovering the ceiling from a job that dies hours in.

    python scripts/probe_batch_size.py --model nequip_nmr_deriv2nd
    python scripts/probe_batch_size.py \\
        --model nequip_nmr --model nequip_nmr_deriv1st --model nequip_nmr_deriv2nd \\
        --mul 8,16,32 --ell-max 2,3 --num-layers 2,3,4
    python scripts/probe_batch_size.py --data si_nmr --model nequip_nmr --subset all

What to probe
-------------
The ceiling depends on the molecules as much as on the model: a batch of large molecules needs
more memory than a batch of small ones.  So probe the subset you will actually train on.

--subset first    the first --n structures in the dataset's own order (default, --n 1000): the
                  same ones a plain `limit: N` trains on.  For QM9 that order is the archive's
                  sorted filenames, not molecule size -- the first 1000 are a contiguous block of
                  ids from ~107000, somewhat larger than QM9 on average and with no small
                  molecules, but also without its largest outliers.
--subset random   --n structures drawn at random from the whole dataset (seeded by --seed): the
                  dataset's real size distribution, largest outliers included, for the price of
                  --n structures.  The one to use if you will train on the full dataset.
--subset all      the whole dataset.  Needed for datasets that cannot be subset (e.g. bto).
--limit SPEC      any raw limit spec, overriding --subset.

The ceiling also depends on the stack: pass the same --mul, --ell-max and --num-layers as the
sweep will use.  Memory grows with all three, so by default only the largest combination is
probed -- the one that has to fit.  --all-configs probes every combination instead.

How it measures
---------------
Each candidate runs in a fresh subprocess, because a JAX process that has hit an out of memory
error cannot be trusted afterwards.  Candidates are tried in increasing order and the search
stops at the first failure, since what fits is monotonic in the batch size.  JAX preallocation
is switched off so the reported peak is what the model needs; the real runs should do the same
(XLA_PYTHON_CLIENT_PREALLOCATE=false), otherwise JAX hands them only part of the card.  The peak
is read from nvidia-smi and so covers the whole card: run it on an otherwise idle GPU.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time

_OOM = re.compile(r"RESOURCE_EXHAUSTED|out of memory|\bOOM\b", re.IGNORECASE)
_NO_LIMIT = re.compile(r"unexpected keyword argument '(limit|batch_size)'")


def _gpu_query(field: str) -> int:
    """Query the GPU this job runs on through nvidia-smi, returning 0 if that is not possible."""
    if not shutil.which("nvidia-smi"):
        return 0
    base = ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"]
    # On a shared node nvidia-smi can list every GPU, not just the one the scheduler assigned,
    # so ask for that one.  Where cgroups hide the others and renumber it, the explicit index
    # may not resolve -- then the plain query sees only this job's GPU anyway.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    attempts = [base + ["-i", visible], base] if visible and visible != "NoDevFiles" else [base]
    for cmd in attempts:
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
            if res.returncode == 0 and res.stdout.strip():
                return int(res.stdout.strip().splitlines()[0])
        except Exception:  # pylint: disable=broad-except
            pass
    return 0


def _sample_peak(stop: threading.Event, peak_out: list) -> None:
    """Poll memory in use until `stop`, recording the high water mark in MiB."""
    peak = 0
    while not stop.is_set():
        peak = max(peak, _gpu_query("memory.used"))  # a failed sample must not stop the probe
        time.sleep(0.25)
    peak_out.append(peak)


def _error_line(output: str) -> str:
    """Pick the line that actually says what went wrong out of a failed run's output."""
    lines = [ln.strip() for ln in output.splitlines()
             if ln.strip() and "HYDRA_FULL_ERROR" not in ln and "it/s" not in ln]
    # A Python traceback ends with the exception itself
    raised = [ln for ln in lines if re.match(r"[\w.]*(Error|Exception)\b", ln)]
    if raised:
        return raised[-1]
    # Hydra reports config problems (bad override, missing key) without a traceback
    hydra = [ln for ln in lines if ln.startswith(("Could not", "Error executing", "Key '"))]
    return (hydra or lines or ["?"])[0]


def _limit_spec(args) -> str | None:
    if args.limit is not None:
        return args.limit
    if args.subset == "first":
        return str(args.n)
    if args.subset == "random":
        return f"random:{args.n}:{args.seed}"
    return None


def _supports_limit(args) -> bool | None:
    """Whether the datamodule takes a `limit`, or None if that cannot be determined.

    Checked in a CPU-only child so that importing JAX here never touches the GPU the probes use.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(args.config)), "data", f"{args.data}.yaml")
    if not os.path.isfile(path):
        return None
    code = (
        "import inspect, hydra.utils, omegaconf;"
        f"c = omegaconf.OmegaConf.load({path!r});"
        "print('limit' in inspect.signature(hydra.utils.get_class(c._target_).__init__).parameters)"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=dict(os.environ, JAX_PLATFORMS="cpu"), check=False)
    out = res.stdout.strip().splitlines()
    return {"True": True, "False": False}.get(out[-1] if out else "")


def _probe(args, model: str, batch_size: int, stack: dict, limit: str | None) -> tuple[str, int]:
    cmd = [
        sys.executable, "-m", "e3response.cli", "train", "-i", args.config,
        f"data={args.data}", f"model={model}", f"trainer={args.trainer}",
        "logger=csv", "listeners=none", "extras.print_config=false", "test=false",
        f"paths.root_dir={args.out}", f"paths.data_dir={os.path.abspath(args.data_dir)}",
        # ++ adds the key if the dataset's config does not declare it, overrides it otherwise
        f"++data.batch_size={batch_size}",
        "train.min_epochs=1", "train.max_epochs=1",
    ]
    if limit is not None:
        cmd.append(f"++data.limit={limit}")
    # The stack keys are only read through ${oc.select:...} by the model configs, so they are not
    # in the base config and need ++ as well
    cmd += [f"++{k}={v}" for k, v in stack.items()] + args.overrides

    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
    stop, peaks = threading.Event(), []
    sampler = threading.Thread(target=_sample_peak, args=(stop, peaks), daemon=True)
    sampler.start()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              timeout=args.timeout, check=False)
        output, code = proc.stdout + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        output, code = "", None
    finally:
        stop.set()
        sampler.join(timeout=2)

    peak = peaks[0] if peaks else 0
    if code is None:
        return f"timeout (>{args.timeout}s)", peak
    if _OOM.search(output):
        return "OOM", peak
    if code != 0:
        # Anything else is a genuine error and must not be read as a memory ceiling
        return f"FAILED: {_error_line(output)[:70]}", peak
    return "ok", peak


def main() -> int:
    par = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    par.add_argument("--model", action="append", required=True,
                     help="model config name; repeat to probe several")
    par.add_argument("--data", default="qm9_nmr", help="data config name")
    par.add_argument("--subset", choices=("first", "random", "all"), default="first")
    par.add_argument("--n", type=int, default=1000, help="structures for --subset first/random")
    par.add_argument("--seed", type=int, default=0, help="seed for --subset random")
    par.add_argument("--limit", default=None, help="raw limit spec, overrides --subset")
    par.add_argument("--sizes", default="8,16,32,64")
    par.add_argument("--mul", default="8")
    par.add_argument("--ell-max", default="2")
    par.add_argument("--num-layers", default="3")
    par.add_argument("--all-configs", action="store_true",
                     help="probe every (mul, ell_max, num_layers), not just the largest")
    par.add_argument("--config", default="configs/train.yaml")
    par.add_argument("--data-dir", default="./data")
    par.add_argument("--trainer", default="gpu")
    par.add_argument("--out", default="/tmp/bs_probe")
    par.add_argument("--timeout", type=int, default=3600)
    par.add_argument("overrides", nargs="*", default=[],
                     help="extra hydra overrides passed to every run")
    args = par.parse_args()

    limit = _limit_spec(args)
    if limit is not None and _supports_limit(args) is False:
        print(f"The '{args.data}' datamodule takes no `limit`, so it cannot be subset: rerun "
              f"with --subset all.")
        return 2

    sizes = sorted(int(s) for s in args.sizes.split(","))
    muls = sorted(int(v) for v in args.mul.split(","))
    ells = sorted(int(v) for v in args.ell_max.split(","))
    layers = sorted(int(v) for v in args.num_layers.split(","))
    if args.all_configs:
        stacks = [{"mul": m, "ell_max": e, "num_layers": n}
                  for m in muls for e in ells for n in layers]
    else:
        stacks = [{"mul": muls[-1], "ell_max": ells[-1], "num_layers": layers[-1]}]

    total = _gpu_query("memory.total")
    print(f"data={args.data}  subset={'limit ' + limit if limit else 'all'}  "
          f"card={total or '?'} MiB")
    if not args.all_configs:
        print(f"Probing the largest stack only (mul={muls[-1]} ell_max={ells[-1]} "
              f"num_layers={layers[-1]}); pass the sweep's --mul/--ell-max/--num-layers, "
              f"or --all-configs for every combination")

    results: dict[tuple, int | None] = {}
    for model in args.model:
        for stack in stacks:
            print(f"\n=== {model} [mul={stack['mul']} ell_max={stack['ell_max']} "
                  f"layers={stack['num_layers']}] ===", flush=True)
            largest, largest_peak = None, 0
            for bs in sizes:
                status, peak = _probe(args, model, bs, stack, limit)
                print(f"  batch_size={bs:<5} {status:<28} "
                      f"{f'{peak} MiB peak' if peak else 'peak unknown'}", flush=True)
                if _NO_LIMIT.search(status):
                    print("  (this datamodule cannot take that argument; try --subset all)")
                if status != "ok":
                    break  # memory use is monotonic, so nothing larger can fit
                largest, largest_peak = bs, peak
            results[(model, *stack.values())] = largest
            if largest and total and largest_peak > 0.75 * total:
                print(f"  NOTE: {largest_peak} MiB is over 75% of the card: under JAX's default "
                      f"preallocation this would OOM, so set XLA_PYTHON_CLIENT_PREALLOCATE=false "
                      f"for the real runs")

    print("\n=== largest batch size that fits ===")
    for (model, mul, ell, nl), largest in results.items():
        print(f"  {model:<26} mul={mul:<3} ell_max={ell} layers={nl}  -> "
              f"{largest if largest else 'none of ' + args.sizes}")
    fitting = [v for v in results.values() if v]
    if len(results) > 1 and len(fitting) == len(results):
        print(f"\nUse {min(fitting)} for all of them: comparing architectures at different "
              f"batch sizes confounds the comparison with gradient noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
