"""B1-B6: numerical tests of the "no mu x B -> 0e path" hypothesis, random weights, no training.

For every (model, ell_max, mul, layers, seed) it reports, relative to the largest |sigma| the same
network gives on the generic non-planar molecule:

  B1 isolated atom, B2 linear molecules, B3 planar molecules (normal = z), B4 generic molecule,
  B5 equivariance under a random rotation + inversion invariance, B6 autodiff vs finite differences.

Usage:  JAX_PLATFORMS=cpu python tests/verifica/b_tests.py [--quick] [--out results_B.json]
"""

import argparse
import json
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import *  # noqa: E402,F401,F403


def rel(x, ref):
    return float(np.abs(x).max() / ref) if ref > 0 else float("nan")


def run_config(name, mul, ell_max, num_layers, seed, geos):
    model, _ = build_model(name, mul=mul, ell_max=ell_max, num_layers=num_layers)
    res = {}
    # parameters are independent of the graph (species embedding is a one-hot), init on generic
    gn, gp, _ = geos["CH3F_ethanol_like"]
    g_gen = make_graph(gn, gp)
    params = init(model, g_gen, seed)

    s_gen = sigma(model, params, g_gen)
    ref = float(np.abs(s_gen).max())
    res["ref_max_generic"] = ref

    # B4: every one of the 9 components non-zero (max over atoms of |sigma_ij|, relative)
    comp = np.abs(s_gen).max(0) / ref
    res["B4_min_component_rel"] = float(comp.min())
    res["B4_comp_rel"] = comp.round(4).tolist()
    # symmetric / antisymmetric split, to see whether l=1 is generated at all
    asym = 0.5 * (s_gen - s_gen.transpose(0, 2, 1))
    res["B4_antisym_over_total"] = float(np.linalg.norm(asym) / np.linalg.norm(s_gen))

    # B1 atom
    n, p, _ = geos["C_atom"]
    g = make_graph(n, p)
    res["B1_atom_rel"] = rel(sigma(model, params, g), ref)
    res["B1_atom_abs"] = float(np.abs(sigma(model, params, g)).max())

    # B2 linear
    for k in ("HCN", "C2H2", "CO2"):
        n, p, _ = geos[k]
        s = sigma(model, params, make_graph(n, p))
        res[f"B2_{k}_rel"] = rel(s, ref)
        res[f"B2_{k}_abs"] = float(np.abs(s).max())

    # B3 planar (normal z): xy block vs components touching z
    for k in ("H2CO", "C6H6"):
        n, p, _ = geos[k]
        s = sigma(model, params, make_graph(n, to_normal_z(p)))
        res[f"B3_{k}_xy_rel"] = rel(s[:, :2, :2], ref)
        res[f"B3_{k}_xz_yz_rel"] = rel(s[:, :2, 2], ref)
        res[f"B3_{k}_zx_zy_rel"] = rel(s[:, 2, :2], ref)
        res[f"B3_{k}_zz_rel"] = rel(s[:, 2, 2], ref)

    # B5 equivariance, inversion
    errs, inv = [], []
    s0 = s_gen
    R = rotation(seed + 100)
    s_rot = sigma(model, params, make_graph(gn, gp @ R.T if isinstance(gp, np.ndarray) else np.asarray(gp) @ R.T))
    errs = np.abs(s_rot - np.einsum("ij,njk,lk->nil", R, s0, R)).max()
    s_inv = sigma(model, params, make_graph(gn, -np.asarray(gp)))
    res["B5_equiv_abs_err"] = float(errs)
    res["B5_equiv_rel_err"] = float(errs / ref)
    res["B5_inversion_abs_err"] = float(np.abs(s_inv - s0).max())
    res["B5_inversion_rel_err"] = float(np.abs(s_inv - s0).max() / ref)
    return res


def fd_check(name, mul, ell_max, num_layers, seed, geos, h=1e-3, atoms=(0, 3)):
    """B6: autodiff sigma of deriv2nd vs central 2nd-order finite differences of E(mu, B)."""
    model, _ = build_model(name, mul=mul, ell_max=ell_max, num_layers=num_layers)
    emodel, _ = build_model(
        name, mul=mul, ell_max=ell_max, num_layers=num_layers, energy_only=True
    )
    gn, gp, _ = geos["CH3F_ethanol_like"]
    g = make_graph(gn, gp)
    params = init(model, g, seed)
    s_ad = sigma(model, params, g)

    n = len(gn)

    def energy(mu, B):
        gg = g
        nodes = dict(gg.nodes)
        nodes["nuclear_magnetic_moment"] = jnp.asarray(mu)
        glob = dict(gg.globals)
        glob["external_magnetic_field"] = jnp.broadcast_to(jnp.asarray(B), (1, 3))
        gg = gg._replace(nodes=nodes, globals=glob)
        out = emodel.apply(params, gg)
        return float(jnp.sum(out.globals["predicted_energy"]))

    ref = np.abs(s_ad).max()
    err = 0.0
    for k in atoms:
        s_fd = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                def E(a, b):
                    mu = np.zeros((n, 3))
                    mu[k, i] = a * h
                    B = np.zeros(3)
                    B[j] = b * h
                    return energy(mu, B)

                s_fd[i, j] = (E(1, 1) - E(1, -1) - E(-1, 1) + E(-1, -1)) / (4 * h * h)
        err = max(err, np.abs(s_fd - s_ad[k]).max())
    return {"B6_max_abs_err": float(err), "B6_ref_max_sigma": float(ref), "B6_rel_err": float(err / ref)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results_B.json"))
    ap.add_argument("--models", nargs="+", default=["deriv2nd", "deriv1st", "direct"])
    ap.add_argument("--mul", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--ell", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    if a.quick:
        a.mul, a.layers, a.seeds = [4], [2], [0, 1]
    geos = geometries()
    rows = []
    for name in a.models:
        for ell in a.ell:
            for mul in a.mul:
                for nl in a.layers:
                    for seed in a.seeds:
                        t = time.time()
                        r = run_config(name, mul, ell, nl, seed, geos)
                        r.update(model=name, ell_max=ell, mul=mul, num_layers=nl, seed=seed)
                        rows.append(r)
                        json.dump({"rows": rows, "fd": []}, open(a.out, "w"), indent=1)
                        print(
                            f"{name:9s} l={ell} mul={mul} n={nl} s={seed} "
                            f"B1={r['B1_atom_rel']:.1e} HCN={r['B2_HCN_rel']:.1e} "
                            f"C2H2={r['B2_C2H2_rel']:.1e} H2CO(xy/z..)="
                            f"{r['B3_H2CO_xy_rel']:.1e}/{r['B3_H2CO_xz_yz_rel']:.1e}/"
                            f"{r['B3_H2CO_zx_zy_rel']:.1e}/{r['B3_H2CO_zz_rel']:.1e} "
                            f"B4min={r['B4_min_component_rel']:.1e} "
                            f"eq={r['B5_equiv_rel_err']:.1e} inv={r['B5_inversion_rel_err']:.1e} "
                            f"({time.time()-t:.0f}s)",
                            flush=True,
                        )
    fd = []
    for name in [m for m in a.models if m.startswith("deriv2nd")]:
        for ell in a.ell:
            for seed in a.seeds[:2]:
                r = fd_check(name, 4, ell, 2, seed, geos)
                r.update(model=name, ell_max=ell, seed=seed)
                fd.append(r)
                print("B6", r, flush=True)
    json.dump({"rows": rows, "fd": fd}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
