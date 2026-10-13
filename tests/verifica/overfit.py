"""Test 1 of REPORT_muB.md: over-fit N molecules, tensors only, identical for every model.

Same molecules (used for training and evaluation), same init key, optimiser and schedule; only the
model config changes.  Float32 like real training (E3R_X64=0).

    E3R_X64=0 python tests/verifica/overfit.py --model deriv2nd_muB --steps 3000 --out overfit_deriv2nd_muB.json
"""
import argparse, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import *  # noqa
import jraph, optax
from e3response.data.qm9_nmr import Qm9NmrDataset

SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS))
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--n-mol", type=int, default=20)
ap.add_argument("--mul", type=int, default=16); ap.add_argument("--ell", type=int, default=2)
ap.add_argument("--layers", type=int, default=2); ap.add_argument("--r-max", type=float, default=5.0)
ap.add_argument("--lr", type=float, default=1e-2)
ap.add_argument("--muB-scale", type=float, default=None)
ap.add_argument("--sample", default="random:20:0", help="Qm9NmrDataset limit spec of the molecules")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--log-every", type=int, default=100)
ap.add_argument("--out", default=None)
a = ap.parse_args()

ds = Qm9NmrDataset(r_max=a.r_max, data_dir=str(ROOT / "data/qm9_nmr"), limit=a.sample)
graphs = [ds[i] for i in range(len(ds))]
batch = jraph.batch(graphs)
tgt = jnp.asarray(batch.nodes["nmr_tensors"]); Z = np.asarray(batch.nodes["atomic_numbers"])[:, 0]
avg_nn = float(len(batch.senders) / len(Z))
extra = {"muB_scale": a.muB_scale} if a.muB_scale is not None else {}
model, _ = build_model(a.model, mul=a.mul, ell_max=a.ell, num_layers=a.layers, r_max=a.r_max, avg_nn=avg_nn, **extra)
params = init(model, batch, a.seed)
sched = optax.cosine_decay_schedule(a.lr, a.steps, alpha=0.01)
opt = optax.adam(sched); state = opt.init(params)

def loss_fn(p):
    pred = model.apply(p, batch).nodes[NMR]
    return jnp.mean((pred - tgt) ** 2), pred

@jax.jit
def step(p, s):
    (l, _), g = jax.value_and_grad(loss_fn, has_aux=True)(p)
    u, s = opt.update(g, s, p)
    return optax.apply_updates(p, u), s, l

t0 = time.time(); curve = []
for i in range(a.steps):
    params, state, l = step(params, state)
    if i == 0:
        print(f"first step (incl. compile) {time.time()-t0:.0f}s", flush=True)
    if i % a.log_every == 0 or i == a.steps - 1:
        curve.append((i, float(jnp.sqrt(l)))); print(f"step {i} rmse={float(jnp.sqrt(l)):.4g} ({time.time()-t0:.0f}s)", flush=True)

_, pred = loss_fn(params)
pred, t = np.asarray(pred, float), np.asarray(tgt, float)
res = dict(model=a.model, steps=a.steps, mul=a.mul, ell_max=a.ell, num_layers=a.layers, r_max=a.r_max,
           muB_scale=a.muB_scale, lr=a.lr, n_mol=len(graphs), n_atoms=int(len(Z)), seconds=time.time() - t0,
           rmse_total=float(np.sqrt(np.mean((pred - t) ** 2))), target_rms=float(np.sqrt(np.mean(t ** 2))),
           curve_rmse=curve, rmse_per_element={})
for z in np.unique(Z):
    m = Z == z
    res["rmse_per_element"][SYM[int(z)]] = dict(n=int(m.sum()), rmse=float(np.sqrt(np.mean((pred[m] - t[m]) ** 2))),
                                               target_rms=float(np.sqrt(np.mean(t[m] ** 2))))
print(json.dumps({k: v for k, v in res.items() if k != "curve_rmse"}, indent=1))
if a.out: json.dump(res, open(a.out, "w"), indent=1)
