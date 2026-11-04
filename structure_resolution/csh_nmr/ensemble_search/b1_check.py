"""Do the new checkpoints work through the branch's live neighbour list, do they match the saved
parity predictions, how fast are they, and how big is the TF32 rounding in ppm?
Run twice: default and JAX_DEFAULT_MATMUL_PRECISION=highest."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/sr")
from common import *

tag = sys.argv[1]
ds = dataset(); ids = test_ids(ds)
iso = lambda t: np.trace(t, axis1=-2, axis2=-1) / 3
for model in ("direct", "deriv1st", "deriv2nd"):
    module, params = load(model, 0)
    d = np.load(f"/home/mattia/Desktop/ml_codes/nmr_project/jz_runs/parity/q1_csh/{TAG[model]}_f0.npz")
    i = ids[0]; a = ds.structures[i]
    k = int(np.where(a.numbers == 11)[0][0])
    one, cap = make_scan(module, params, a, k)
    f = jax.jit(one)
    pk = jnp.asarray(a.positions[k], dtype=jnp.float32)
    t0 = time.time(); p = np.asarray(f(pk, params)); t1 = time.time(); p = np.asarray(f(pk, params)); t2 = time.time()
    m = d["test/nmr_tensors/struct"] == i
    saved = d["test/nmr_tensors/pred"][m]; z = d["test/nmr_tensors/species"][m]
    dif = np.abs(iso(p) - iso(saved))
    print(f"{model}: n={len(a)} cap={cap} compile {t1-t0:.1f}s eval {t2-t1:.3f}s |iso-saved| max {dif.max():.3f} "
          f"Na {dif[z==11].max():.3f} Ca {dif[z==20].mean():.3f} Si {dif[z==14].mean():.3f} rmse-to-label {np.sqrt(np.mean((p-d['test/nmr_tensors/target'][m])**2)):.2f}", flush=True)
    # jitter: 1 mA moves of the Na; a smooth function changes by S*1e-3 only
    rng = np.random.default_rng(0)
    ps = np.array([np.asarray(f(pk + jnp.asarray(rng.normal(size=3) * 1e-3, dtype=jnp.float32), params)) for _ in range(16)])
    print(f"   std over 1 mA jitter: Na {iso(ps)[:, a.numbers==11].std(0).mean():.3f} Ca {iso(ps)[:, a.numbers==20].std(0).mean():.3f} Si {iso(ps)[:, a.numbers==14].std(0).mean():.3f}", flush=True)
    gf = jax.jit(jax.value_and_grad(lambda x, q: jnp.sum(jnp.trace(one(x, q)[np.where(a.numbers == 20)[0]], axis1=-2, axis2=-1))))
    t0 = time.time()
    try:
        gf(pk, params)[1].block_until_ready(); t1 = time.time(); gf(pk, params)[1].block_until_ready(); print(f"   grad compile {t1-t0:.1f}s eval {time.time()-t1:.3f}s", flush=True)
    except Exception as e:
        print("   grad failed", type(e).__name__, str(e)[:120], flush=True)
    del f, gf, one
    jax.clear_caches()
