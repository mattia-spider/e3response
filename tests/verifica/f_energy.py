"""F1/F2/F3: energy units, per-species baseline and the relative weight of the energy loss.

Samples N random molecules of the QM9-NMR gas-phase archive, fits the per-species baseline the way
``tensorial.gcnn.atomic.EnergyContributionLstsq(normalize=True)`` does (least squares on per-atom
normalised counts and energies) and, for comparison, the plain least squares on total energies,
then reports on a held out part:

* the MAE of the baseline-only energy prediction (Hartree and kcal/mol),
* the energy MSE [Ha^2] next to the tensor MSE [ppm^2] of a zero prediction (= mean sigma^2) and
  of a per-species-mean prediction, i.e. the two terms of ``response_loss(energy=True,
  nmr_tensors=True)`` whose weights are both 1.0.
"""
import argparse, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import *  # noqa
from e3response.data.qm9_nmr import Qm9NmrDataset

HA2KCAL = 627.5094740631

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--out", default=str(HERE / "results_F.json"))
a = ap.parse_args()

ds = Qm9NmrDataset(r_max=2.0, data_dir=str(ROOT / "data/qm9_nmr"), limit=f"random:{a.n}:0")
Z = ATOMIC_NUMBERS
counts, E, tens, zs = [], [], [], []
for i in range(len(ds)):
    g = ds[i]
    z = np.asarray(g.nodes["atomic_numbers"])[:, 0]
    counts.append([(z == k).sum() for k in Z]); zs.append(z)
    E.append(float(np.asarray(g.globals["energy"]).reshape(-1)[0]))
    tens.append(np.asarray(g.nodes["nmr_tensors"]))
C = np.array(counts, float); E = np.array(E); N = C.sum(1)
rng = np.random.default_rng(0); perm = rng.permutation(len(E)); ntr = int(0.8 * len(E))
tr, te = perm[:ntr], perm[ntr:]

def fit(Cm, Em): return np.linalg.lstsq(Cm, Em, rcond=None)[0]
s_norm = fit(C[tr] / N[tr, None], E[tr] / N[tr])      # EnergyContributionLstsq(normalize=True)
s_plain = fit(C[tr], E[tr])
res = {"n_molecules": len(E), "energy_range_Ha": [E.min(), E.max()], "energy_mean_Ha": E.mean()}
for name, s, pred in (("normalized", s_norm, (C[te] @ s_norm)), ("plain", s_plain, C[te] @ s_plain)):
    err = pred - E[te]
    res[f"baseline_{name}"] = dict(shifts_Ha=dict(zip(map(int, Z), s.tolist())),
        MAE_Ha=float(np.abs(err).mean()), MAE_kcal=float(np.abs(err).mean() * HA2KCAL),
        MSE_Ha2=float((err**2).mean()), RMSE_kcal=float(np.sqrt((err**2).mean()) * HA2KCAL))
# the loss terms: energy MSE per molecule (mean over molecules) vs tensor MSE (mean over atoms of a
# molecule, then over molecules, then over the 3x3 components)
mse_zero = np.mean([np.mean(tens[i] ** 2) for i in te])
means = {k: np.mean(np.concatenate([tens[i][zs[i] == k] for i in tr if (zs[i] == k).any()]), axis=0) for k in Z}
mse_mean = np.mean([np.mean((tens[i] - np.stack([means[k] for k in zs[i]])) ** 2) for i in te])
res["tensor_MSE_ppm2"] = dict(zero_prediction=float(mse_zero), per_species_mean=float(mse_mean))
res["energy_MSE_Ha2_baseline"] = res["baseline_normalized"]["MSE_Ha2"]
res["ratio_energy_term_over_tensor_term_at_baseline"] = res["energy_MSE_Ha2_baseline"] / mse_mean
json.dump(res, open(a.out, "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
