"""Relax each cell with MACE-MP, fixed cell, with and without the Na being placed.

with    = control: how far MACE itself moves a DFT structure.
without = the framework as it would be with no Na in it, so the cavity is no longer built around the true site.
Rigid drift is removed. Same order of atoms as the input; without keeps the Na row at its DFT position.
"""
import os, sys, time, numpy as np
from ase import Atoms
from ase.optimize import LBFGS
from mace.calculators import mace_mp

HERE = os.path.dirname(os.path.abspath(__file__))
FMAX, STEPS = 0.03, 400
D = np.load(f"{HERE}/cavity/cells.npz")
calc = mace_mp(model="medium", device=sys.argv[1], default_dtype="float64")


def relax(a):
    a.calc = calc
    opt = LBFGS(a, logfile=None)
    opt.run(fmax=FMAX, steps=STEPS)
    f = np.abs(a.get_forces()).max()
    return a.positions.copy(), opt.get_number_of_steps(), f


for c in D["cids"]:
    f = f"{HERE}/cavity/relaxed_{c}.npz"
    if os.path.exists(f):
        continue
    t0 = time.time()
    pos, z, cell, k = D[f"pos_{c}"], D[f"z_{c}"], D[f"cell_{c}"], int(D[f"k_{c}"])
    keep = np.arange(len(z)) != k
    pw, nw, fw = relax(Atoms(numbers=z, positions=pos, cell=cell, pbc=True))
    pn, nn, fn = relax(Atoms(numbers=z[keep], positions=pos[keep], cell=cell, pbc=True))
    pw = pw - (pw[keep] - pos[keep]).mean(0)
    pn = pn - (pn - pos[keep]).mean(0)
    full = pos.copy(); full[keep] = pn
    np.savez(f, with_na=pw, without_na=full, steps=(nw, nn), fmax=(fw, fn))
    print(f"cell {c}: n={len(z)} with {nw} steps f {fw:.3f}, without {nn} steps f {fn:.3f}, rmsd with {np.sqrt(((pw-pos)[keep]**2).sum(1).mean()):.3f} without {np.sqrt(((full-pos)[keep]**2).sum(1).mean()):.3f}, Na moved {np.linalg.norm(pw[k]-pos[k]):.2f}, {time.time()-t0:.0f}s", flush=True)
