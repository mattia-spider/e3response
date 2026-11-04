"""Do the summary tables change after fixing the overflowed points?"""
import glob, os, numpy as np
from d1_analyse import load, evaluate, variant, coarse_only, five_fold, geometry_only, chi2, w1, mic, ok
NCS, NCSO, NC = (11, 20, 14), (11, 20, 14, 8), (11, 20)
ROWS = [("geometry", geometry_only("soft")), ("chi2 NaCaSi", variant("soft")), ("NaCaSi loose", variant("loose")), ("NaCaSi tight", variant("tight")),
        ("chi2 NaCa", variant("soft", sp=NC)), ("Na", variant("soft", sp=(11,))), ("NaSi", variant("soft", sp=(11, 14))), ("CaSi", variant("soft", sp=(20, 14))),
        ("chi2 NaCaSiO", variant("soft", sp=NCSO)), ("all", variant("soft", sp=(11, 20, 14, 8, 1))), ("W1", variant("soft", w1)),
        ("coarse NaCaSi", coarse_only("soft", chi2)), ("coarse NaCa", coarse_only("soft", chi2, NC)), ("coarse W1", coarse_only("soft", w1)), ("coarse W1 NaCa", coarse_only("soft", w1, NC)),
        ("ens5", five_fold("soft")), ("self", variant("soft", tgt="self")), ("heldout", variant("soft", tgt="heldout")),
        ("fold0", variant("soft", fold=0)), ("fold1", variant("soft", fold=1)), ("fold2", variant("soft", fold=2))]
for model in ("direct", "deriv1st"):
    names = sorted(os.path.basename(f) for f in glob.glob(f"sweep_{model}/cell_*.npz"))
    N = [load(f"sweep_{model}/{n}") for n in names]; O = [load(f"sweep_{model}_v0/{n}") for n in names]
    print(model)
    for nm, fn in ROWS:
        a, ra = evaluate(N, fn); b, rb = evaluate(O, fn)
        t = lambda r: sum(1 for x in r if x and x <= 3)
        new = (np.sum(a < .5), np.sum(a < 1), t(ra), round(float(np.median(a)), 2)); old = (np.sum(b < .5), np.sum(b < 1), t(rb), round(float(np.median(b)), 2))
        changed = [names[i] for i in range(len(a)) if abs(a[i] - b[i]) > 1e-6]
        print(f"  {nm:16s} old {old}  new {new}  {'SAME' if old == new else 'CHANGED'}  cells moved: {[c[5:-4] for c in changed]}")
