"""Write the 30 bulk test cells for the MACE relaxations, which run in another conda env."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE + "/sr")
from common import *

ds = dataset(); ids = test_ids(ds)
bulk = [i for i in ids if ds.structures[i].info["struct_type"] == "bulk"]
out = {}
for c in bulk:
    a = ds.structures[c]
    out[f"pos_{c}"] = a.positions; out[f"z_{c}"] = a.numbers; out[f"cell_{c}"] = np.array(a.cell)
    out[f"k_{c}"] = int(np.where(a.numbers == 11)[0][0])
np.savez(f"{HERE}/cavity/cells.npz", cids=np.array(bulk), **out)
print(len(bulk), "cells")
