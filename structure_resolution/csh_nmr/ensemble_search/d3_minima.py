"""minima() from d3_calibration.py, which runs on import."""
import numpy as np
from d1_analyse import mic


def minima(pts, s, cell, sep=1.0):
    """Well-separated minima, best first: (position, chi2)."""
    out = []
    for i in np.argsort(s):
        if not np.isfinite(s[i]):
            break
        if all(mic(pts[i], p, cell) > sep for p, _ in out):
            out.append((pts[i], s[i]))
    return out
