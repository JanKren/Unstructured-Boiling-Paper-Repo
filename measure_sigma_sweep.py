#!/usr/bin/env python3
"""Anisotropy of the sigma sweep at the latest snapshot common to all four runs.

The earlier pass measured ts1000 (t = 0.1 ms), which was as far as the slowest
run had got.  At that radius the deformation is barely developed and three of
the four runs returned the same axis-to-diagonal difference to four significant
figures, which is not a result so much as a statement that nothing had happened
yet.  ts3000 (t = 0.3 ms) is the first common snapshot at which the distortion
is established.

Two quantities, both on the liquid-side interface band (alpha > 0.5 with at
least one face-neighbour at alpha < 0.5), which is the mask the paper uses --
the 0.3 < alpha < 0.7 band straddles both phases and averages in near-zero
vapour gradients:

  shape     regress the interface radius on the cubic invariant
            s = nx^4 + ny^4 + nz^4 and convert to the axis-to-diagonal
            difference, (2/3) (dR/ds) / <R>, the measure the paper defines.

  gradient  fit the azimuthal Fourier modes of |grad T . n| by least squares
            to the scattered points.  Binning a one-cell equatorial slice
            aliases badly; m=2 and m=6 are the cubic-symmetry sanity check and
            must sit near zero.
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/jan/papers/UnstructuredBoiling")
from analyze_anisotropy_attribution import load_lattice, fit_bubble

ROOT = os.path.join(os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling"), "Data/campaign/SurfTension")
SIGMA_BASE = 0.059                      # N/m, the baseline used in the runs
RUNS = [("s025", 0.25), ("s100", 1.00), ("s200", 2.00), ("s400", 4.00)]


def fourier_lsq(theta, g, m_max=8):
    cols = [np.ones_like(theta)]
    for m in range(1, m_max + 1):
        cols += [np.cos(m * theta), np.sin(m * theta)]
    coef, *_ = np.linalg.lstsq(np.stack(cols, axis=1), g, rcond=None)
    a0 = coef[0]
    amp, pha = {}, {}
    for m in range(1, m_max + 1):
        c, s = coef[2 * m - 1], coef[2 * m]
        amp[m] = float(np.hypot(c, s) / abs(a0))
        pha[m] = float(np.degrees(np.arctan2(s, c)) / m) % (360.0 / m)
    return amp, pha


def measure(path):
    d = load_lattice(path)
    fit = fit_bubble(d)
    vof = d["vof"]

    # liquid cells with at least one vapour face-neighbour
    is_vap = vof < 0.5
    nb = np.zeros_like(is_vap)
    for ax in range(3):
        nb |= np.roll(is_vap, 1, ax) | np.roll(is_vap, -1, ax)
    mask = (~is_vap) & nb

    q = (fit["P"] - fit["ctr"])[mask]
    r = np.linalg.norm(q, axis=-1)
    n = q / r[:, None]
    s = (n ** 4).sum(-1)
    theta = np.arctan2(q[:, 1], q[:, 0])

    # shape: radius regressed on the cubic invariant
    A = np.vstack([np.ones_like(s), s]).T
    c0, c1 = np.linalg.lstsq(A, r, rcond=None)[0]
    axis_diag = (2.0 / 3.0) * c1 / r.mean()

    # gradient: azimuthal modes of the interface-normal gradient magnitude
    g = d["grad_saved"][mask]
    gn = np.abs((g * n).sum(-1))
    amp, pha = fourier_lsq(theta, gn)
    return dict(n=len(r), R=r.mean(), axis_diag=axis_diag,
                m4=amp[4], phi4=pha[4], m2=amp[2], m6=amp[6])


print(f"\n  sigma sweep, structured $75^3$, dt = 0.1 us, see header")
print(f"  {'run':<6s}{'sigma':>9s}{'x base':>8s}{'cells':>7s}{'<R> um':>9s}"
      f"{'axis-diag':>11s}{'m4 grad':>9s}{'phi4':>8s}{'m2':>7s}{'m6':>7s}")
print("  " + "-" * 82)
res = {}
for tag, fac in RUNS:
    ts = os.environ.get("TS", "003000")
    # the run also writes a front pvtu with the same timestep tag, which has
    # none of the cell fields; glob order is arbitrary, so exclude it explicitly
    f = [x for x in glob.glob(os.path.join(ROOT, tag, f"*-ts{ts}.pvtu"))
         if "front" not in os.path.basename(x)]
    if not f:
        print(f"  {tag:<6s}  (no ts3000 snapshot)")
        continue
    m = measure(f[0])
    res[tag] = m
    print(f"  {tag:<6s}{SIGMA_BASE*fac:9.5f}{fac:8.2f}{m['n']:7d}{m['R']*1e6:9.3f}"
          f"{100*m['axis_diag']:10.3f}%{100*m['m4']:8.3f}%{m['phi4']:8.2f}"
          f"{100*m['m2']:6.2f}%{100*m['m6']:6.2f}%")

if "s025" in res and "s400" in res:
    lo, hi = res["s025"], res["s400"]
    print(f"\n  sixteen-fold in sigma:  axis-to-diagonal "
          f"{100*lo['axis_diag']:+.3f}% -> {100*hi['axis_diag']:+.3f}%  "
          f"({100*(hi['axis_diag']/lo['axis_diag']-1):+.1f}% relative)")
    print(f"                          gradient m=4     "
          f"{100*lo['m4']:.3f}%  -> {100*hi['m4']:.3f}%   "
          f"({100*(hi['m4']/lo['m4']-1):+.1f}% relative)")
    print(f"                          mean radius      "
          f"{lo['R']*1e6:.3f} -> {hi['R']*1e6:.3f} um "
          f"({100*(hi['R']/lo['R']-1):+.2f}%)")
print()
