#!/usr/bin/env python3
"""Does resolving the capillary time step remove the anisotropy?

Two run families on the SAME structured 75^3 mesh, same solver, same baseline
surface tension (0.059 N/m), differing only in the time step:

  aniso-075-off   dt = 2 us     -- the production step.  dt/dt_sigma = 6.8, so
                                   capillary waves are NOT resolved in time.
  SurfTension/s100  dt = 0.1 us -- 20x smaller, dt/dt_sigma < 1, so they are.

If the four-fold anisotropy were an artefact of an under-resolved capillary
restoring force, the 0.1 us family should show much less of it at the same
bubble radius.  If the paper's "halving dt halves m=4" relation continued down
to 0.1 us, a 20x reduction would leave almost nothing.

Compared at matched <R>, because m=4 is a function of R/h at fixed dt.
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/jan/papers/UnstructuredBoiling")
from analyze_anisotropy_attribution import load_lattice, fit_bubble

ROOT = os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling")
H = 4.05e-6


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
    A = np.vstack([np.ones_like(s), s]).T
    _, c1 = np.linalg.lstsq(A, r, rcond=None)[0]
    gn = np.abs((d["grad_saved"][mask] * n).sum(-1))
    amp, pha = fourier_lsq(theta, gn)
    return dict(R=r.mean(), axis_diag=(2 / 3) * c1 / r.mean(),
                m4=amp[4], phi4=pha[4], m2=amp[2], m6=amp[6])


FAMILIES = [
    ("dt = 2 us   (production, capillary UNresolved)", 2.0e-6,
     os.path.join(ROOT, "Data/Aniso/aniso-075-off"), [100, 200, 300]),
    ("dt = 0.1 us (capillary resolved)", 0.1e-6,
     os.path.join(ROOT, "Data/campaign/SurfTension/s100"), [1000, 2000, 3000]),
]

print(f"\n  Structured $75^3$, sigma = 0.059 N/m in both families, h = 4.05 um")
print(f"  {'family':<46s}{'t [ms]':>8s}{'<R> um':>9s}{'R/h':>7s}"
      f"{'axis-diag':>11s}{'m4 grad':>9s}{'phi4':>8s}{'m2/m6':>10s}")
print("  " + "-" * 108)
rows = []
for label, dt, d, tss in FAMILIES:
    for ts in tss:
        f = [x for x in glob.glob(os.path.join(d, f"*-ts{ts:06d}.pvtu"))
             if "front" not in os.path.basename(x)]
        if not f:
            print(f"  {label:<46s}  ts{ts}: missing")
            continue
        m = measure(f[0])
        rows.append((dt, m))
        print(f"  {label:<46s}{ts*dt*1e3:8.3f}{m['R']*1e6:9.3f}{m['R']/H:7.1f}"
              f"{100*m['axis_diag']:10.3f}%{100*m['m4']:8.3f}%{m['phi4']:8.2f}"
              f"{100*m['m2']:5.2f}/{100*m['m6']:.2f}%")
        label = ""

# interpolate the 2 us family onto the 0.1 us radii for a matched comparison
big = [(m['R'], m['m4'], m['axis_diag']) for dt, m in rows if dt > 1e-6]
sml = [(m['R'], m['m4'], m['axis_diag']) for dt, m in rows if dt < 1e-6]
if len(big) >= 2 and sml:
    bR = np.array([b[0] for b in big]); bm = np.array([b[1] for b in big])
    ba = np.array([b[2] for b in big])
    print("\n  matched-radius comparison (2 us family interpolated onto the 0.1 us radii)")
    print(f"  {'<R> um':>9s}{'m4 @2us':>10s}{'m4 @0.1us':>11s}{'ratio':>8s}"
          f"{'axdiag @2us':>13s}{'axdiag @0.1us':>15s}{'ratio':>8s}")
    print("  " + "-" * 76)
    for R, m4, ad in sml:
        if not (bR.min() <= R <= bR.max()):
            print(f"  {R*1e6:9.3f}   (outside the 2 us radius range, not compared)")
            continue
        m4b = float(np.interp(R, bR, bm)); adb = float(np.interp(R, bR, ba))
        print(f"  {R*1e6:9.3f}{100*m4b:9.3f}%{100*m4:10.3f}%{m4/m4b:8.2f}"
              f"{100*adb:12.3f}%{100*ad:14.3f}%{ad/adb:8.2f}")
print()
