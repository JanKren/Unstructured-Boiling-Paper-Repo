#!/usr/bin/env python3
"""Four-fold mode of beta_f and gamma_f measured in a RUNNING simulation.

    python3 analyze_beta_m4_running.py [dump_dir]

Table 8 of the paper evaluates the four-fold content of the CICSAM blending
factor offline: on the exact volume fraction of a sphere, at a prescribed
Courant number of 0.2, on the structured 125^3 mesh. That is a controlled
evaluation and it is labelled as one. What it cannot say is whether the same
four-fold amplitude appears at the operating point of a real simulation, where
the Courant number varies, the mass flux is computed rather than prescribed, and
the interface is no longer a sphere.

This script closes that gap. `Predict_Beta` was instrumented to dump, at
selected timesteps, one line per owned interfacial face:

    x_face  y_face  z_face  gamma_f  beta_f

and the same azimuthal fit used for the exact-sphere evaluation is applied here.

WHY BOTH FIELDS ARE FITTED. beta_f is the product of three things: gamma_f (the
orientation weight), alpha_tilde_D (the ramp width, where Section 4 locates the
mechanism) and the CBC Courant bound. Fitting gamma_f alongside beta_f separates
them on the running state: if gamma_f carries much less four-fold content than
beta_f, the anisotropy demonstrably enters downstream of the blend, which is
what the frozen-gamma control argues indirectly.

METHOD. The modes are fitted by least squares to the scattered (azimuth, value)
points over every interfacial face, not by binning an equatorial slice -- a
one-cell slice holds too few faces and aliases badly, which is how an earlier
analysis produced an m = 8 of 151%. Cubic symmetry is the check: m = 2 and m = 6
should sit near zero while m = 4 carries the signal, and phi_4 should come out
at 45 degrees.
"""

import glob
import os
import re
import sys

import numpy as np

DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "Data", "BetaDump")


def load(pattern):
    """Concatenate the per-rank dumps for one timestep."""
    rows = []
    for f in sorted(glob.glob(pattern)):
        try:
            d = np.loadtxt(f)
        except Exception:
            continue
        if d.size:
            rows.append(np.atleast_2d(d))
    return np.vstack(rows) if rows else None


def fit_modes(phi, val, modes=(2, 4, 6, 8)):
    """Least-squares Fourier amplitudes about the mean, as percentages."""
    cols = [np.ones_like(phi)]
    for m in modes:
        cols += [np.cos(m * phi), np.sin(m * phi)]
    A = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(A, val, rcond=None)
    mean = c[0]
    out = {}
    for i, m in enumerate(modes):
        a, b = c[1 + 2 * i], c[2 + 2 * i]
        amp = np.hypot(a, b)
        pha = np.degrees(np.arctan2(b, a)) / m
        out[m] = (100.0 * amp / abs(mean), pha % (360.0 / m))
    r2 = 1.0 - (val - A @ c).var() / val.var()
    return mean, out, r2


def main(root):
    stamps = sorted({re.search(r"ts(\d+)", f).group(1)
                     for f in glob.glob(os.path.join(root, "beta_dump-ts*.dat"))})
    if not stamps:
        sys.exit(f"no dumps in {root}")

    print("Four-fold content measured in the running simulation")
    print("(offline exact-sphere reference: beta_f m=4 = 6.45%, phi_4 = 45.0 deg,")
    print(" at Co = 0.2 on the structured 125^3 mesh)\n")

    for ts in stamps:
        d = load(os.path.join(root, f"beta_dump-ts{ts}-p*.dat"))
        if d is None or len(d) < 500:
            print(f"ts {ts}: too few faces"); continue
        x, y, z, gam, beta = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4]
        cx, cy, cz = x.mean(), y.mean(), z.mean()
        phi = np.arctan2(y - cy, x - cx)

        print(f"timestep {int(ts)}   {len(d):,} interfacial faces   "
              f"centre ({cx*1e6:.1f}, {cy*1e6:.1f}, {cz*1e6:.1f}) um")
        for name, v in (("beta_f", beta), ("gamma_f", gam)):
            ok = v >= 0.0                      # gam = -1 marks a face never set
            mean, modes, r2 = fit_modes(phi[ok], v[ok])
            m4, p4 = modes[4]
            m2, _ = modes[2]
            m6, _ = modes[6]
            print(f"   {name:<8} mean {mean:6.4f}   m=4 {m4:6.2f}%  "
                  f"phi_4 {p4:5.2f} deg   (m=2 {m2:5.2f}%, m=6 {m6:5.2f}%)  R2 {r2:5.3f}")
        print()
    print("  Cubic symmetry requires m=2 and m=6 near zero; if they are not,")
    print("  the fit is contaminated and the m=4 should not be quoted.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT))
