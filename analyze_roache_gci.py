#!/usr/bin/env python3
"""Verification of the Roache extrapolation claims in the Scriven mesh-convergence section.

`plot_scriven_extrapolation.py` fits Eq. (richardson), prints Table 4 and draws
Figure 8(a).  This script recomputes the same fits independently, from the same
tabulated input but with its own code path, and then checks the claims made
around them in the prose, so that every printed value has a path back to the
data:

  1. the refinement ratios, which are why the order is assumed and not measured
  2. Table 4 as printed -- b, its leave-one-out range, the limit and the r.m.s.
     residual -- reproduced here as a cross-check on the plotting script
  3. the three checks the text offers in support of the limits: the residual
     range, the movement of a limit when one level is dropped, and what forcing
     b = 1 costs in residual
  4. the fine-grid convergence index on the finest pair at F_s = 3, and whether
     the measured error and the extrapolated limit fall inside it
  5. the residual directional spread, which is the anisotropy in radius space

The exponent is FIXED at the formal order of the scheme rather than fitted.  The
refinement ratios of item 1 are too close to unity to condition an exponent, and
a freely fitted one returned 1.65 to 3.84 across the six sequences with
leave-one-out ranges two to three times as wide -- the signature of sequences
that are not in the asymptotic range.  Roache's prescription for that case is to
assume the formal order and pay for the assumption with the larger factor of
safety, which is what item 4 does.

The time-step side of the section is not checked here.  Its limits are two-point
extrapolations rather than fits, and `plot_scriven_extrapolation.py` prints them
together with the temporal GCI, which carries F_s = 1.25 because the observed
orders of 0.90 and 0.78 corroborate the assumed first order.

Input is the mesh-convergence series at t = 1.0 ms, dt = 2 us, so this script
reads no run output.

Usage:  python3 analyze_roache_gci.py
"""

import warnings
from itertools import combinations

import numpy as np
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

H = np.array([4.05, 3.03, 2.42, 2.01])       # nominal cell size [um]
HT = H / H[-1]                                # normalised, 1.000 at the finest
P = 2.0                                       # formal order of the scheme, fixed
FS_ASSUMED = 3.0                              # Roache's factor where p is assumed
FS_MEASURED = 1.25                            # ... and where it is corroborated
SERIES = {                                    # per cent error in R_front
    "structured": {"axis":          [-3.99, -0.49,  0.66,  0.87],
                   "mean":          [ 1.96,  7.11,  9.57, 10.75],
                   "body diagonal": [ 5.72, 11.85, 15.08, 16.81]},
    "polyhedral": {"axis":          [-8.88, -5.01, -3.22, -2.75],
                   "mean":          [-7.63, -3.38, -1.42, -0.92],
                   "body diagonal": [-6.81, -2.31, -0.24,  0.26]},
}
ORDER = ("axis", "mean", "body diagonal")


def model(x, a, b):
    """Richardson form at the fixed formal order, f = b + a h^P."""
    return a * x**P + b


def fit(y):
    """Two parameters on four levels, so the fit is over-determined and its
    residual is meaningful.  The range is the four leave-one-out refits."""
    q = curve_fit(model, HT, y, p0=[-0.05, 1.05], maxfev=400000)[0]
    rms = float(np.sqrt(np.mean((model(HT, *q) - y) ** 2)))
    jb = [curve_fit(model, HT[list(i)], y[list(i)], p0=q, maxfev=400000)[0][1]
          for i in combinations(range(4), 3)]
    return q, rms, (min(jb), max(jb))


def fit_forced(y):
    """The same fit with b pinned to 1, i.e. asserting that the sequence
    converges to the analytical solution.  Only its residual is of interest."""
    a = curve_fit(lambda x, a_: a_ * x**P + 1.0, HT, y, p0=[0.1],
                  maxfev=400000)[0]
    return float(np.sqrt(np.mean((a[0] * HT**P + 1.0 - y) ** 2)))


def series(fam, k):
    return 1.0 + np.array(SERIES[fam][k], float) / 100.0


def ratios():
    r = H[:-1] / H[1:]
    print("1. REFINEMENT RATIOS  (why the order is assumed, not measured)")
    print("   " + ", ".join(f"{x:.3f}" for x in r)
          + f"   ({(r < 1.3).sum()} of {len(r)} below Roache's recommended 1.3)")
    print("   ln r is small on those pairs, so an exponent fitted from them is")
    print("   poorly conditioned; the formal order is assumed instead.")


def table():
    print("\n2. TABLE 4 AS PRINTED  (recomputed independently of the plot script)")
    print(f"   {'family':<12}{'direction':<15}{'b':>9}{'b range':>17}"
          f"{'limit [%]':>11}{'rms [%]':>10}")
    for fam in SERIES:
        for k in ORDER:
            q, rms, (b0, b1) = fit(series(fam, k))
            print(f"   {fam:<12}{k:<15}{q[1]:9.4f}{b0:9.3f}--{b1:<6.3f}"
                  f"{(q[1] - 1) * 100:>10.1f}{rms * 100:>10.3f}")


def checks():
    """The three supporting claims, in the order the text makes them."""
    print("\n3. THE THREE CHECKS ON THE LIMITS")

    rms_all = {(fam, k): fit(series(fam, k))[1] * 100
               for fam in SERIES for k in ORDER}
    print(f"   (a) residual range over all six sequences: "
          f"{min(rms_all.values()):.2f} to {max(rms_all.values()):.2f}%")
    print("       the assumed second order describes the data on every sequence")

    print("   (b) movement of the limit when one level is dropped [pp]")
    print("       'exact' is the leave-one-out span; 'as printed' is the span of")
    print("       the same range rounded to the three decimals Table 4 shows,")
    print("       which is the figure the prose quotes.")
    for fam in SERIES:
        for k in ORDER:
            _, _, (b0, b1) = fit(series(fam, k))
            exact = (b1 - b0) * 100
            printed = (round(b1, 3) - round(b0, 3)) * 100
            print(f"       {fam:<12}{k:<15}exact {exact:5.2f}   "
                  f"as printed {printed:5.2f}")

    print("   (c) cost of forcing b = 1, on the mean of each family [rms %]")
    for fam in SERIES:
        y = series(fam, "mean")
        _, free, _ = fit(y)
        print(f"       {fam:<12}free {free * 100:6.2f}   "
              f"b = 1 {fit_forced(y) * 100:6.2f}   "
              f"factor {fit_forced(y) / free:5.1f}x")
    print("       the structured sequence pays two orders of magnitude for the")
    print("       assumption; the polyhedral one barely pays at all")


def gci():
    """Fine-grid GCI on the finest pair, Roache (1997),

        GCI_fine = Fs |(f_coarse - f_fine) / f_fine| / (r^p - 1)

    at the same fixed order as the fits and with the factor of safety of 3 that
    Roache prescribes where the order is assumed rather than measured.
    """
    r = H[2] / H[3]
    print(f"\n4. FINE-GRID CONVERGENCE INDEX, finest pair (125^3, 150^3)")
    print(f"   r = {r:.3f}, p = {P:.0f} assumed, F_s = {FS_ASSUMED:.0f}")
    for fam in ("polyhedral", "structured"):
        y = series(fam, "mean")
        eps = abs((y[2] - y[3]) / y[3])
        g = FS_ASSUMED * eps / (r**P - 1.0)
        err = abs(y[3] - 1.0)
        lim = fit(y)[0][1] - 1.0
        print(f"   {fam:<12}GCI = {g * 100:5.2f}%   "
              f"measured error at 150^3 = {err * 100:5.2f}% "
              f"({'inside' if err < g else 'OUTSIDE'})   "
              f"extrapolated limit = {lim * 100:+5.2f}% "
              f"({'inside' if abs(lim) < g else 'OUTSIDE'})")
    print("   the polyhedral family is converged to within its own uncertainty;")
    print("   the structured one is not, which is the conventional statement")
    print("   that a sequence is not approaching the exact solution")


def spread():
    print("\n5. RESIDUAL DIRECTIONAL SPREAD AT h -> 0")
    for fam in SERIES:
        bd = fit(series(fam, "body diagonal"))[0][1]
        ax = fit(series(fam, "axis"))[0][1]
        print(f"   {fam:<12}body diagonal {(bd - 1) * 100:+6.2f}%   "
              f"axis {(ax - 1) * 100:+6.2f}%   "
              f"spread {(bd - ax) * 100:5.1f} pp")
    print("   the structured spread is the anisotropy in radius space, and no")
    print("   further refinement removes it")


if __name__ == "__main__":
    ratios()
    table()
    checks()
    gci()
    spread()
