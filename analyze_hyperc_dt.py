#!/usr/bin/env python3
"""Is the pure Hyper-C ladder's -4.28% asymptote temporal or spatial?

`results/hyperc-ladder.md` fits R_vol(1 ms) over four structured levels at a
FIXED dt = 1 us and finds pure Hyper-C converging at p = 2.30 to
R_inf = 111.41 um, i.e. -4.28% against the Scriven radius as it stood then
(116.39 um; scriven_reference now gives 116.64).  The note argued
that holding dt fixed while h falls splits the error budget cleanly --

    a SPATIAL error vanishes into p;
    a TEMPORAL error is constant along the ladder and lands in R_inf

-- so a mesh-independent -4.28% would be the signature of a time-splitting
error.  This script tests that from runs already on disk.

**The premise turns out to be false**, and that is the main finding here.  The
interface Courant number Co = (dR/dt)*dt/h RISES under refinement at fixed dt,
so the temporal error is not constant along the ladder: it grows, and therefore
contaminates p as well as R_inf.  The correction has to be applied level by
level, not as a single offset.

Two independent measurements of the temporal sensitivity of R_vol(1 ms):

  1. The blend on the structured 75^3 at three time steps, one binary
     (`results/dt-series.md` established the build equivalence):
     2.0 / 1.0 / 0.1 us.  Three points fix the temporal order q.

  2. dt = 2.0 us against 1.0 us at fixed mesh, for both schemes and every
     level.  The pure Hyper-C 2 us arm is the FIRST, DISCARDED ladder
     (`/data/scratch/shared/kren_j/HyperC/`), which hyperc-ladder.md rejected
     for a *refinement* study because corrector exhaustion rose
     27 -> 60 -> 95 -> 99% along it.  That objection does not apply to a dt
     contrast at FIXED mesh, but the contamination still grows with level, so
     075 (27% vs 10%) is the trustworthy pair and 100/125 corroborate only.

The cross-check that makes this safe: at 075 the blend and pure Hyper-C give
+1.36% and +1.34% for the same halving of dt.  The blend arm has no corrector
exhaustion at all, so the ~1.35% is a property of the time integration, not of
the boundedness corrector.

Run from the paper directory.  Reads only bench-data.dat column 4 (R_vol).
"""

import os
import sys

import numpy as np
from scipy.optimize import curve_fit

ROOT = os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling")
# Derived in scriven_reference from the Table 1 properties, not restated here.
# This module previously carried 116.39, the value the stale beta = 4.06022 gave.
import scriven_reference
R_SCRIVEN = scriven_reference.r_scriven_um(1.0e-3)   # um at 1 ms, dT = 1.25 K
T_TARGET = 1.0e-3           # s
L_DOMAIN = 300.0            # um, cube edge; h = L/n reproduces the documented
                            # H = 4.05 um at n = 74 (075 level)

# bench-data.dat columns: t [s], A_interface [m2], V [m3], R_vol [m], and in
# runs made after the Scriven User_Mod change a fifth, sum(m_dot) [kg/s].  The
# Hyper-C campaign has five columns, the aniso-*/DtSeries/SurfTension runs four;
# R_vol is column 4 in both, so index it positionally and never by -1.
COL_T, COL_R = 0, 3

# level, cells per direction, dt=2us file, dt=1us file, corrector exhaustion@2us
HYPERC = [
    ("075",  74.0, "Data/hyperc-dt2/hyperc-075.dat", "Data/hyperc-bench/pure-075.dat", "27%"),
    ("100",  99.0, "Data/hyperc-dt2/hyperc-100.dat", "Data/hyperc-bench/pure-100.dat", "60%"),
    ("125", 124.0, "Data/hyperc-dt2/hyperc-125.dat", "Data/hyperc-bench/pure-125.dat", "95%"),
]
# hyperc-150 at 2 us died at 0.318 ms, so the fourth level is dt = 1 us only
HYPERC_150 = (149.0, "Data/hyperc-bench/pure-150.dat")

BLEND = [
    ("075",  74.0, "Data/blend-dt2/aniso-075-off.dat", "Data/campaign/DtSeries/st75-dt1000/bench-data.dat"),
    ("100",  99.0, "Data/blend-dt2/aniso-100-off.dat", "Data/campaign/DtSeries/st100-dt1000/bench-data.dat"),
    ("125", 124.0, "Data/blend-dt2/aniso-125-off.dat", "Data/campaign/DtSeries/st125-dt1000/bench-data.dat"),
    ("150", 149.0, "Data/blend-dt2/aniso-150-off.dat", "Data/hyperc-bench/base-150-dt1.dat"),
]
# the third time step, blend only, 75^3: the sigma-sweep baseline run
BLEND_075_DT01 = "Data/campaign/SurfTension/s100/bench-data.dat"


def load(path):
    """Return (t [s], R_vol [um]) with the integrity screen this repo insists on."""
    d = np.loadtxt(os.path.join(ROOT, path))
    t, r = d[:, COL_T], d[:, COL_R] * 1e6
    issues = []
    if not np.all(np.diff(t) > 0):
        issues.append("time not monotone -- spliced output, see CLAUDE.md")
    dts = np.diff(t)
    if dts.size and (dts.max() - dts.min()) / dts.mean() > 1e-6:
        issues.append("dt is not constant over the run")
    if not np.all(np.isfinite(d)):
        issues.append("NaN or inf present")
    if issues:
        raise ValueError(f"{path}: " + "; ".join(issues))
    return t, r


def r_at(path, t_target=T_TARGET):
    t, r = load(path)
    if t[-1] < t_target - 1e-12:
        raise ValueError(f"{path}: reaches only {t[-1]*1e3:.4f} ms")
    i = np.argmin(np.abs(t - t_target))
    return r[i] if abs(t[i] - t_target) < 1e-12 else float(np.interp(t_target, t, r))


def courant(path, n, dt=1e-6):
    """Interface Courant number at 1 ms: Co = (dR/dt) * dt / h."""
    t, r = load(path)
    u = np.gradient(r * 1e-6, t)[-1]        # m/s
    return u * dt / (L_DOMAIN * 1e-6 / n)


def err(r):
    return 100.0 * (r - R_SCRIVEN) / R_SCRIVEN


def temporal(x, r0, c, q):
    """R(dt) -> r0 as dt -> 0, leading error c*dt^q."""
    return r0 - c * x ** q


def spatial(x, r_inf, c, p):
    """R(n) -> r_inf as n -> inf, leading error c*n^-p."""
    return r_inf - c * x ** (-p)


def fit_spatial(n, r, p0):
    po, _ = curve_fit(spatial, n, r, p0=p0, maxfev=400000)
    return po[0], po[2], float(np.max(np.abs(r - spatial(n, *po))))


def main():
    rule = "=" * 78

    # ---- 1. temporal order, from three time steps on one mesh, one binary ----
    print(rule)
    print("1.  TEMPORAL ORDER  --  blend, structured 75^3, one binary")
    print(rule)
    dts = np.array([2.0, 1.0, 0.1])
    rb = np.array([r_at(BLEND[0][2]), r_at(BLEND[0][3]), r_at(BLEND_075_DT01)])
    for d, r in zip(dts, rb):
        print(f"    dt = {d:>4} us    R_vol(1 ms) = {r:8.3f} um   {err(r):+7.2f}%")
    (r0b, _, qb), _ = curve_fit(temporal, dts, rb, p0=[118.5, 1.8, 1.0], maxfev=100000)
    print(f"\n    fit R = R0 - c*dt^q  (3 points, exact):")
    print(f"      q  = {qb:.3f}       -> essentially FIRST order in dt")
    print(f"      R0 = {r0b:.3f} um   {err(r0b):+.2f}%   (dt -> 0 limit at 75^3)")
    print(f"      still temporal at dt = 1 us: {r0b - rb[1]:+.3f} um "
          f"= {100*(r0b - rb[1])/rb[1]:+.2f}%")

    # ---- 2. why it is not a constant offset: Courant rises with refinement ---
    print()
    print(rule)
    print("2.  THE LADDER'S PREMISE FAILS  --  Co rises under refinement")
    print(rule)
    print(f"    {'lvl':<5}{'h [um]':>9}{'Co @1ms':>10}     blend dR/R (2->1 us)"
          f"   Hyper-C dR/R")
    hyp = {L: (r_at(f2), r_at(f1)) for L, _, f2, f1, _ in HYPERC}
    for L, n, f2, f1 in BLEND:
        b2, b1 = r_at(f2), r_at(f1)
        db = 100 * (b1 - b2) / b1
        dh = ""
        if L in hyp:
            h2, h1 = hyp[L]
            dh = f"{100*(h1-h2)/h1:>+13.2f}%"
        print(f"    {L:<5}{L_DOMAIN/n:>9.3f}{courant(f1, n):>10.4f}"
              f"{db:>+22.2f}%{dh}")
    print("\n    Co grows 2.2x across the ladder at fixed dt, so the temporal error"
          "\n    grows too.  It does NOT land entirely in R_inf -- it also biases p."
          "\n    The blend's grows faster than Co, pure Hyper-C's slower.")

    # ---- 3. dt -> 0 at each level, then the double limit --------------------
    print()
    print(rule)
    print("3.  DOUBLE LIMIT  --  Richardson in dt at each level, then refit in h")
    print(rule)
    fac = 2 ** qb - 1
    print(f"    Richardson multiplier 1/(2^q - 1) = {1/fac:.3f}  at q = {qb:.3f}\n")

    n_h = np.array([n for _, n, *_ in HYPERC])
    h1 = np.array([r_at(f1) for *_, f1, _ in [(a, b, c, d, e) for a, b, c, d, e in HYPERC]])
    h2 = np.array([r_at(f2) for _, _, f2, _, _ in HYPERC])
    h0 = h1 + (h1 - h2) / fac

    n_b = np.array([n for _, n, *_ in BLEND])
    b1 = np.array([r_at(f1) for *_, f1 in BLEND])
    b2 = np.array([r_at(f2) for _, _, f2, _ in BLEND])
    b0 = b1 + (b1 - b2) / fac

    cell = lambda v: f"{v:8.3f} ({err(v):+6.2f}%)"
    blank = " " * 17
    print(f"    {'lvl':<5}{'Hyper-C @1us':>17} {'Hyper-C dt->0':>17}"
          f" {'blend @1us':>17} {'blend dt->0':>17}")
    for i, (L, *_) in enumerate(BLEND):
        have = i < len(h1)
        print(f"    {L:<5}{cell(h1[i]) if have else blank:>17} "
              f"{cell(h0[i]) if have else blank:>17} "
              f"{cell(b1[i]):>17} {cell(b0[i]):>17}")
    print()

    n4 = np.append(n_h, HYPERC_150[0])
    r4 = np.append(h1, r_at(HYPERC_150[1]))
    for label, nn, arr, p0 in (
            ("Hyper-C, 4 pts, dt = 1 us (published)", n4, r4, [111.0, 5e4, 2.3]),
            ("Hyper-C, 3 pts, dt = 1 us",             n_h, h1, [112.0, 5e4, 2.2]),
            ("Hyper-C, 3 pts, dt -> 0",               n_h, h0, [114.0, 5e4, 2.1]),
            ("blend,   4 pts, dt = 1 us (published)", n_b, b1, [139.0, 1e4, 1.3]),
            ("blend,   4 pts, dt -> 0",               n_b, b0, [155.0, 1e4, 1.0])):
        ri, p, res = fit_spatial(nn, arr, p0)
        print(f"    {label:<39}: p = {p:.2f}  R_inf = {ri:8.3f} um "
              f"{err(ri):+7.2f}%   maxres {res:.3f}")
    print("\n    Do not quote the blend's dt -> 0 R_inf as a number: p < 1 makes the"
          "\n    extrapolation unreliable.  The measured sequence +1.85 / +9.26 /"
          "\n    +13.78 / +16.80% already shows the divergence without any fit.")

    # ---- 4. how much does the answer depend on q? --------------------------
    print()
    print(rule)
    print("4.  SENSITIVITY  --  the assumed temporal order q")
    print(rule)
    print(f"    {'q':>6}{'multiplier':>12}{'R_inf(dt->0)':>15}{'vs Scriven':>13}{'p':>7}")
    for q in (0.80, 0.85, qb, 0.95, 1.00, 1.10, 1.20):
        f = 2 ** q - 1
        ri, p, _ = fit_spatial(n_h, h1 + (h1 - h2) / f, [114.0, 5e4, 2.1])
        mark = "  <- measured" if abs(q - qb) < 1e-9 else ""
        print(f"    {q:>6.2f}{1/f:>12.3f}{ri:>15.3f}{err(ri):>+12.2f}%{p:>7.2f}{mark}")
    rel = (h0[0] - h1[0]) / h1[0]
    ri4, _, _ = fit_spatial(n4, r4, [111.0, 5e4, 2.3])
    print(f"\n    conservative -- apply only the clean 075 correction "
          f"({100*rel:+.2f}%) to the")
    print(f"    published R_inf: {ri4:.2f} -> {ri4*(1+rel):.2f} um  "
          f"{err(ri4*(1+rel)):+.2f}%")

    print()
    print(rule)
    print("VERDICT")
    print(rule)
    print(f"    R_vol(1 ms) is NOT dt-converged at 1 us -- {100*(r0b-rb[1])/rb[1]:+.1f}% "
          f"remains at 75^3,")
    print("    first order, so it does not fall away quickly.  Roughly HALF of the")
    print("    published -4.28% is temporal; the double limit is -1.6% to -2.8%.")
    print("    Both headline trends SURVIVE the correction and sharpen:")
    print("      pure Hyper-C still converges toward Scriven, blend still diverges.")
    print("    The anisotropy is dt-INSENSITIVE (results/dt-series.md: 1.18/1.07/")
    print("    1.03/1.01 over the same halving, 1.2x over 20x), so the radius error")
    print("    and the orientation error are separable.")


if __name__ == "__main__":
    sys.exit(main())
