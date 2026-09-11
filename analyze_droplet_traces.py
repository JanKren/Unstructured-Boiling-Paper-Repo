#!/usr/bin/env python3
"""Static droplet benchmark (Appendix D.1) from the per-step solver traces.

    python3 analyze_droplet_traces.py

Each run writes `spurious.dat` from User_Mod/End_Of_Time_Step.f90:108, one row
per step, six columns:

    time | a_vof | u_rms | u_max | p_max-p_min | p_in-p_out

`p_in-p_out` is the volume-weighted mean pressure over cells entirely inside
the droplet minus those entirely outside; the exact value for a 2-D cylinder is
sigma/R = 15000/0.4 = 37500 (one principal curvature, not two).

The published table quoted only the second-half MEAN of that column. That is
safe for a converged run and misleading for one that is not, and several of the
polyhedral 32-equivalent runs are not: their pressure jump swings over a factor
of four while the mean stays quotable. So this script reports, for the same
averaging window:

  * std   -- scatter about the mean
  * range -- min to max, which is what exposes a run that is still moving
  * drift -- (4th quarter mean) - (3rd quarter mean), a check that the average
             is taken over a plateau and not over a transient
  * Ca    -- mu max|u| / sigma with mu = 1, the published definition

A run is called UNSTEADY when its second-half range exceeds a quarter of the
exact jump, i.e. when the quantity being averaged moves by more than 25% of the
thing it is supposed to measure. On that test the hexahedral runs and the
polyhedral 64-equivalent pass, and the polyhedral 32-equivalent runs pass only
at 2-4 smoothing cycles.
"""

import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "Data", "StaticDroplet")
EXACT = 37500.0        # sigma/R for the 2-D cylinder [Pa]
SIGMA = 15.0e3         # surface tension [N/m]
MU = 1.0               # dynamic viscosity, both phases [Pa s]
UNSTEADY = 0.25        # second-half range, as a fraction of EXACT


def stats(run):
    """Second-half statistics of one run, or None if the trace is missing."""
    path = os.path.join(ROOT, run, "spurious.dat")
    if not os.path.exists(path):
        return None
    d = np.loadtxt(path)
    if d.ndim != 2 or len(d) < 8:
        return None
    n = len(d)
    dp, um = d[:, 5], d[:, 3]
    h = dp[n // 2:]
    q3 = dp[n // 2:3 * n // 4].mean()
    q4 = dp[3 * n // 4:].mean()
    return dict(rows=n, mean=h.mean(), std=h.std(), lo=h.min(), hi=h.max(),
                drift=q4 - q3, ca=MU * um[n // 2:].mean() / SIGMA,
                err=100.0 * (h.mean() - EXACT) / EXACT,
                rng=(h.max() - h.min()) / EXACT)


def row(label, run, s):
    if s is None:
        return f"  {label:<34} {'trace missing':>12}"
    flag = "UNSTEADY" if s["rng"] > UNSTEADY else ""
    return (f"  {label:<34} {s['err']:+7.2f}%  {s['ca']:9.2e}  "
            f"{s['std']:8.0f}  {s['lo']:7.0f}-{s['hi']:<7.0f} "
            f"{s['drift']:+8.0f}  {flag}")


HEAD = (f"  {'configuration':<34} {'dp err':>8}  {'Ca':>9}  "
        f"{'std':>8}  {'range':>15} {'drift':>8}")

MAIN = [("hex 32,  flux omitted",          "cavity32"),
        ("hex 64,  flux omitted",          "cavity64"),
        ("hex 128, flux omitted",          "cavity128"),
        ("hex 32,  consistent flux",       "cavity32_corr"),
        ("hex 64,  consistent flux",       "cavity64_corr"),
        ("hex 128, consistent flux",       "cavity128_corr"),
        ("poly 32-equiv,  flux omitted",   "poly32_off"),
        ("poly 64-equiv,  flux omitted",   "poly64_off"),
        ("poly 32-equiv,  consistent",     "poly32_corr"),
        ("poly 64-equiv,  consistent",     "poly64_corr")]

FACT = [("hex  omitted, 2 cycles",         "cavity32"),
        ("hex  omitted, 12 cycles",        "hex32_nocorr_sm12"),
        ("hex  applied, 2 cycles",         "hex32_rc_sm2"),
        ("hex  applied, 12 cycles",        "cavity32_corr"),
        ("poly omitted, 2 cycles",         "poly32_off"),
        ("poly omitted, 12 cycles",        "poly32_nocorr_sm12"),
        ("poly applied, 2 cycles",         "poly32_rc_sm2"),
        ("poly applied, 12 cycles",        "poly32_corr")]

SWEEP = [("poly 32-equiv, 0 cycles",       "poly32_rc_sm0"),
         ("poly 32-equiv, 2 cycles",       "poly32_rc_sm2"),
         ("poly 32-equiv, 4 cycles",       "poly32_rc_sm4"),
         ("poly 32-equiv, 8 cycles",       "poly32_rc_sm8"),
         ("poly 32-equiv, 12 cycles",      "poly32_corr"),
         ("poly 64-equiv, 4 cycles",       "poly64_rc_sm4"),
         ("poly 64-equiv, 12 cycles",      "poly64_corr")]


def main():
    if not os.path.isdir(ROOT):
        sys.exit(f"no data at {ROOT}")
    print(f"exact sigma/R = {EXACT:.0f} Pa; means over the second half of each "
          f"run\nUNSTEADY: second-half range exceeds {UNSTEADY:.0%} of the "
          f"exact jump\n")
    unsteady = []
    for title, block in (("Table 12 -- flux treatment", MAIN),
                         ("Table 13 -- factorial at 32-equivalent", FACT),
                         ("Smoothing sweep, Rhie-Chow applied", SWEEP)):
        print(title)
        print(HEAD)
        for label, run in block:
            s = stats(run)
            print(row(label, run, s))
            if s and s["rng"] > UNSTEADY and run not in unsteady:
                unsteady.append(run)
        print()
    print(f"runs failing the steadiness test: {len(unsteady)}")
    for r in unsteady:
        print(f"   {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
