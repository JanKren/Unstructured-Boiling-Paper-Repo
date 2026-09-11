#!/usr/bin/env python3
"""Does the static-droplet pressure jump converge under refinement?

    python3 analyze_droplet_convergence.py

Table 12 quotes the second-half mean of the Laplace jump at three resolutions
for each mesh family, all with the consistent flux and 12 curvature smoothing
cycles.  It reports the three numbers but never asks whether they form a
convergent sequence, which is the question a referee will put to them.

For each family this reports, level by level: the mean jump, its error against
the exact sigma/R, the second-half scatter, and the observed order

    p = log2( e_coarse / e_fine )

Two cautions are built in rather than left to the reader:

  * a level whose second-half range exceeds a quarter of the exact jump is not
    a converged steady state, and an order computed across it is meaningless.
    Those levels are flagged and excluded from the fit.
  * an order estimated from errors that change sign, or from a pair whose
    errors are within their own scatter, is not a measurement.  Both cases are
    reported rather than silently fitted.

Richardson extrapolation to h -> 0 is given where the sequence supports it, as
a check on what the sequence is converging *to* -- the exact answer is known
here, so a limit that misses 37500 Pa indicates a floor, not just slow decay.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "Data", "StaticDroplet")
EXACT = 37500.0        # sigma/R for the 2-D cylinder [Pa]
UNSTEADY = 0.25

# Run names follow `analyze_droplet_traces.py`, which is authoritative:
# *_corr is the 12-cycle arm, *_rc_sm4 the 4-cycle one.
FAMILIES = {
    "hexahedral, 12 smoothing cycles": [
        (32, "cavity32_corr.dat"), (64, "cavity64_corr.dat"),
        (128, "cavity128_corr.dat")],
    "polyhedral, 12 smoothing cycles": [
        (32, "poly32_corr.dat"), (64, "poly64_corr.dat"),
        (128, "poly128_corr/spurious.dat")],
    "polyhedral, 4 smoothing cycles": [
        (32, "poly32_rc_sm4.dat"), (64, "poly64_rc_sm4.dat"),
        (128, "poly128_rc_sm4/spurious.dat")],
}


def stats(rel):
    for p in (os.path.join(ROOT, rel), os.path.join(ROOT, rel + ".dat")):
        if os.path.exists(p):
            break
    else:
        return None
    d = np.loadtxt(p)
    if d.ndim != 2 or len(d) < 8:
        return None
    n = len(d)
    dp = np.abs(d[:, 5])
    h = dp[n // 2:]
    return dict(rows=n, mean=h.mean(), std=h.std(),
                err=100.0 * (h.mean() - EXACT) / EXACT,
                rng=(h.max() - h.min()) / EXACT,
                drift=dp[3 * n // 4:].mean() - dp[n // 2:3 * n // 4].mean())


def main():
    for fam, levels in FAMILIES.items():
        print(f"\n{fam}, consistent flux")
        print(f"  {'level':>6} {'h':>9} {'rows':>6} {'mean dp':>10} {'err':>9}"
              f" {'std':>7} {'drift':>8}   state")
        got = []
        for lev, rel in levels:
            s = stats(rel)
            if s is None:
                print(f"  {lev:>6} {'--':>9}   (no trace: {rel})")
                continue
            steady = s["rng"] <= UNSTEADY
            print(f"  {lev:>6} {2.0/lev:9.5f} {s['rows']:>6} {s['mean']:10.1f}"
                  f" {s['err']:+8.2f}% {s['std']:7.0f} {s['drift']:+8.0f}   "
                  f"{'steady' if steady else 'UNSTEADY'}")
            got.append((lev, s, steady))

        use = [(l, s) for l, s, ok in got if ok]
        if len(use) < 2:
            print("  fewer than two steady levels -- no order can be quoted")
            continue

        print("  observed order between consecutive steady levels:")
        for (l1, s1), (l2, s2) in zip(use, use[1:]):
            e1, e2 = abs(s1["err"]), abs(s2["err"])
            note = ""
            if s1["err"] * s2["err"] < 0:
                note = "  <- errors change SIGN; the order is not meaningful"
            elif abs(s2["mean"] - EXACT) < 2.0 * s2["std"]:
                note = "  <- fine-level error is inside its own scatter"
            p = np.log2(e1 / e2) if e2 > 0 else float("inf")
            print(f"    {l1:>4} -> {l2:<4}  {e1:6.2f}% -> {e2:5.2f}%   "
                  f"p = {p:5.2f}{note}")

        if len(use) >= 3:
            (_, sc), (_, sm), (_, sf) = use[-3:]          # coarse, medium, fine
            f_c, f_m, f_f = sc["mean"], sm["mean"], sf["mean"]
            d_coarse, d_fine = f_c - f_m, f_m - f_f       # successive increments
            ratio = d_coarse / d_fine if abs(d_fine) > 1e-9 else float("nan")
            print(f"  successive increments: {d_coarse:+.1f} then {d_fine:+.1f} Pa"
                  f"   (ratio {ratio:.2f})")
            if not np.isfinite(ratio) or ratio <= 1.0:
                print("  the increments do NOT shrink under refinement, so the")
                print("  triple is outside the asymptotic range: no order and no")
                print("  Richardson limit can be quoted from it.")
            else:
                p = np.log2(ratio)
                rich = f_f + (f_f - f_m) / (2.0**p - 1.0)
                print(f"  Richardson: p = {p:.2f}, dp(h->0) = {rich:.1f} Pa "
                      f"({100.0*(rich-EXACT)/EXACT:+.2f}% of exact)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
