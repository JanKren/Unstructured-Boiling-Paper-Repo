#!/usr/bin/env python3
"""The normalised donor value written out for the three limiting orientations.

    python3 analyze_alpha_tilde_orientations.py

Section 4 says alpha_tilde_D is orientation-dependent before any blending is
applied.  This writes that statement as an equation for the cases that bound it,
which is what makes the mechanism concrete.

SETUP.  A cubic cell of side h, a plane interface of unit normal n, and alpha(s)
the cell's volume fraction when the plane sits at signed distance s from the cell
centre, measured along n.  The plane must travel

    W = h ||n||_1

for alpha to run 0 to 1.  The limiter reads three cells spaced h apart along a
face direction i, so stepping from one to the next moves the plane by

    ds = h |n_i|

in normal distance, and the normalised donor value is

    alpha_tilde_D(s) = [ alpha(s) - alpha(s + ds) ] / [ alpha(s - ds) - alpha(s + ds) ].

AXIS.  n = (1,0,0), face along x.  Then W = h and ds = h = W: one step spans the
whole ramp, so the neighbours sit outside it and saturate, alpha_U = 0 and
alpha_A = 1, the denominator is unity, and alpha(s) is exactly linear, so
alpha_tilde_D = 1/2 - s/h sweeps [0,1] as the interface crosses the cell.

BODY DIAGONAL.  n = (1,1,1)/sqrt(3).  Then W = sqrt(3) h and ds = h/sqrt(3) =
W/3: one step spans a third of the ramp, both neighbours stay inside it, neither
saturates, and the denominator alpha_A - alpha_U is no longer unity.  The three
samples then straddle a nearly linear stretch of the profile and alpha_tilde_D
is compressed towards 1/2.

EXACTNESS.  alpha(s) is the volume of a cube cut by a plane, which is piecewise
cubic and known in closed form, so it is evaluated analytically here rather than
by sub-sampling the cell.  An earlier version of this script cut the cube on a
400^3 lattice and returned -1.359 for the face diagonal against the exact
-sqrt(2); the quantisation error fell entirely on the slope, which is a
difference of two nearly equal cross-sections.  The slope is likewise taken in
closed form.  With the interface at the cell centre both neighbours are placed
symmetrically, so d(alpha_A - alpha_U)/ds vanishes there and

    d alpha_tilde_D / ds = [ A(0) - A(ds) ] / [ alpha(-ds) - alpha(ds) ],

with A = d alpha / ds the cross-sectional area of the cut.  That is -1 on the
axis, -sqrt(2) on the face diagonal and -15 sqrt(3) / 23 on the body diagonal.
"""

import sys

import numpy as np


def _sorted_components(n):
    """Unit normal, absolute components, descending.  The cut volume of a cube
    depends only on these, by the cube's own symmetry."""
    nn = np.asarray(n, float)
    nn = nn / np.linalg.norm(nn)
    return np.sort(np.abs(nn))[::-1]


def alpha_of_s(n, s):
    """Exact volume fraction of a unit cube on the s' < s side of the plane.

    Closed form for the cube-plane cut.  Written in the shifted coordinate
    t = s + W/2 in [0, W], for which the fraction is a sum of clipped powers."""
    c = _sorted_components(n)
    c = c[c > 1e-14]
    W = c.sum()
    t = np.asarray(s, float) + 0.5 * W
    pos = lambda x, k: np.where(x > 0.0, np.maximum(x, 0.0) ** k, 0.0)

    if c.size == 3:
        a, b, d = c
        num = (pos(t, 3) - pos(t - a, 3) - pos(t - b, 3) - pos(t - d, 3)
               + pos(t - a - b, 3) + pos(t - a - d, 3) + pos(t - b - d, 3)
               - pos(t - W, 3))
        out = num / (6.0 * a * b * d)
    elif c.size == 2:
        a, b = c
        num = pos(t, 2) - pos(t - a, 2) - pos(t - b, 2) + pos(t - W, 2)
        out = num / (2.0 * a * b)
    else:
        out = np.clip(t / c[0], 0.0, 1.0)
    return np.clip(out, 0.0, 1.0)


def dalpha_ds(n, s):
    """Exact d alpha / d s, the area of the cut cross-section."""
    c = _sorted_components(n)
    c = c[c > 1e-14]
    W = c.sum()
    t = np.asarray(s, float) + 0.5 * W
    pos = lambda x, k: np.where(x > 0.0, np.maximum(x, 0.0) ** k, 0.0)

    if c.size == 3:
        a, b, d = c
        num = (pos(t, 2) - pos(t - a, 2) - pos(t - b, 2) - pos(t - d, 2)
               + pos(t - a - b, 2) + pos(t - a - d, 2) + pos(t - b - d, 2)
               - pos(t - W, 2))
        return num / (2.0 * a * b * d)
    if c.size == 2:
        a, b = c
        num = pos(t, 1) - pos(t - a, 1) - pos(t - b, 1) + pos(t - W, 1)
        return num / (a * b)
    return np.where((t > 0.0) & (t < c[0]), 1.0 / c[0], 0.0)


def main():
    print("alpha_tilde_D(s) = [a(s) - a(s+ds)] / [a(s-ds) - a(s+ds)],   ds = h|n_i|\n")
    print(f"  {'orientation':<16}{'||n||_1':>9}{'W/h':>7}{'ds/h':>7}{'ds/W':>7}"
          f"{'a_U':>7}{'a_A':>7}{'range of a~_D':>16}{'slope':>10}{'exact':>12}")

    exact = {"axis": (-1.0, "-1"),
             "face diagonal": (-np.sqrt(2.0), "-sqrt(2)"),
             "body diagonal": (-15.0 * np.sqrt(3.0) / 23.0, "-15sqrt(3)/23")}

    for lab, n in (("axis", (1, 0, 0)),
                   ("face diagonal", (1, 1, 0)),
                   ("body diagonal", (1, 1, 1))):
        nn = np.asarray(n, float) / np.linalg.norm(n)
        l1 = np.abs(nn).sum()
        ds = abs(nn[0])                       # in units of h, for an x-face
        # sweep the interface across the whole ramp: along the normal the
        # cell spans +-W/2, not +-h/2, and W = h||n||_1
        s = np.linspace(-0.5 * l1, 0.5 * l1, 2001)
        aU, aD, aA = (alpha_of_s(n, s + ds), alpha_of_s(n, s),
                      alpha_of_s(n, s - ds))
        den = aA - aU
        ok = np.abs(den) > 1e-12
        at = np.full_like(s, np.nan)
        at[ok] = (aD[ok] - aU[ok]) / den[ok]
        # slope at the cell centre, in closed form (see the module docstring)
        a_plus = float(alpha_of_s(n, ds))
        slope = float(dalpha_ds(n, 0.0) - dalpha_ds(n, ds)) / (1.0 - 2.0 * a_plus)
        ex, exs = exact[lab]
        flag = "" if abs(slope - ex) < 5e-9 else "   <-- MISMATCH"
        print(f"  {lab:<16}{l1:>9.4f}{l1:>7.3f}{ds:>7.3f}{ds/l1:>7.3f}"
              f"{float(alpha_of_s(n, ds)):>7.3f}{float(alpha_of_s(n, -ds)):>7.3f}"
              f"{np.nanmin(at):>8.3f}-{np.nanmax(at):<7.3f}{slope:>10.4f}"
              f"{exs:>12}{flag}")

    print("\n  a_U and a_A are quoted with the interface at the cell centre.")
    print("  On the axis they are exactly 1 and 0 -- the neighbours are outside")
    print("  the ramp -- so the denominator is unity and a~_D IS the volume")
    print("  fraction.  On the body diagonal both neighbours are still inside the")
    print("  transition, the denominator shrinks, and a~_D is squeezed towards 1/2.")
    print("\n  That contraction is the orientation dependence, and it is present")
    print("  before gamma_f multiplies anything.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
