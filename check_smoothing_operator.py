#!/usr/bin/env python3
"""Can 12 colour-field smoothing cycles produce the -41.78% Laplace-jump error?

Reproduces T-Flows' Smooth_Scalar operator (face-area-weighted average of
face-interpolated values, Vof_Mod/Curvature/Smooth_Scalar.f90:56-72) on the
static-droplet geometry of Appendix D.1: a 2-D cylinder R = 0.4 in a 2x2 box,
32 cells across, so R/h = 6.4.  On a uniform mesh with fs = 0.5 and equal face
areas the operator is

    phi_new(c) = 0.5 phi(c) + 0.5 * mean over faces of phi(neighbour)

which on a cube is the 1/2 + 1/12*sum(6) form that T1.1's forensic recovery
matched to machine precision.  The case is 32x3x32 with the cylinder axis along
y, so the two y-neighbours carry the same value and the in-plane operator is
2/3 phi(c) + 1/12 sum(4).

Curvature is then CSF: kappa = -div(grad c/|grad c|), central differences (the
LSQ gradient reduces to that on a uniform Cartesian mesh).  The reported kappa
is weighted by |grad c| of the same field that enters the force, since the
pressure jump reflects the curvature where the force actually acts.

Exact for a cylinder: kappa = 1/R = 2.5, so sigma/R = 15e3/0.4 = 37500.
"""

import numpy as np

R, L, SIGMA = 0.4, 2.0, 15.0e3


def colour_field(n, sub=8):
    """Area fraction of the cylinder in each cell, sub-sampled."""
    h = L / n
    c = (np.arange(n) + 0.5) * h - L / 2
    off = (np.arange(sub) + 0.5) * h / sub - h / 2
    X = (c[:, None, None, None] + off[None, None, :, None])
    Z = (c[None, :, None, None] + off[None, None, None, :])
    return ((X**2 + Z**2) < R * R).mean(axis=(2, 3)), h


def smooth(c, n_cyc):
    """T-Flows Smooth_Scalar on the quasi-2-D mesh: 2/3 self + 1/12 * sum(4)."""
    s = c.copy()
    for _ in range(n_cyc):
        nb = (np.roll(s, 1, 0) + np.roll(s, -1, 0)
              + np.roll(s, 1, 1) + np.roll(s, -1, 1))
        s = (2.0 / 3.0) * s + (1.0 / 12.0) * nb
    return s


def curvature(c, h):
    """kappa = -div(grad c / |grad c|), central differences."""
    gx = (np.roll(c, -1, 0) - np.roll(c, 1, 0)) / (2 * h)
    gz = (np.roll(c, -1, 1) - np.roll(c, 1, 1)) / (2 * h)
    mag = np.sqrt(gx**2 + gz**2)
    m = np.maximum(mag, 1e-12)
    nx, nz = gx / m, gz / m
    div = ((np.roll(nx, -1, 0) - np.roll(nx, 1, 0)) / (2 * h)
           + (np.roll(nz, -1, 1) - np.roll(nz, 1, 1)) / (2 * h))
    return -div, mag


def effective(c_smooth, h):
    """|grad c|-weighted mean curvature -- the kappa the CSF force applies."""
    k, w = curvature(c_smooth, h)
    band = w > 0.01 * w.max()
    return float(np.average(k[band], weights=w[band]))


def main():
    print("2-D cylinder, R = 0.4, exact kappa = 1/R = 2.5, sigma/R = 37500\n")
    for n in (32, 64, 128):
        c0, h = colour_field(n)
        print(f"  mesh {n:>3}   h = {h:.4f}   R/h = {R/h:.1f}")
        for cyc in (0, 2, 4, 8, 12):
            k = effective(smooth(c0, cyc), h)
            err = 100.0 * (k - 1.0 / R) / (1.0 / R)
            print(f"     {cyc:>2} cycles   kappa = {k:7.4f}   "
                  f"dp = {SIGMA/R*k*R:9.1f}   error = {err:+7.2f}%")
        print()

    print("  paper Appendix D.1, hexahedral 32-equiv:")
    print("     2 cycles  -> +3.34% (omitted) / +11.56% (Rhie-Chow applied)")
    print("    12 cycles  -> +1.40% (omitted) /  +1.53% (Rhie-Chow applied)")
    print("  paper Appendix D.1, polyhedral 32-equiv:")
    print("     2 cycles  ->  -3.19% (omitted) / -10.72% (applied)")
    print("    12 cycles  -> -38.58% (omitted) / -41.78% (applied)")


if __name__ == "__main__":
    main()
