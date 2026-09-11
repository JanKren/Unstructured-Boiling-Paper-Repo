#!/usr/bin/env python3
"""Bubble shape at t = 1.0 ms for the four surface tension coefficients.

The sigma sweep is quoted in the paper as two numbers per run.  This draws it:
the alpha = 0.5 interface on the z = 0 plane for sigma/4, sigma, 2 sigma and
4 sigma, all at t = 1.0 ms on the structured 75^3 mesh at dt = 0.1 us.

Panel (a) overlays the four contours as measured.  They differ in mean radius
by only 1.29%, so what the eye sees is almost entirely shape.  Panel (b)
divides each contour by its own mean radius, which removes that residual size
difference and leaves the four-fold distortion alone -- from a visibly square
bubble at sigma/4 to a nearly round one at 4 sigma.

The point of the figure is the contrast with the gradient: over this same
sixteen-fold range the shape distortion falls by 79% while the four-fold mode
of the interfacial temperature gradient falls by only 26%.  Rounding the bubble
does not round the gradient.

Usage:  python3 plot_sigma_shapes.py [--out pics/sigma_shapes.pdf]
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from analyze_scriven_fields import extract_slice

ROOT = "Data/campaign/SurfTension"
SIGMA_BASE = 0.059
RUNS = [("s025", 0.25, r"$\sigma/4$",  "#1f77b4"),
        ("s100", 1.00, r"$\sigma$",    "#000000"),
        ("s200", 2.00, r"$2\sigma$",   "#ff7f0e"),
        ("s400", 4.00, r"$4\sigma$",   "#d62728")]
BETA, ALPHA_L, R0 = 4.06022, 1.6753e-7, 50.0e-6
T0 = R0 ** 2 / (4 * BETA ** 2 * ALPHA_L)
T_PLOT = 1.0e-3
NGRID = 500
HALF = 150.0


def interface_contour(path):
    """(x, y) of the alpha = 0.5 contour on the z = 0 plane, in micrometres."""
    s = extract_slice(path, normal="z")
    c = s.cell_centers().points
    vof = np.asarray(s.cell_data["Vof Sharp [1]"], float)
    gi = np.linspace(-HALF, HALF, NGRID)
    X, Y = np.meshgrid(gi, gi)
    V = griddata(np.column_stack([c[:, 0] * 1e6, c[:, 1] * 1e6]), vof, (X, Y),
                 method="linear")
    fig = plt.figure()
    cs = plt.contour(X, Y, V, levels=[0.5])
    # take the longest closed segment: the bubble, not any stray fragment
    segs = [v for coll in cs.allsegs for v in coll]
    plt.close(fig)
    xy = max(segs, key=len)
    return xy[:, 0], xy[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pics/sigma_shapes.pdf")
    a = ap.parse_args()

    plt.rcParams.update({"font.size": 9, "font.family": "serif",
                         "savefig.dpi": 300, "axes.grid": False,
                         "axes.labelsize": 9, "axes.titlesize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8})

    R_anal = 2 * BETA * np.sqrt(ALPHA_L * (T_PLOT + T0)) * 1e6
    fig = plt.figure(figsize=(7.2, 3.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")

    th = np.linspace(0, 2 * np.pi, 400)
    ax1.plot(R_anal * np.cos(th), R_anal * np.sin(th), color="0.6", ls="--",
             lw=1.0, label="Scriven", zorder=1)

    for tag, fac, lab, col in RUNS:
        f = [x for x in glob.glob(os.path.join(ROOT, tag, "*-ts010000.pvtu"))
             if "front" not in os.path.basename(x)]
        if not f:
            print(f"  {tag}: no ts10000 snapshot")
            continue
        x, y = interface_contour(f[0])
        ax1.plot(x, y, color=col, lw=1.3, label=lab, zorder=2)

        r = np.hypot(x, y)
        t = np.arctan2(y, x)
        o = np.argsort(t)
        rn = r[o] / r.mean()
        ax2.plot(np.append(t[o], t[o][0] + 2 * np.pi), np.append(rn, rn[0]),
                 color=col, lw=1.2, label=lab)
        print(f"  {tag:5s} sigma={SIGMA_BASE*fac:7.5f}  <R>={r.mean():7.3f} um  "
              f"min/max = {rn.min():.4f}/{rn.max():.4f}  "
              f"peak-to-peak {100*(rn.max()-rn.min()):.2f}%")

    ax1.set_aspect("equal")
    ax1.set_xlim(-HALF, HALF); ax1.set_ylim(-HALF, HALF)
    ax1.set_xlabel(r"$x$ [$\mu$m]"); ax1.set_ylabel(r"$y$ [$\mu$m]")
    ax1.set_title("(a) interface at $t = 1.0$~ms".replace("~", " "), fontsize=9)
    ax1.legend(frameon=False, loc="lower right", fontsize=7.5)

    ax2.plot(th, np.ones_like(th), color="0.6", ls="--", lw=1.0)
    ax2.set_ylim(0.90, 1.10)
    ax2.set_yticks([0.95, 1.0, 1.05])
    ax2.set_yticklabels(["0.95", "1", "1.05"], fontsize=6.5)
    ax2.set_rlabel_position(22.5)
    ax2.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax2.set_xticklabels([r"$0^\circ$", "", r"$90^\circ$", "", r"$180^\circ$",
                         "", r"$270^\circ$", ""], fontsize=7)
    ax2.set_title(r"(b) $R(\theta)/\langle R\rangle$", fontsize=9, pad=12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"\n  analytical radius {R_anal:.2f} um")
    print("  NB: this is the z = 0 slice, which spans the cubic invariant from")
    print("  s = 1 (axis) only to s = 1/2 (face diagonal), i.e. 3/4 of the range")
    print("  of the axis-to-body-diagonal measure used in the paper -- so these")
    print("  peak-to-peak values run about 3/4 of those, plus a little from the")
    print("  contour's cell-scale wiggle, which a fitted mode averages out.")
    print(f"  figure -> {a.out}")


if __name__ == "__main__":
    main()
