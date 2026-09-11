#!/usr/bin/env python3
"""Curvature smoothing at a fixed resolution, on both mesh topologies.

    python3 plot_smoothing_sweep.py

All six panels are the same droplet at the same resolution, R/h = 6.4, and
differ only in the number of curvature pre-smoothing cycles.  The point is that
the two topologies fail differently:

  hexahedral   four cycles is an interior optimum -- two is under-smoothed and
               twelve is far over-smoothed, the jump ceasing to settle at all.
  polyhedral   the error rises monotonically from two, so the optimum is at or
               below the lowest count tested.

That is what the neighbour count predicts.  The pre-smoothing is a face-area
weighted average, so one cycle diffuses over the cell's own neighbours: six on a
hexahedron, about fourteen on a dual cell.  The dual therefore reaches any given
convolution length in fewer cycles, and its optimum sits at a lower count.

The convolution length quoted on each panel is the mapping given in the main
text, eps ~ h sqrt(n/2) rescaled onto the cosine kernel, expressed here relative
to the droplet radius because that is the ratio that matters: at twelve cycles
eps reaches 0.69 R, most of the way to the droplet centre, and what the operator
returns is an average over a large fraction of the sphere rather than a local
curvature.
"""

import os
import sys

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("DROPLET_DIR", "/home/jan/runs/Droplet3D")
R0, C, NG = 0.4, 1.0, 24
EXACT = 2 * 15.0e3 / R0

# cycles -> eps/R from the mapping in the main text (eps = 1.8, 2.5, 4.4 h)
EPS_R = {2: 0.28, 4: 0.39, 12: 0.69}

CASES = [
    ("hexahedral", [(2,  "hex32_sm2",  "drop3d_hex32-ts011758.pvtu"),
                    (4,  "hex32_sm4",  "drop3d_hex32-ts011758.pvtu"),
                    (12, "hex32",      "drop3d_hex32-ts011758.pvtu")]),
    ("polyhedral", [(2,  "poly32_sm2", "drop3d_poly32_dual-ts011758.pvtu"),
                    (4,  "poly32",     "drop3d_poly32_dual-ts011758.pvtu"),
                    (12, "poly32_sm12", "drop3d_poly32_dual-ts011758.pvtu")]),
]

plt.rcParams.update({"font.size": 9.5, "font.family": "serif"})


def stats(d):
    # take whichever trace is complete: a rerun in progress leaves a partial
    # spurious.dat beside the finished one, and reading it silently produces
    # plausible but wrong errors
    cand = [os.path.join(ROOT, d, n)
            for n in ("spurious.dat", "spurious_prev.dat")]
    cand = [f for f in cand if os.path.exists(f)]
    a = max((np.loadtxt(f) for f in cand), key=len)
    n = len(a)
    h = np.abs(a[:, 5])[n // 2:]
    return 100.0 * (h.mean() - EXACT) / EXACT, (h.max() - h.min()) / EXACT


def slab(d, f):
    m = pv.read(os.path.join(ROOT, d, f))
    s = m.slice(normal="y", origin=(C, C, C))
    c = np.asarray(s.cell_centers().points, float)
    v = np.asarray(s.cell_data["Velocity [m/s]"], float)
    a = np.asarray(s.cell_data["Vof Sharp [1]"], float)
    # alpha promoted to nodes, then contoured there: tricontour on cell centres
    # can only join ~40 samples around this droplet and renders as a polygon
    ct = s.cell_data_to_point_data().contour([0.5], scalars="Vof Sharp [1]")
    seg = np.asarray(ct.points)[:, [0, 2]] if ct.n_points else np.empty((0, 2))
    return c[:, 0], c[:, 2], v[:, 0], v[:, 2], a, seg


def main():
    panels, vmax, vmin = [], 0.0, np.inf
    for topo, runs in CASES:
        for n, d, f in runs:
            if d is None or not os.path.exists(os.path.join(ROOT, d, f)):
                panels.append((topo, n, None)); continue
            x, z, u, w, a, seg = slab(d, f)
            g = np.linspace(0.0, 2.0, NG)
            GX, GZ = np.meshgrid(g, g)
            pts = np.column_stack([x, z])
            GU = griddata(pts, u, (GX, GZ), method="linear", fill_value=0.0)
            GW = griddata(pts, w, (GX, GZ), method="linear", fill_value=0.0)
            mag = np.hypot(GU, GW)
            vmax = max(vmax, mag.max()); vmin = min(vmin, max(mag[mag > 0].min(), 1e-6))
            panels.append((topo, n, (x, z, a, seg, GX, GZ, GU, GW, mag, *stats(d))))

    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.8))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    th = np.linspace(0, 2 * np.pi, 200)
    q = None

    for ax, (topo, n, P) in zip(axes.ravel(), panels):
        if P is None:
            ax.text(0.5, 0.5, "field output\nnot retained", ha="center",
                    va="center", fontsize=9, color="0.45",
                    transform=ax.transAxes)
            ax.set_title(f"{topo}, {n} cycles", fontsize=9.5, loc="left")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            continue
        x, z, a, seg, GX, GZ, GU, GW, mag, err, rng = P
        nn = np.hypot(GU, GW); nn[nn == 0] = 1.0
        q = ax.quiver(GX, GZ, GU / nn, GW / nn, mag, norm=norm, cmap="viridis",
                      scale=30, width=0.005, pivot="mid")
        if len(seg):
            ang = np.arctan2(seg[:, 1] - C, seg[:, 0] - C)
            o = np.argsort(ang)
            ax.plot(np.append(seg[o, 0], seg[o[0], 0]),
                    np.append(seg[o, 1], seg[o[0], 1]),
                    color="C3", lw=1.5)
        ax.plot(C + R0 * np.cos(th), C + R0 * np.sin(th), color="0.25",
                lw=0.9, ls="--")
        flag = r"$^\dagger$" if rng > 0.25 else ""
        ax.set_title(f"{topo}, {n} cycles  ($\\varepsilon \\approx {EPS_R[n]:.2f}\\,R$)"
                     f"\n$\\Delta p$ error {err:+.2f}%{flag}",
                     fontsize=9.5, loc="left")
        ax.set_aspect("equal"); ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])

    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.90, 0.14, 0.02, 0.72])
    fig.colorbar(q, cax=cax, label=r"$|\mathbf{u}|$ [m s$^{-1}$]")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "pics", f"smoothing_sweep.{ext}"),
                    dpi=200, bbox_inches="tight")

    print(f"  {'topology':<12}{'cycles':>7}{'eps/R':>8}{'dp err':>10}{'range':>8}")
    for topo, n, P in panels:
        if P is None:
            print(f"  {topo:<12}{n:>7}{EPS_R[n]:>8.2f}{'--':>10}"); continue
        print(f"  {topo:<12}{n:>7}{EPS_R[n]:>8.2f}{P[-2]:>9.2f}%{P[-1]:>8.2f}")
    print("written: pics/smoothing_sweep.{pdf,png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
