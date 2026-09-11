#!/usr/bin/env python3
"""Spurious currents around the static droplet, Appendix D.1.

    python3 plot_spurious_currents.py

Six panels at the last step (t = 0.12 s), all with the consistent flux
formulation: the hexahedral refinement sequence on top, the polyhedral one
below at matched interface resolution.

  (a) hexahedral 32     (b) hexahedral 64     (c) hexahedral 128
  (d) polyhedral 32-eq  (e) polyhedral 64-eq  (f) polyhedral 128-eq

Each topology is shown at its own appropriate smoothing count: 12 cycles on the
hexahedral meshes, 4 on the polyhedral. That is not an inconsistency but the
paper's own recommendation -- the sweep in D.1 puts the polyhedral optimum at
2-4 cycles, and carrying the hexahedral count across is what the appendix warns
against. Carried across, it is destructive: the polyhedral 32-equivalent case at
12 cycles ends with its droplet advected 1.93 radii onto the lower wall. That
run is reported in the text as the cautionary result, not plotted here.

At 4 cycles the polyhedral droplet stays intact at every resolution and the
displacement converges under refinement -- 0.367, 0.154, 0.023 radii at 32-, 64-
and 128-equivalent -- which is the honest statement of how the formulation
behaves on unstructured meshes.

Arrows are scaled within each panel, because the peak velocities span two orders
of magnitude and a common length would leave the converged panels blank;
magnitude is carried by the colour, on one logarithmic scale shared by all six.
Velocity is taken at cell centres on the mid-plane in y and interpolated onto a
common grid, which also stops the polyhedral panels looking noisier merely
because their cell centres are irregular.

Ca and the pressure-jump error are read from analyze_droplet_traces, i.e. from
the per-step spurious.dat over the second half of each run, so the figure and
the tables cannot drift apart; a dagger is attached by the same steadiness test
the tables use. max|u| is the instantaneous value in the snapshot shown, and it
differs from Ca where the run is unsteady, which is itself the signal.

Centroid displacement and area-equivalent radius come from the raw volume
fraction rather than from the alpha = 1/2 contour, for the reason in droplet().

WHICH RUNS CAN BE USED. The four hexahedral run directories on the cluster share
one output directory -- `Sub` is a symlink to `meshes/Sub` in cavity32,
cavity32_corr, cavity64 and cavity128 alike -- so each overwrote the previous
one's field files. Only the last writer of a given filename survives, which for
the three shown here is the _corr run in each case (their mtimes match their own
spurious.dat). The flux-omitted hexahedral fields are gone, which is why this
figure is a refinement sequence rather than a with/without pair. The per-step
traces live in the run directories and are unaffected.
"""

import os
import sys

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata

import analyze_droplet_traces as adt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "Data", "StaticDroplet")
R0, CX, CZ = 0.4, 1.0, 1.0
NG = 26

CASES = [
    ("(a) hexahedral $32$",      "cavity32_corr",
     "cavity32_corr/cavity32-ts012000.pvtu",       "12"),
    ("(b) hexahedral $64$",      "cavity64_corr",
     "cavity64_corr/cavity64-ts012000.pvtu",       "12"),
    ("(c) hexahedral $128$",     "cavity128_corr",
     "cavity128_corr/cavity128-ts012000.pvtu",     "12"),
    ("(d) polyhedral 32-equiv",  "poly32_rc_sm4",
     "poly32_rc_sm4/tri_32_dual-ts012000.vtu",     "4"),
    ("(e) polyhedral 64-equiv",  "poly64_rc_sm4",
     "poly64_rc_sm4/cavity_tri_dual-ts012000.vtu", "4"),
    ("(f) polyhedral 128-equiv", "poly128_rc_sm4",
     "poly128_rc_sm4/tri_128_dual-ts012000.vtu",   "4"),
]


def midplane(path):
    """Cell-centre x, z, u, w and volume fraction on the central y layer."""
    m = pv.read(os.path.join(ROOT, path))
    c = np.asarray(m.cell_centers().points, float)
    vel = np.asarray(m.cell_data["Velocity [m/s]"], float)
    vof = np.asarray(m.cell_data["Vof Sharp [1]"], float)
    ys = np.unique(np.round(c[:, 1], 9))
    sel = np.isclose(c[:, 1], ys[len(ys) // 2], atol=1e-9)
    return c[sel, 0], c[sel, 2], vel[sel, 0], vel[sel, 2], vof[sel]


def droplet(path):
    """Centroid displacement and area-equivalent radius from the raw field.

    Volume-weighted over the whole mesh rather than taken from the alpha = 1/2
    contour. A contour drawn from cell centres stops at the outermost centre, so
    for a droplet touching a wall it misses the last half cell and under-reads
    the area by about 16% -- which looked like lost volume until checked this
    way. The volume-weighted form has no such edge and reproduces the conserved
    a_vof of the solver trace exactly.
    """
    m = pv.read(os.path.join(ROOT, path))
    c = np.asarray(m.cell_centers().points, float)
    w = (np.asarray(m.cell_data["Vof Sharp [1]"], float)
         * np.asarray(m.cell_data["Grid Cell Volume [m^3]"], float))
    depth = m.bounds[3] - m.bounds[2]
    cx, cz = (w * c[:, 0]).sum() / w.sum(), (w * c[:, 2]).sum() / w.sum()
    return (np.hypot(cx - CX, cz - CZ) / R0,
            np.sqrt((w.sum() / depth) / np.pi) / R0, cx, cz)


def main():
    data = []
    for label, run, path, cyc in CASES:
        if not os.path.exists(os.path.join(ROOT, path)):
            sys.exit(f"missing field file: {path}")
        s = adt.stats(run)
        if s is None:
            sys.exit(f"missing trace: {run}")
        data.append((label, path, cyc, s) + midplane(path))

    hi = max(np.hypot(u, w).max() for *_, u, w, _ in data)
    norm = LogNorm(vmin=hi * 1e-4, vmax=hi)

    g = np.linspace(0.02, 1.98, NG)
    GX, GZ = np.meshgrid(g, g)
    th = np.linspace(0, 2 * np.pi, 400)

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 9.8))
    fig.subplots_adjust(hspace=0.40, wspace=0.26)
    print(f"{'panel':<32}{'dp err':>17}{'Ca':>10}{'max|u|':>9}"
          f"{'moved/R':>9}{'R_eq/R':>8}")
    for ax, (label, path, cyc, s, x, z, u, w, vof) in zip(axes.ravel(), data):
        pts = np.column_stack([x, z])
        GU = griddata(pts, u, (GX, GZ), method="linear", fill_value=0.0)
        GW = griddata(pts, w, (GX, GZ), method="linear", fill_value=0.0)
        mag = np.hypot(GU, GW)
        q = ax.quiver(GX, GZ, GU, GW, np.maximum(mag, norm.vmin), norm=norm,
                      cmap="viridis", scale=22.0 * mag.max(), width=0.0050,
                      pivot="mid")

        ax.tricontour(Triangulation(x, z), vof, levels=[0.5],
                      colors="crimson", linewidths=1.8)
        ax.plot(CX + R0 * np.cos(th), CZ + R0 * np.sin(th), "--",
                color="k", lw=1.1, alpha=0.7)

        moved, r_eq, cx, cz = droplet(path)
        umax = float(np.hypot(u, w).max())
        if moved > 0.25:
            ax.plot(cx, cz, "x", color="crimson", ms=9, mew=2.2)
            ax.annotate(f"centroid displaced ${moved:.2f}\\,R$",
                        xy=(cx, cz), xytext=(0.04, 0.80),
                        textcoords="axes fraction", fontsize=8.5,
                        color="crimson",
                        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.1))
        dag = "^\\dagger" if s["rng"] > adt.UNSTEADY else ""
        tag = f", {cyc} smoothing cycles"
        ax.set_title(f"{label}{tag}\n$\\Delta p\\ {s['err']:+.2f} \\pm "
                     f"{100*s['std']/adt.EXACT:.2f}{dag}$,  "
                     f"$\\mathrm{{Ca}} = {s['ca']:.2e}$,  "
                     f"$\\max|\\mathbf{{u}}| = {umax:.1f}$",
                     fontsize=8.8, loc="left")
        ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal")
        ax.set_xlabel("$x$ [m]"); ax.set_ylabel("$z$ [m]")
        ax.tick_params(labelsize=8)
        print(f"  {label+tag:<30}{s['err']:>10.2f} +-{100*s['std']/adt.EXACT:5.2f}"
              f"{s['ca']:>10.2e}{umax:>9.2f}{moved:>9.3f}{r_eq:>8.4f}")

    cb = fig.colorbar(q, ax=axes, orientation="horizontal",
                      fraction=0.045, pad=0.09, aspect=50)
    cb.set_label(r"$|\mathbf{u}|$ [m s$^{-1}$], shared logarithmic scale "
                 r"(arrow lengths are normalised within each panel)",
                 fontsize=9)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "pics", f"spurious_currents.{ext}"),
                    dpi=200, bbox_inches="tight")
    print("\nwritten: pics/spurious_currents.{pdf,png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
