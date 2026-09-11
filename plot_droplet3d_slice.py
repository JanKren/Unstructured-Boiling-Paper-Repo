#!/usr/bin/env python3
"""Spurious currents around the three-dimensional static droplet, on a slice.

    python3 plot_droplet3d_slice.py [--mode cbrt|column|panel|sqrt|shared]

and writes pics/droplet3d_slice_arrows.  The earlier pics/droplet3d_slice,
which put magnitude in colour and drew every arrow at unit length, is left
where it is; this script no longer produces it.

The quasi-two-dimensional predecessor of this figure could plot the mesh's own
mid-layer directly, because the domain was three or four cells thick.  Here the
droplet is a sphere in a cube, so the panels are a genuine cut: the plane
y = 1 through the droplet centre, with the velocity sampled onto a common
Cartesian grid so that the hexahedral and polyhedral fields can be compared
arrow for arrow.

WHY A CUT RATHER THAN A LAYER.  Slicing is not free of choices.  A polyhedral
dual has no cell layer aligned with any plane, so `slice` interpolates, and the
arrows are therefore an interpolation of the field rather than the field.  The
alternative -- picking the cells whose centres lie nearest the plane -- keeps
raw values but samples a ragged surface, and on a dual that surface wanders by
most of a cell.  The interpolation is the lesser distortion for a figure whose
purpose is the pattern and the magnitude, and the pressure-jump table carries
the quantitative statements regardless.

MAGNITUDE IS CARRIED BY ARROW LENGTH, not by colour.  Length is the channel a
reader already reads as "how fast", and it makes the ink density of a panel a
second, redundant reading of the same quantity: a quiet field is short arrows
on a mostly empty panel, a loud one is long arrows that crowd.  The colour axis
is then free, and is spent on nothing -- one ink, so the figure survives
greyscale printing.

THE CHOICE THAT MATTERS IS THE LENGTH MAP, and it is a real trade, because the
field spans five decades.  Measured on the interpolated grid, the panel peaks
are 0.33 and 0.11 m/s hexahedral against 6.79 and 6.02 polyhedral, and within
any one panel the lower quartile is 30 to 100 times below that panel's peak.
Four maps were built and compared:

  panel   length proportional to |u|, rescaled per panel.  The conventional
          choice, and the one that shows structure best: it is what makes the
          four-fold lobed pattern of the hexahedral parasitic flow, aligned
          with the coordinate axes, visible at all.  Nothing is comparable
          between panels except by reading the four printed peaks.
  column  length proportional to |u|, one scale down each column.  Keeps
          proportionality and most of the structure, and makes the refinement
          claim directly readable -- the hexahedral peak arrow falls to 32%
          between the rows while the polyhedral only falls to 89% -- but says
          nothing across topologies.
  shared  length proportional to |u|, one scale throughout.  Honest and
          useless: the hexahedral peak is 1.6-4.9% of the longest arrow and
          their lower quartile is 2e-4 of it, so panels (a) and (c) go blank.
  sqrt    length proportional to |u|^(1/2), shared.  Hexahedral peaks reach
          12-20%; the hexahedral bulk is still sub-pixel and panel (c) is
          nearly empty.
  cbrt    length proportional to |u|^(1/3), shared.  DEFAULT.  Hexahedral peaks
          reach 25-37% and their lower quartile 6-8%, so both topologies stay
          legible on one scale and the 20-fold gap is still plainly visible.
          Chosen because that gap is what this appendix is for; it is paid for
          in within-panel contrast, which is where `panel` is better.

A power-law length map is monotone but not proportional, so the legend carries
four reference arrows at decade values rather than one, which is what lets a
reader recover the compression.  Do not quote a velocity off an arrow: the
peaks are in the caption and the averages in the table.

Two resolutions are shown, R/h = 12.8 and 25.6, so that the columns are the two
topologies and the rows the refinement.  The R/h = 6.4 runs are not: at six
cells per radius neither mesh resolves two principal curvatures at once, and
what those runs measure is the tuning of the smoothing operator rather than the
surface tension treatment.
"""

import argparse
import os
import sys

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.interpolate import griddata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("DROPLET_DIR", "/home/jan/runs/Droplet3D")
# The manuscript's figure.  Kept distinct from the earlier `droplet3d_slice`,
# which carried magnitude in colour on unit-length arrows; that file stays in
# pics/ as the published version of this appendix's first submission.
BASE = "droplet3d_slice_arrows"
R0, C = 0.4, 1.0            # droplet radius and box centre
# NG and the grid points below are deliberately unchanged from the colour
# version of this figure: the caption quotes the peak velocity of each panel,
# and that peak is a property of this sampling, not of the field.  Moving the
# grid silently restates four published numbers.
NG = 26                     # arrows per side
ARROW = 2.2 / NG            # longest arrow, as a fraction of the axes width
INK = "#1f3b63"             # single ink; magnitude is length, not colour

# Reference magnitudes drawn in the legend strip.  Chosen as decades spanning
# the whole field so that the curvature of the length map is readable off the
# figure itself; a power law needs more than one key to be interpretable.
KEYS = (0.01, 0.1, 1.0, 5.0)

# Columns are the two topologies, rows the two resolutions.  The polyhedral
# R/h = 25.6 field is written by a one-step restart from the run's last backup,
# because that run was stopped at its wall and RESULTS_SAVE_INTERVAL fires only
# at the end; if it is absent the panel is drawn as a placeholder rather than
# silently dropped.
CASES = [
    ("(a) hexahedral, $R/h = 12.8$",  "hex64_result/drop3d_hex64-ts033255.pvtu"),
    ("(b) polyhedral, $R/h = 12.8$",  "poly64_result/drop3d_poly64_dual-ts033255.pvtu"),
    ("(c) hexahedral, $R/h = 25.6$",  "hex128_result/drop3d_hex128-ts094060.pvtu"),
    # NOTE: this file is named ts000000 but holds the ts072000 state.  The run was
    # stopped at its wall with no field written, so the field was recovered by a
    # one-step restart from the last backup; that restart resets the step counter,
    # so the solver wrote it under the initial-step name.  Verified by content:
    # max|u| = 16.8 m/s, which is the run's spurious current, not a quiescent start.
    ("(d) polyhedral, $R/h = 25.6$",  "poly128_result/drop3d_poly128_dual-ts000000.pvtu"),
]

# gamma is the exponent of the length map; scope says how widely one scale is
# reused.  gamma = 1 is proportional length, gamma < 1 compresses.  "figure"
# normalises on the largest arrow anywhere, "column" on the largest in that
# topology, "panel" on the panel's own peak.
MODES = {
    "cbrt":   (1.0 / 3.0, "figure"),
    "sqrt":   (0.5,       "figure"),
    "shared": (1.0,       "figure"),
    "column": (1.0,       "column"),
    "panel":  (1.0,       "panel"),
}

plt.rcParams.update({"font.size": 10, "font.family": "serif"})


def slab(path):
    """Cell data on the y = C plane, plus the alpha = 1/2 curve.

    The curve is contoured from the volume fraction promoted to the mesh nodes.
    Contouring the cell centres directly can only join about forty samples
    around the droplet at the coarser level and renders it as a polygon; node
    promotion is itself a smoothing, so the curve understates the roughness,
    which is why the roughness is quoted numerically in the text instead.
    """
    m = pv.read(os.path.join(ROOT, path))
    s = m.slice(normal="y", origin=(C, C, C))
    c = np.asarray(s.cell_centers().points, float)
    vel = np.asarray(s.cell_data["Velocity [m/s]"], float)
    vof = np.asarray(s.cell_data["Vof Sharp [1]"], float)
    ct = s.cell_data_to_point_data().contour([0.5], scalars="Vof Sharp [1]")
    seg = np.asarray(ct.points)[:, [0, 2]] if ct.n_points else np.empty((0, 2))
    return c[:, 0], c[:, 2], vel[:, 0], vel[:, 2], vof, seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cbrt", choices=sorted(MODES))
    ap.add_argument("--tag", default="-",
                    help='suffix for the output name; "-" is the canonical '
                         f'pics/{BASE} used by the manuscript')
    ap.add_argument("--outdir", default=os.path.join(HERE, "pics"))
    a = ap.parse_args()
    gamma, scope = MODES[a.mode]

    data, vmax = [], 0.0
    for lab, p in CASES:
        if not os.path.exists(os.path.join(ROOT, p)):
            data.append((lab, None)); continue
        x, z, u, w, f, seg = slab(p)
        # Samples on the domain edge fall outside the convex hull of the cell
        # centres.  They are left NaN rather than filled with zero, so quiver
        # draws nothing there: a zero arrow would read as "no flow" where the
        # truth is "no data".  The interior samples, and hence the panel peaks,
        # are unchanged.
        g = np.linspace(0.0, 2.0, NG)
        GX, GZ = np.meshgrid(g, g)
        pts = np.column_stack([x, z])
        GU = griddata(pts, u, (GX, GZ), method="linear", fill_value=np.nan)
        GW = griddata(pts, w, (GX, GZ), method="linear", fill_value=np.nan)
        mag = np.hypot(GU, GW)
        vmax = max(vmax, np.nanmax(mag))
        data.append((lab, (x, z, f, seg, GX, GZ, GU, GW, mag)))
    if all(P is None for _, P in data):
        sys.exit("no field output found")

    # One reference magnitude per panel, in the ravel order of the axes grid.
    # Columns are the topologies, so a column reference compares the two
    # refinements of one topology and says nothing across topologies.
    peak = [np.nanmax(P[-1]) if P is not None else 0.0 for _, P in data]
    col = [max(peak[0], peak[2]), max(peak[1], peak[3])]
    if scope == "figure":
        refs = [vmax] * 4
    elif scope == "column":
        refs = [col[0], col[1], col[0], col[1]]
    else:
        refs = peak

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 9.0))
    th = np.linspace(0, 2 * np.pi, 200)
    quivers = []

    for i, (ax, (lab, P)) in enumerate(zip(axes.ravel(), data)):
        if P is None:
            ax.text(0.5, 0.5, "field output\npending", ha="center", va="center",
                    fontsize=9, color="0.45", transform=ax.transAxes)
            ax.set_title(lab, fontsize=10, loc="left")
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            quivers.append(None)
            continue
        x, z, f, seg, GX, GZ, GU, GW, mag = P

        # Direction from the field, length from the map.  Dividing by the
        # magnitude first keeps the two independent, so the length map can be
        # changed without touching the direction.
        ref = refs[i]
        n = np.where(mag > 0, mag, 1.0)
        s = np.where(mag > 0, (mag / ref) ** gamma, 0.0)
        # scale_units="width": a vector of length `scale` spans the axes width,
        # so this puts the reference magnitude at ARROW of the width.
        q = ax.quiver(GX, GZ, GU / n * s, GW / n * s, color=INK,
                      angles="xy", scale=1.0 / ARROW, scale_units="width",
                      width=0.0042, headwidth=3.2, headlength=4.0,
                      headaxislength=3.4, minshaft=1.0, minlength=0.0,
                      pivot="tail", zorder=2)
        quivers.append(q)

        if len(seg):
            o = np.argsort(np.arctan2(seg[:, 1] - C, seg[:, 0] - C))
            ax.plot(np.append(seg[o, 0], seg[o[0], 0]),
                    np.append(seg[o, 1], seg[o[0], 1]), color="C3", lw=1.7,
                    zorder=4, path_effects=[pe.Stroke(linewidth=3.2,
                                                      foreground="white"),
                                            pe.Normal()])
        ax.plot(C + R0 * np.cos(th), C + R0 * np.sin(th), color="0.2",
                lw=1.0, ls="--", zorder=4,
                path_effects=[pe.Stroke(linewidth=2.6, foreground="white"),
                              pe.Normal()])
        ax.set_title(lab, fontsize=10, loc="left")
        ax.set_aspect("equal")
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])

        # Anything narrower than one scale for the whole figure needs the peak
        # printed, or the panels are unlabelled scales the reader cannot join.
        if scope != "figure":
            ax.text(0.985, 0.015,
                    rf"$\max|\mathbf{{u}}| = {np.nanmax(mag):.2f}$ m s$^{{-1}}$",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                    bbox=dict(fc="white", ec="0.7", lw=0.5, pad=1.8))

    fig.subplots_adjust(bottom=0.115, hspace=0.16, wspace=0.13)
    live = next(q for q in quivers if q is not None)

    if scope == "column":
        # One key per column, at a round value inside that column's range.
        fig.text(0.5, 0.012, "arrow length $\\propto |\\mathbf{u}|$, "
                 "one scale down each column", fontsize=8.5,
                 ha="center", va="center", color="0.35")
        for j, (xk, v) in enumerate(zip((0.185, 0.605),
                                        (0.3, 5.0))):
            q = quivers[j]
            if q is None:
                continue
            axes.ravel()[j].quiverkey(
                q, xk, 0.049, v / col[j],
                rf"{v:g} m s$^{{-1}}$", coordinates="figure", labelpos="E",
                labelsep=0.035, fontproperties={"size": 9.5}, color=INK)
    elif scope == "figure":
        # One legend for the whole figure.  Several reference arrows, because a
        # power-law map cannot be read off a single one.
        fig.text(0.105, 0.049, r"$|\mathbf{u}|$ [m s$^{-1}$]", fontsize=9.5,
                 ha="left", va="center")
        kax = axes.ravel()[0]
        for i, v in enumerate(KEYS):
            kax.quiverkey(live, 0.30 + 0.165 * i, 0.049, (v / vmax) ** gamma,
                          rf"{v:g}", coordinates="figure", labelpos="E",
                          labelsep=0.035, fontproperties={"size": 9.5},
                          color=INK)
        fig.text(0.5, 0.012,
                 rf"arrow length $\propto |\mathbf{{u}}|^{{{gamma:.2f}}}$, "
                 r"one scale for all four panels", fontsize=8.5,
                 ha="center", va="center", color="0.35")
    else:
        fig.text(0.5, 0.035, "arrow length $\\propto |\\mathbf{u}|$, "
                 "rescaled to the peak of each panel", fontsize=9,
                 ha="center", va="center", color="0.35")

    base = BASE if a.tag == "-" else f"droplet3d_slice_{a.tag or a.mode}"
    os.makedirs(a.outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.outdir, f"{base}.{ext}"),
                    dpi=200, bbox_inches="tight")

    print(f"  mode={a.mode}  gamma={gamma:.3f}  scope={scope}  "
          f"global max={vmax:.3f} m/s")
    print(f"  {'panel':<32}{'max |u|':>11}{'mean |u|':>11}{'peak arrow':>12}")
    for i, (lab, P) in enumerate(data):
        if P is None:
            print(f"  {lab:<32}{'--':>11}{'--':>11}{'--':>12}"); continue
        m = P[-1]
        pk = (np.nanmax(m) / refs[i]) ** gamma
        print(f"  {lab:<32}{np.nanmax(m):>11.3e}{np.nanmean(m):>11.3e}"
              f"{pk:>11.1%}")
    print(f"written: {a.outdir}/{base}.{{pdf,png}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
