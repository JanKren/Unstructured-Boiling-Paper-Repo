#!/usr/bin/env python3
"""One figure in place of three: T, m_dot and |grad T| on the same cross-section.

Figures 6, 7 and 10 of the manuscript were each a 2x3 array of the SAME
z = 0 cross-section of the structured and polyhedral 125^3 bubbles at three
instants, painted with temperature, mass transfer rate and gradient magnitude
respectively.  Three arrays of six panels to make one point per row.  This
collapses them to a 3x2 grid at a single instant -- rows are the three fields,
columns the two topologies -- which is what the accompanying text actually
argues from, and saves roughly two pages.

Both topologies are read at dt = 2 us and t = 1.0 ms, so the comparison is at
matched time and matched step, and the structured data is the same
`aniso-125-off` run that Table 7 and the convergence tables use.  The earlier
figures took the structured panels from `Scriven-Struct-125-smallDt`, a
different solver build; keeping the figure on the table's own run removes that
inconsistency.

`Vof MassTransfer [kg/m^3/s]` is mislabelled in the solver output -- the stored
quantity is kg/s PER CELL, so it is divided by the cell volume here to get the
volumetric rate the axis claims.

Usage:  python3 plot_scriven_panels.py [--out pics/scriven_panels.pdf]
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pyvista as pv
from matplotlib.collections import PolyCollection

from analyze_scriven_fields import extract_slice

import scriven_reference

T_SAT, T_INF = 100.0, 101.25
# The reference solution is scriven_reference's, not a local copy.  This file
# carried beta = 4.06022 and a diffusivity built on k_l = 0.677, so its
# analytical circle stood at 116.39 um where the section now normalises by
# 116.64 -- 0.22% small, on the very circle the caption asks the reader to
# judge the interface against.
R0 = scriven_reference.R0
HALF = 148.0                      # plotted half-width [um]
NGRID = 400

STRUCT = "Data/Aniso/aniso-125-off/Scriven-Struct-125-ts000500.pvtu"
POLY = os.path.join(os.environ.get("MERLIN_DIR", "/home/jan/archive/Scriven-Merlin"), "Scriven-Poly-125/Scriven-Poly-125_dual-ts000500.pvtu")
T_PLOT = 1.0e-3


# metres, and the virtual origin is already inside it
analytical_radius = scriven_reference.r_scriven


def load(path):
    """Slice at z = 0 and return cell-centred fields on that plane."""
    s = extract_slice(path, normal="z")
    c = s.cell_centers().points
    vol = np.asarray(s.cell_data["Grid Cell Volume [m^3]"], float)
    g = np.asarray(s.cell_data["Temperature Gradients [K/m]"], float)
    return dict(
        x=c[:, 0] * 1e6, y=c[:, 1] * 1e6,
        T=np.asarray(s.cell_data["Temperature [K]"], float),
        vof=np.asarray(s.cell_data["Vof Sharp [1]"], float),
        vel=np.asarray(s.cell_data["Velocity [m/s]"], float),
        # stored per cell despite the label; divide to get kg/(m^3 s)
        mdot=np.asarray(s.cell_data["Vof MassTransfer [kg/m^3/s]"], float) / vol,
        gmag=np.linalg.norm(g, axis=1),
        # kept so the mass-transfer row can be drawn as the cells themselves
        slc=s,
    )


def cell_polygons(s, idx):
    """xy vertices [um] of the sliced cells at `idx`, in VTK winding order."""
    return [s.get_cell(int(i)).points[:, :2] * 1e6 for i in idx]


def to_grid(d, field):
    gi = np.linspace(-HALF, HALF, NGRID)
    X, Y = np.meshgrid(gi, gi)
    return X, Y, griddata(np.column_stack([d["x"], d["y"]]), field, (X, Y),
                          method="linear")


def panel(ax, d, field, levels, cmap, R_um, quiver=False, cells=False):
    """One cross-section panel.

    Fields that fill the domain are contoured on an interpolated grid.  The
    mass transfer rate is not one of them: it is identically zero outside a
    one- to two-cell band, so only ~400 of the ~15000 slice cells carry it, and
    interpolating that onto a 400x400 grid smears the ring until its angular
    variation is invisible.  Those cells are drawn directly instead -- as the
    polygons the slice actually has, not as markers at their centres.  Drawing
    them as cells says what the row is about, that the band is one to two cells
    wide and that the two topologies tile it differently, and it puts the
    extent in DATA units, so an inset magnifies a cell instead of showing
    fewer dots at the same point size.
    """
    if cells:
        sel = np.flatnonzero(field > 0)
        ax.set_facecolor("0.97")
        cf = PolyCollection(cell_polygons(d["slc"], sel), cmap=cmap,
                            edgecolors="face", linewidths=0.15)
        cf.set_array(field[sel])
        cf.set_clim(levels[0], levels[-1])
        ax.add_collection(cf)
    else:
        X, Y, F = to_grid(d, field)
        cf = ax.contourf(X, Y, F, levels=levels, cmap=cmap, extend="both")
    XV, YV, V = to_grid(d, d["vof"])
    ax.contour(XV, YV, V, levels=[0.5], colors="k",
               linewidths=1.0 if cells else 1.4)
    th = np.linspace(0, 2 * np.pi, 240)
    ax.plot(R_um * np.cos(th), R_um * np.sin(th), "w--", linewidth=1.2)
    if quiver:
        st = max(1, len(d["x"]) // 900)
        ax.quiver(d["x"][::st], d["y"][::st],
                  d["vel"][::st, 0], d["vel"][::st, 1],
                  color="k", alpha=0.55, scale=1.6, width=0.004)
    ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF)
    ax.set_aspect("equal")
    ax.set_xticks([-100, 0, 100]); ax.set_yticks([-100, 0, 100])
    return cf


def diag_radius(d, half_angle=10.0):
    """Radius of the alpha = 1/2 interface where it crosses the 45 degree
    diagonal, measured from the slice itself.

    Each column is framed on its OWN interface.  One shared window cannot
    centre both: the structured bubble reaches 129 um on the diagonal and
    the polyhedral one about 115, so a window placed between them puts the
    polyhedral interface in a corner of its inset and shows only part of
    it.  The window SIZE stays common, so the magnification is identical
    and the two insets remain comparable.
    """
    r = np.hypot(d["x"], d["y"])
    th = np.degrees(np.arctan2(d["y"], d["x"])) % 360.0
    near = np.abs(((th - 45.0 + 180.0) % 360.0) - 180.0) < half_angle
    m = (d["vof"] > 0.35) & (d["vof"] < 0.65) & near
    return float(np.mean(r[m]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pics/scriven_panels.pdf")
    a = ap.parse_args()

    # analyze_scriven_fields sets axes.labelsize and the tick sizes to 24 at
    # import time, and font.size does not override an explicit setting, so each
    # one has to be reset by name.
    plt.rcParams.update({"font.size": 9, "font.family": "serif",
                         "savefig.dpi": 300, "axes.grid": False,
                         "axes.labelsize": 9, "axes.titlesize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "figure.titlesize": 10})

    ds = load(STRUCT)
    dp = load(POLY)
    R_um = analytical_radius(T_PLOT) * 1e6

    # Common levels per row so the two topologies are directly comparable.
    gmax = np.nanpercentile(np.concatenate([ds["gmag"], dp["gmag"]]), 99.5)
    # m_dot is identically zero outside the one- to two-cell interface band, so
    # a percentile over all cells is set by the zeros and flattens the ring.
    # Scale on the band itself.
    _md = np.abs(np.concatenate([ds["mdot"], dp["mdot"]]))
    _nz = _md[_md > 0]
    mmin, mmax = np.percentile(_nz, 2), np.percentile(_nz, 98)
    rows = [
        ("Temperature [$^{\\circ}$C]", "T", np.linspace(T_SAT, T_INF, 30),
         "RdYlBu_r", True, False),
        ("$\\dot{m}$ [$\\mathrm{kg\\,m^{-3}s^{-1}}$]", "mdot",
         np.linspace(mmin, mmax, 30), "viridis", False, True),
        ("$|\\nabla T|$ [$\\mathrm{K\\,m^{-1}}$]", "gmag",
         np.linspace(0, gmax, 30), "magma", False, False),
    ]

    # Insets follow Figure 6 exactly: a large lower-left panel, a square
    # window on the interface where it crosses the 45 degree diagonal, and
    # indicate_inset_zoom's connectors kept.  From upper-right to lower-left
    # those read as a pull-out; the earlier arrangement put the window on the
    # +x axis and the inset opposite it, which made them full-width
    # horizontals across every panel.
    #
    # One window SIZE for both columns, so the magnification is identical and
    # the insets stay comparable, but each centred on its own interface (see
    # diag_radius).  At 50 um the window holds the ~13 um thermal layer either
    # side of the interface and magnifies about 2.5x, close to Figure 6's 2.8x.
    INSET_BOX = [0.02, 0.02, 0.42, 0.42]
    ZOOM = 50.0
    ZC = {}                           # 45 degree point, per column

    fig, axes = plt.subplots(3, 2, figsize=(6.6, 9.4))
    for r, (label, key, lev, cmap, quiv, scat) in enumerate(rows):
        for c, (d, topo) in enumerate(((ds, "Structured"), (dp, "Polyhedral"))):
            cf = panel(axes[r, c], d, d[key], lev, cmap, R_um,
                       quiver=quiv, cells=scat)
            # Magnified inset on the interface and the thermal layer it
            # carries, as Figure 6 already does.  panel() is re-called on
            # the inset axis so the inset shows the same construction as
            # the panel, then the window and ticks are overridden.
            #
            # The source region is marked with a plain rectangle rather than
            # indicate_inset_zoom.  Its leader lines ran the full width of
            # every panel, because the window is on the +x interface and the
            # inset has to sit in the far corner to stay off the interface,
            # and in the mdot row they sprawled across an empty field.
            if topo not in ZC:
                ZC[topo] = diag_radius(d) / np.sqrt(2.0)
            z0 = ZC[topo] - ZOOM / 2
            axin = axes[r, c].inset_axes(INSET_BOX)
            # The mass-transfer row needs no special handling here any more.
            # Its cells are polygons in data units, so the inset magnifies a
            # cell rather than showing fewer markers at the same point size,
            # and the one-to-two-cell band reads directly.
            panel(axin, d, d[key], lev, cmap, R_um,
                  quiver=False, cells=scat)
            axin.set_xlim(z0, z0 + ZOOM)
            axin.set_ylim(z0, z0 + ZOOM)
            axin.set_xticks([])
            axin.set_yticks([])
            axes[r, c].indicate_inset_zoom(axin, edgecolor='black',
                                           linewidth=1.5)
            if r == 0:
                axes[r, c].set_title(f"{topo} $125^3$", fontsize=10)
            if c == 1:
                axes[r, c].set_yticklabels([])
            else:
                axes[r, c].set_ylabel(r"$y$ [$\mu$m]")
            if r < 2:
                axes[r, c].set_xticklabels([])
            else:
                axes[r, c].set_xlabel(r"$x$ [$\mu$m]")
        cb = fig.colorbar(cf, ax=axes[r, :].tolist(), fraction=0.046, pad=0.02)
        cb.set_label(label, fontsize=9)
        cb.ax.tick_params(labelsize=8)
        for lbl, ax in zip("abcdef"[2 * r:2 * r + 2], axes[r, :]):
            ax.text(0.03, 0.95, f"({lbl})", transform=ax.transAxes,
                    fontsize=9, va="top",
                    bbox=dict(fc="w", ec="none", alpha=0.75, pad=1.5))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  t = {T_PLOT*1e3:.2f} ms, R_analytical = {R_um:.2f} um")
    print(f"  structured slice {len(ds['x']):6d} cells, polyhedral {len(dp['x']):6d}")
    print(f"  |grad T| 99.5th pct {gmax:.3e} K/m,  m_dot band {mmin:.3e}-{mmax:.3e} kg/m^3/s")
    print(f"  figure -> {a.out}")


if __name__ == "__main__":
    main()
