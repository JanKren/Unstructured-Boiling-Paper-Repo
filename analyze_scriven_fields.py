#!/usr/bin/env python3
"""
Scriven bubble growth: temperature contours, velocity vectors, and mesh.

YS review comment: Show instantaneous temperature and velocity vector
distributions together with mesh for the Scriven problem.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
import pyvista as pv
import os

plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'font.family': 'sans-serif',
    'mathtext.fontset': 'dejavusans',
    'mathtext.default': 'regular',
    'savefig.dpi': 300,
    'axes.grid': False,
    'lines.linewidth': 1.5,
})

# ===========================================================================
#  Paths and physical parameters
# ===========================================================================
SCRIVEN_DIR = os.environ.get("SCRIVEN_DIR",
                          "/home/jan/runs/tflows-vof/Validation/Scriven")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", os.path.join(os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling"), "pics"))

# Scriven analytical parameters
BETA    = 4.06022
ALPHA_L = 0.679 / (958.4 * 4216.0)  # thermal diffusivity [m^2/s]
R0      = 5.0e-5                      # initial bubble radius [m]
T0      = R0**2 / (4.0 * BETA**2 * ALPHA_L)  # virtual origin time [s]
DT_STRUCT = 1.0e-6                     # structured mesh time step [s]
DT_POLY   = 2.0e-6                     # polyhedral mesh time step [s]
T_SAT   = 100.0                       # K
T_INF   = 101.25                      # K (T_sat + dT)


def analytical_radius(t):
    """Scriven bubble radius R(t) = 2*beta*sqrt(alpha_l * t)."""
    return 2.0 * BETA * np.sqrt(ALPHA_L * t)


def extract_slice(pvtu_path, normal='z'):
    """Read PVTU and slice through center perpendicular to 'normal' axis.

    T-Flows polyhedral .pvtu output does not set vtkGhostType, so every
    piece writes its real cells plus buffer copies of neighbor cells.
    After slicing, the same physical cell then appears once per piece
    that owns a copy, producing the visible ring-shaped seams at
    processor boundaries. Deduplicating slice cells by centroid
    collapses the buffer copies onto their real counterpart.
    """
    mesh = pv.read(pvtu_path)
    if 'vtkGhostType' in mesh.cell_data:
        real_cells = mesh.cell_data['vtkGhostType'] == 0
        mesh = mesh.extract_cells(real_cells)
    sliced = mesh.slice(normal=normal, origin=(0, 0, 0))

    if sliced.n_cells > 0:
        cc = sliced.cell_centers().points
        _, unique_idx = np.unique(np.round(cc, 9), axis=0,
                                  return_index=True)
        if len(unique_idx) < sliced.n_cells:
            mask = np.zeros(sliced.n_cells, dtype=bool)
            mask[unique_idx] = True
            sliced = sliced.extract_cells(mask)

    # Merge coincident points across partition boundaries. PyVista does
    # not merge points when reading a multi-piece PVTU, so the same
    # physical point at a partition interface exists once per piece with
    # a different index. extract_all_edges() deduplicates edges by index
    # pairs, not coordinates, so partition-boundary edges get written
    # once per side and then alpha-composite into a visibly darker line.
    # Merging points first makes partition edges share a single index
    # pair and VTK collapses them.
    if hasattr(sliced, 'clean'):
        sliced = sliced.clean(tolerance=1e-10, absolute=True)
    return sliced


def get_mesh_edges(sliced):
    """Extract mesh edges from a sliced PolyData for plotting."""
    edges = sliced.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=True,
        non_manifold_edges=False
    )
    lines = []
    pts = edges.points
    # Parse VTK lines connectivity
    i = 0
    conn = edges.lines
    while i < len(conn):
        npts = conn[i]
        i += 1
        seg_pts = conn[i:i+npts]
        for j in range(npts - 1):
            p0 = pts[seg_pts[j]]
            p1 = pts[seg_pts[j+1]]
            lines.append([(p0[0], p0[1]), (p1[0], p1[1])])
        i += npts
    return lines


def get_cell_edges(sliced):
    """Extract all internal cell edges from the slice for mesh visualization.

    Deduplicates edges by rounded endpoint coordinates. VTK's
    extract_all_edges deduplicates by point index pairs, which does not
    catch edges that share geometry but were written with different
    point indices on either side of a partition interface.
    """
    wireframe = sliced.extract_all_edges()
    pts = wireframe.points
    conn = wireframe.lines
    lines = []
    seen = set()
    i = 0
    while i < len(conn):
        npts = conn[i]
        i += 1
        seg_pts = conn[i:i+npts]
        for j in range(npts - 1):
            p0 = pts[seg_pts[j]]
            p1 = pts[seg_pts[j+1]]
            key = tuple(sorted((
                (round(float(p0[0]), 9), round(float(p0[1]), 9)),
                (round(float(p1[0]), 9), round(float(p1[1]), 9)),
            )))
            if key in seen:
                continue
            seen.add(key)
            lines.append([(p0[0], p0[1]), (p1[0], p1[1])])
        i += npts
    return lines


def plot_single_panel(ax, pvtu_path, t_anal, show_ylabel=False,
                      title='', panel_label=''):
    """Plot temperature contours + velocity vectors + mesh on one axis.

    Uses scipy.griddata interpolation onto a regular grid for clean
    contour plots on both structured and polyhedral meshes.
    Returns the contourf artist (for colorbar).
    """
    from scipy.spatial import cKDTree
    from scipy.interpolate import griddata

    R_anal = analytical_radius(t_anal)

    sliced = extract_slice(pvtu_path, normal='z')
    centers = sliced.cell_centers()
    cx = centers.points[:, 0]
    cy = centers.points[:, 1]

    T = sliced.cell_data['Temperature [K]']
    vof = sliced.cell_data['Vof Sharp [1]']
    vel = sliced.cell_data['Velocity [m/s]']
    vx, vy = vel[:, 0], vel[:, 1]

    cx_um = cx * 1e6
    cy_um = cy * 1e6
    R_um = R_anal * 1e6

    # Interpolate onto a regular grid for clean contours
    ngrid = 400
    yi = np.linspace(-148, 148, ngrid)
    zi = np.linspace(-148, 148, ngrid)
    YI, ZI = np.meshgrid(yi, zi)
    points = np.column_stack([cx_um, cy_um])

    T_grid = griddata(points, T, (YI, ZI), method='linear')
    vof_grid = griddata(points, vof, (YI, ZI), method='linear')

    # Temperature contours
    levels = np.linspace(T_SAT, T_INF, 30)
    cf = ax.contourf(YI, ZI, T_grid, levels=levels, cmap='RdYlBu_r',
                     extend='both')

    # VOF = 0.5 interface contour
    ax.contour(YI, ZI, vof_grid, levels=[0.5], colors='k', linewidths=2)

    # Analytical circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(R_um * np.cos(theta), R_um * np.sin(theta),
            'k-', linewidth=2.5, alpha=0.5)
    ax.plot(R_um * np.cos(theta), R_um * np.sin(theta),
            'w--', linewidth=1.5)

    # Velocity vectors (subsampled on a regular grid)
    vmag = np.sqrt(vx**2 + vy**2)
    max_v = vmag.max()
    if max_v > 0:
        n_arrows = 18
        arr_y = np.linspace(cx_um.min(), cx_um.max(), n_arrows)
        arr_z = np.linspace(cy_um.min(), cy_um.max(), n_arrows)
        ay, az = np.meshgrid(arr_y, arr_z)
        tree = cKDTree(points)
        _, nearest = tree.query(np.column_stack([ay.ravel(), az.ravel()]))
        sig = vmag[nearest] > 0.1 * max_v
        ax.quiver(cx_um[nearest[sig]], cy_um[nearest[sig]],
                  vx[nearest[sig]], vy[nearest[sig]],
                  color='k', alpha=0.7, scale=max_v*20,
                  width=0.004, headwidth=3, headlength=4)

    # Mesh edges (no radial clipping — avoids a visible circle at the
    # diagonal corners of the square viewport where the clip intersects).
    cell_edges = get_cell_edges(sliced)
    cell_edges_um = [[(p[0]*1e6, p[1]*1e6) for p in seg]
                     for seg in cell_edges]
    lc = LineCollection(cell_edges_um, colors='gray',
                        linewidths=0.3, alpha=0.4)
    ax.add_collection(lc)

    ax.set_xlim([-150, 150])
    ax.set_ylim([-150, 150])
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$ [$\mu$m]')
    if show_ylabel:
        ax.set_ylabel(r'$y$ [$\mu$m]')
    ax.set_title(title)
    if panel_label:
        ax.text(0.04, 0.96, panel_label, transform=ax.transAxes,
                ha='left', va='top', fontsize=22,
                bbox=dict(facecolor='white', edgecolor='none',
                          pad=2, alpha=0.85))

    return cf


def plot_scriven_fields():
    """Create the 2x3 Scriven visualization: structured vs polyhedral."""

    print("=" * 60)
    print("  SCRIVEN PROBLEM - Field Visualization")
    print("=" * 60)

    STRUCT_DIR = os.path.join(os.environ.get("MERLIN_DIR", "/home/jan/archive/Scriven-Merlin"), "Scriven-Struct-125-smallDt")
    POLY_DIR   = os.path.join(SCRIVEN_DIR, "Scriven-Poly-125")

    # Select 2 matching physical times (t_sim = 0.5, 1.0 ms)
    # Structured (Merlin, 125^3): dt=1e-6 s -> ts = 500, 1000
    # Poly-125:                   dt=2e-6 s -> ts = 250, 500
    ts_struct = [500, 1000]
    ts_poly   = [250, 500]
    col_labels = ['(a)', '(b)', '(c)', '(d)']

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    cf = None
    for idx in range(2):
        t_sim = ts_struct[idx] * DT_STRUCT
        t_anal = t_sim + T0
        R_anal = analytical_radius(t_anal)
        t_ms = t_anal * 1e3

        # --- Top row: structured mesh ---
        pvtu_struct = os.path.join(
            STRUCT_DIR, f"bubble-ts{ts_struct[idx]:06d}.pvtu")
        if os.path.exists(pvtu_struct):
            label_s = col_labels[idx]
            title_s = f'Structured, $t = {t_ms:.2f}$ ms'
            print(f"  Structured ts={ts_struct[idx]}: t_anal={t_ms:.3f} ms, "
                  f"R={R_anal*1e6:.1f} um")
            cf = plot_single_panel(axes[0, idx], pvtu_struct, t_anal,
                                   show_ylabel=(idx == 0), title=title_s,
                                   panel_label=label_s)
        else:
            print(f"  WARNING: {pvtu_struct} not found")

        # --- Bottom row: polyhedral mesh ---
        pvtu_poly = os.path.join(
            POLY_DIR, f"Scriven-Poly-125_dual-ts{ts_poly[idx]:06d}.pvtu")
        if os.path.exists(pvtu_poly):
            label_p = col_labels[idx + 2]
            title_p = f'Polyhedral, $t = {t_ms:.2f}$ ms'
            print(f"  Polyhedral ts={ts_poly[idx]}: t_anal={t_ms:.3f} ms, "
                  f"R={R_anal*1e6:.1f} um")
            cf = plot_single_panel(axes[1, idx], pvtu_poly, t_anal,
                                   show_ylabel=(idx == 0), title=title_p,
                                   panel_label=label_p)
        else:
            print(f"  WARNING: {pvtu_poly} not found")

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(cf, cax=cbar_ax, label='Temperature [K]')

    plt.subplots_adjust(left=0.07, right=0.90, top=0.95, bottom=0.08,
                        wspace=0.28, hspace=0.40)
    outpath = os.path.join(OUTPUT_DIR, "scriven_fields.png")
    plt.savefig(outpath, bbox_inches='tight')
    plt.savefig(outpath.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n  Saved: {outpath}")
    plt.close()


def plot_scriven_mesh_comparison():
    """Figure 6: the two meshes and the initial condition they start from.

    Drawn at t = 0 rather than mid-run, and for a reason.  The evolved
    comparison at 125^3 -- temperature, interface and thermal layer on both
    topologies -- is Figure 7's subject, so a second figure of the same state
    at another instant says the same thing twice.  At t = 0 the interface is
    the initialised sphere and the temperature is the analytical Scriven
    profile, identical on both meshes by construction, so what separates the
    panels is the mesh alone: this figure introduces the discretisations and
    shows how each resolves a known layer, and Figure 7 shows what becomes of
    it.  It also makes the figure reproducible from committed inputs, where
    the previous one was maintained by hand.
    """
    from scipy.interpolate import griddata

    STRUCT = os.path.join(os.environ.get("MERLIN_DIR", "/home/jan/archive/Scriven-Merlin"),
        "Scriven-Struct-125-smallDt", "bubble-ts000000.pvtu")
    POLY = os.path.join(SCRIVEN_DIR, "Scriven-Poly-125",
                        "Scriven-Poly-125_dual-ts000000.pvtu")

    print("\n" + "=" * 60)
    print("  SCRIVEN PROBLEM - meshes and initial condition")
    print("=" * 60)

    # The module sets font.size 24 for scriven_fields, which is a far larger
    # multi-panel.  Figure 6 sits beside Figure 7 in the manuscript and is set
    # in Figure 7's convention instead, locally so the other figure is
    # unaffected.
    plt.rcParams.update({"font.size": 9, "font.family": "serif",
                         "axes.labelsize": 9, "axes.titlesize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "savefig.dpi": 300})

    FRAME = 80.0                      # um, framed on R0 = 50 and its layer
    CEN = R0 * 1e6 / np.sqrt(2.0)     # interface on the 45 degree diagonal
    HALF = 18.0                       # inset half-width, holds R0 and the layer

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    theta = np.linspace(0, 2 * np.pi, 400)

    # Cell sizes are the volume-equivalent ones of tab:mesh_quality, which for
    # the polyhedral dual is 10.1% coarser than the structured mesh at the same
    # nominal level -- 1,428,219 cells against 1,906,624 in the same 300 um
    # cube.  Both reproduce from the files drawn here.  Table 3 quotes 2.42 for
    # both families instead, as the common abscissa its refinement study needs.
    for ax, (path, title, lw) in zip(axes, (
            (STRUCT, r"(a) structured hexahedral ($125^3$, $h = 2.42\,\mu$m)", 0.25),
            (POLY,   r"(b) polyhedral ($125^3$, $h = 2.67\,\mu$m)", 0.15))):
        # A missing input must stop the run rather than draw an empty frame.
        assert os.path.exists(path), f"missing initial condition: {path}"
        sliced = extract_slice(path, normal="z")
        c = sliced.cell_centers().points
        cx, cy = c[:, 0] * 1e6, c[:, 1] * 1e6
        T = np.asarray(sliced.cell_data["Temperature [K]"])
        vof = np.asarray(sliced.cell_data["Vof Sharp [1]"])

        g = np.linspace(-FRAME, FRAME, 500)
        XI, YI = np.meshgrid(g, g)
        pts = np.column_stack([cx, cy])
        T_g = griddata(pts, T, (XI, YI), method="linear")
        v_g = griddata(pts, vof, (XI, YI), method="linear")

        cf = ax.contourf(XI, YI, T_g, levels=np.linspace(T_SAT, T_INF, 30),
                         cmap="RdYlBu_r", extend="both")
        edges = [[(q[0] * 1e6, q[1] * 1e6) for q in seg]
                 for seg in get_cell_edges(sliced)]
        ax.add_collection(LineCollection(edges, colors="k", linewidths=lw,
                                         alpha=0.55))
        # Only the numerical interface is drawn.  At t = 0 the alpha = 1/2
        # contour and the initialised R0 sphere coincide, so the analytical
        # circle would sit exactly under it and hide it.
        ax.contour(XI, YI, v_g, levels=[0.5], colors="k", linewidths=1.6)

        ax.set_xlim(-FRAME, FRAME); ax.set_ylim(-FRAME, FRAME)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ [$\mu$m]")
        ax.set_title(title, fontsize=10, loc="left")

        # Figure 7's recipe: a large corner inset on the 45 degree diagonal,
        # which is where the structured mesh is least aligned with the
        # interface and so where the two discretisations differ most.
        ins = ax.inset_axes([0.02, 0.02, 0.42, 0.42])
        ins.contourf(XI, YI, T_g, levels=np.linspace(T_SAT, T_INF, 30),
                     cmap="RdYlBu_r", extend="both")
        ins.add_collection(LineCollection(
            [s for s in edges
             if all(CEN - HALF <= q[0] <= CEN + HALF
                    and CEN - HALF <= q[1] <= CEN + HALF for q in s)],
            colors="k", linewidths=0.7))
        ins.contour(XI, YI, v_g, levels=[0.5], colors="k", linewidths=1.4)
        ins.set_xlim(CEN - HALF, CEN + HALF)
        ins.set_ylim(CEN - HALF, CEN + HALF)
        ins.set_aspect("equal"); ins.set_xticks([]); ins.set_yticks([])
        ax.indicate_inset_zoom(ins, edgecolor="black", linewidth=1.2)

        # How many cells the initial layer spans, measured rather than quoted
        r = np.sqrt(cx ** 2 + cy ** 2)
        band = (T > T_SAT + 0.02 * (T_INF - T_SAT)) & \
               (T < T_SAT + 0.98 * (T_INF - T_SAT)) & (vof > 0.5)
        h = np.cbrt(np.median(np.asarray(sliced.cell_data["Grid Cell Volume [m^3]"]))) * 1e6 \
            if "Grid Cell Volume [m^3]" in sliced.cell_data else np.nan
        if band.any():
            d = r[band].max() - r[band].min()
            print(f"  {title:38s} layer {d:6.2f} um"
                  + (f", h = {h:.2f} um, {d / h:.1f} cells" if h == h else ""))

    axes[0].set_ylabel(r"$y$ [$\mu$m]")
    cb = fig.colorbar(cf, ax=axes, fraction=0.030, pad=0.02, extend="both")
    cb.set_label(r"Temperature [$^\circ$C]")

    outpath = os.path.join(OUTPUT_DIR, "scriven_meshes.png")
    for ext in ("png", "pdf"):
        plt.savefig(outpath.replace(".png", "." + ext), bbox_inches="tight",
                    dpi=200)
    print(f"\n  Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_scriven_fields()
    plot_scriven_mesh_comparison()
    print("\nDone.")
