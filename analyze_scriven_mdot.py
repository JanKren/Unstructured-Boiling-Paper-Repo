#!/usr/bin/env python3
"""
Scriven bubble growth: mass transfer (m_dot) contours, velocity vectors, and mesh.
Adapted from analyze_scriven_fields.py — uses Vof MassTransfer field instead of T.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pyvista as pv
import os

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'savefig.dpi': 300,
    'axes.grid': False,
    'lines.linewidth': 1.5,
})

# ===========================================================================
#  Paths and physical parameters
# ===========================================================================
SCRIVEN_DIR = os.environ.get("SCRIVEN_DIR",
                          "/home/jan/runs/tflows-vof/Validation/Scriven")
STRUCT_DIR  = os.path.join(SCRIVEN_DIR, "Scriven-Struct-125")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", os.path.join(os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling"), "pics"))

# Scriven analytical parameters
BETA    = 4.06022
ALPHA_L = 0.679 / (958.4 * 4216.0)  # thermal diffusivity [m^2/s]
R0      = 5.0e-5                      # initial bubble radius [m]
T0      = R0**2 / (4.0 * BETA**2 * ALPHA_L)  # virtual origin time [s]
DT_STRUCT = 1.0e-5                     # structured mesh time step [s]
DT_POLY   = 2.0e-6                     # polyhedral mesh time step [s]


def analytical_radius(t):
    """Scriven bubble radius R(t) = 2*beta*sqrt(alpha_l * t)."""
    return 2.0 * BETA * np.sqrt(ALPHA_L * t)


def extract_slice(pvtu_path, normal='z'):
    """Read PVTU and slice through center perpendicular to 'normal' axis."""
    mesh = pv.read(pvtu_path)
    sliced = mesh.slice(normal=normal, origin=(0, 0, 0))
    return sliced


def get_cell_edges(sliced):
    """Extract all internal cell edges from the slice for mesh visualization."""
    wireframe = sliced.extract_all_edges()
    pts = wireframe.points
    lines = []
    i = 0
    conn = wireframe.lines
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


def plot_single_panel(ax, pvtu_path, t_anal, show_ylabel=False,
                      title='', vmin=None, vmax=None):
    """Plot m_dot contours + velocity vectors + mesh on one axis.

    Returns (contourf artist, min_mdot, max_mdot).
    """
    from scipy.spatial import cKDTree
    from scipy.interpolate import griddata

    R_anal = analytical_radius(t_anal)

    sliced = extract_slice(pvtu_path, normal='z')
    centers = sliced.cell_centers()
    cx = centers.points[:, 0]
    cy = centers.points[:, 1]

    # Get mass transfer field
    mdot = sliced.cell_data['Vof MassTransfer [kg/m^3/s]']
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

    mdot_grid = griddata(points, mdot, (YI, ZI), method='linear')
    vof_grid = griddata(points, vof, (YI, ZI), method='linear')

    # m_dot contours — sequential colormap (mdot is non-negative for evaporation)
    if vmin is None or vmax is None:
        nonzero = np.abs(mdot) > 0
        if nonzero.any():
            vmax = np.percentile(mdot[nonzero], 95)
        else:
            vmax = 1.0
        vmin = 0.0

    levels = np.linspace(vmin, vmax, 40)
    cf = ax.contourf(YI, ZI, mdot_grid, levels=levels, cmap='hot_r',
                     extend='max')

    # Analytical circle (thin reference, no VOF contour to avoid obscuring mdot)
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(R_um * np.cos(theta), R_um * np.sin(theta),
            'w--', linewidth=1.0, alpha=0.6)

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

    # Mesh edges near the interface
    cell_edges = get_cell_edges(sliced)
    cell_edges_um = [[(p[0]*1e6, p[1]*1e6) for p in seg]
                     for seg in cell_edges]
    R_show = R_um * 1.8
    filtered_edges = [seg for seg in cell_edges_um
                      if np.sqrt((0.5*(seg[0][0]+seg[1][0]))**2 +
                                 (0.5*(seg[0][1]+seg[1][1]))**2) < R_show]
    lc = LineCollection(filtered_edges, colors='gray',
                        linewidths=0.3, alpha=0.4)
    ax.add_collection(lc)

    ax.set_xlim([-150, 150])
    ax.set_ylim([-150, 150])
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$ [$\mu$m]')
    if show_ylabel:
        ax.set_ylabel(r'$y$ [$\mu$m]')
    ax.set_title(title)

    return cf, mdot.min(), mdot.max()


def plot_scriven_mdot():
    """Create the 2x3 Scriven m_dot visualization: structured vs polyhedral."""

    print("=" * 60)
    print("  SCRIVEN PROBLEM - Mass Transfer (m_dot) Visualization")
    print("=" * 60)

    POLY_DIR = os.path.join(SCRIVEN_DIR, "Scriven-Poly-125")

    # Same physical times as temperature figure (0.5, 0.8, 1.1 ms)
    # Structured: dt=1e-5 -> ts = 50, 80, 110
    # Poly-125:   dt=2e-6 -> ts = 250, 400, 550
    ts_struct = [50, 80, 110]
    ts_poly   = [250, 400, 550]
    col_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

    # First pass: determine global m_dot range for consistent coloring
    print("\n  First pass: determining global m_dot range...")
    all_mdot_nonzero = []
    for idx in range(3):
        pvtu_struct = os.path.join(
            STRUCT_DIR, f"bubble-ts{ts_struct[idx]:06d}.pvtu")
        if os.path.exists(pvtu_struct):
            mesh = pv.read(pvtu_struct)
            sliced = mesh.slice(normal='z', origin=(0, 0, 0))
            mdot = sliced.cell_data['Vof MassTransfer [kg/m^3/s]']
            nonzero = mdot > 0
            if nonzero.any():
                all_mdot_nonzero.append(mdot[nonzero])
            print(f"    Structured ts={ts_struct[idx]}: "
                  f"m_dot range [{mdot.min():.2e}, {mdot.max():.2e}]")

        pvtu_poly = os.path.join(
            POLY_DIR, f"Scriven-Poly-125_dual-ts{ts_poly[idx]:06d}.pvtu")
        if os.path.exists(pvtu_poly):
            mesh = pv.read(pvtu_poly)
            sliced = mesh.slice(normal='z', origin=(0, 0, 0))
            mdot = sliced.cell_data['Vof MassTransfer [kg/m^3/s]']
            nonzero = mdot > 0
            if nonzero.any():
                all_mdot_nonzero.append(mdot[nonzero])
            print(f"    Polyhedral ts={ts_poly[idx]}: "
                  f"m_dot range [{mdot.min():.2e}, {mdot.max():.2e}]")

    # Use 95th percentile of positive values for tighter range
    combined = np.concatenate(all_mdot_nonzero)
    vmin = 0.0
    vmax = np.percentile(combined, 95)
    print(f"  Global range: [{vmin:.2e}, {vmax:.2e}]")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    cf = None
    for idx in range(3):
        t_sim = ts_struct[idx] * DT_STRUCT
        t_anal = t_sim + T0
        R_anal = analytical_radius(t_anal)
        t_ms = t_anal * 1e3

        # --- Top row: structured mesh ---
        pvtu_struct = os.path.join(
            STRUCT_DIR, f"bubble-ts{ts_struct[idx]:06d}.pvtu")
        if os.path.exists(pvtu_struct):
            label_s = col_labels[idx]
            title_s = f'{label_s} Structured, $t = {t_ms:.2f}$ ms'
            print(f"\n  Structured ts={ts_struct[idx]}: t_anal={t_ms:.3f} ms, "
                  f"R={R_anal*1e6:.1f} um")
            cf, mn, mx = plot_single_panel(
                axes[0, idx], pvtu_struct, t_anal,
                show_ylabel=(idx == 0), title=title_s,
                vmin=vmin, vmax=vmax)
        else:
            print(f"  WARNING: {pvtu_struct} not found")

        # --- Bottom row: polyhedral mesh ---
        pvtu_poly = os.path.join(
            POLY_DIR, f"Scriven-Poly-125_dual-ts{ts_poly[idx]:06d}.pvtu")
        if os.path.exists(pvtu_poly):
            label_p = col_labels[idx + 3]
            title_p = f'{label_p} Polyhedral, $t = {t_ms:.2f}$ ms'
            print(f"  Polyhedral ts={ts_poly[idx]}: t_anal={t_ms:.3f} ms, "
                  f"R={R_anal*1e6:.1f} um")
            cf, mn, mx = plot_single_panel(
                axes[1, idx], pvtu_poly, t_anal,
                show_ylabel=(idx == 0), title=title_p,
                vmin=vmin, vmax=vmax)
        else:
            print(f"  WARNING: {pvtu_poly} not found")

    # Shared colorbar (wider for readability)
    if cf is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(cf, cax=cbar_ax,
                     label=r'Mass transfer $\dot{m}$ [kg/m$^3$/s]')

    plt.subplots_adjust(left=0.06, right=0.90, wspace=0.2, hspace=0.25)
    outpath = os.path.join(OUTPUT_DIR, "scriven_mdot.png")
    plt.savefig(outpath, bbox_inches='tight')
    plt.savefig(outpath.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n  Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_scriven_mdot()
    print("\nDone.")
