#!/usr/bin/env python3
"""
Scriven bubble growth: temperature gradient analysis.

Shows temperature gradient magnitude on z=0 slice and
polar plot of |grad T| vs angle around the interface
to reveal anisotropy on structured vs polyhedral meshes.
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

SCRIVEN_DIR = os.environ.get("SCRIVEN_DIR",
                          "/home/jan/runs/tflows-vof/Validation/Scriven")
# The top-level 75^3 snapshots and the Sub/NN/*.vtu pieces of
# Scriven-Struct-125 were removed in the July 2026 cleanup.  The .pvtu
# containers of the latter survive, so it opens without error and yields no
# cells; these are the directories that still carry the pieces.  Both advance
# at DT_STRUCT, so ts = 100 is t = 1.0 ms in either.
STRUCT75_DIR = os.path.join(SCRIVEN_DIR, "Scriven-Struct-75-largeDt")
STRUCT_DIR  = os.path.join(SCRIVEN_DIR, "Scriven-Struct-125-smallDt")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", os.path.join(os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling"), "pics"))

BETA    = 4.06022
ALPHA_L = 0.679 / (958.4 * 4216.0)
R0      = 5.0e-5
T0      = R0**2 / (4.0 * BETA**2 * ALPHA_L)
DT_STRUCT = 1.0e-5
DT_POLY   = 2.0e-6


def analytical_radius(t):
    return 2.0 * BETA * np.sqrt(ALPHA_L * t)


def get_cell_edges(sliced):
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


def analyze_gradients(pvtu_path, t_anal, label=""):
    """Extract gradient info near the interface on z=0 slice."""
    from scipy.interpolate import griddata

    R_anal = analytical_radius(t_anal)

    mesh = pv.read(pvtu_path)
    sliced = mesh.slice(normal='z', origin=(0, 0, 0))
    centers = sliced.cell_centers()
    cx = centers.points[:, 0]
    cy = centers.points[:, 1]

    T = sliced.cell_data['Temperature [K]']
    gradT = sliced.cell_data['Temperature Gradients [K/m]']
    vof = sliced.cell_data['Vof Sharp [1]']
    grad_mag = np.sqrt(gradT[:, 0]**2 + gradT[:, 1]**2 + gradT[:, 2]**2)

    # Distance from center
    r = np.sqrt(cx**2 + cy**2)
    # Angle from center
    theta = np.arctan2(cy, cx)

    # Select cells near the interface (within 1.5 cell widths of R_anal)
    # Cell width ~ 4 um = 4e-6 m
    dx = 4.0e-6
    near_interface = (np.abs(r - R_anal) < 2.0 * dx) & (grad_mag > 0)

    return {
        'cx': cx, 'cy': cy,
        'T': T, 'gradT': gradT, 'grad_mag': grad_mag,
        'vof': vof, 'r': r, 'theta': theta,
        'near_interface': near_interface,
        'R_anal': R_anal,
        'sliced': sliced,
    }


def plot_gradient_fields():
    """2x3 figure: grad_T magnitude for structured and polyhedral."""
    from scipy.interpolate import griddata

    POLY_DIR = os.path.join(SCRIVEN_DIR, "Scriven-Poly-125")

    # Structured: dt=1e-5 -> ts = 50, 80, 110
    # Poly-125:   dt=2e-6 -> ts = 250, 400, 550 (same physical times)
    ts_struct = [50, 80, 110]
    ts_poly   = [250, 400, 550]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    cf = None
    for idx in range(3):
        t_sim = ts_struct[idx] * DT_STRUCT
        t_anal = t_sim + T0
        R_anal = analytical_radius(t_anal)
        R_um = R_anal * 1e6
        t_ms = t_anal * 1e3

        # --- Structured ---
        pvtu = os.path.join(STRUCT_DIR,
                            f"bubble-ts{ts_struct[idx]:06d}.pvtu")
        if os.path.exists(pvtu):
            data = analyze_gradients(pvtu, t_anal)
            cx_um = data['cx'] * 1e6
            cy_um = data['cy'] * 1e6
            points = np.column_stack([cx_um, cy_um])

            ngrid = 400
            yi = np.linspace(-148, 148, ngrid)
            zi = np.linspace(-148, 148, ngrid)
            YI, ZI = np.meshgrid(yi, zi)

            gmag_grid = griddata(points, data['grad_mag'],
                                 (YI, ZI), method='linear')
            vof_grid = griddata(points, data['vof'],
                                (YI, ZI), method='linear')

            # Plot gradient magnitude
            levels = np.linspace(
                0,
                np.percentile(data['grad_mag'][data['grad_mag'] > 0], 99),
                30)
            if levels[-1] > 0:
                cf = axes[0, idx].contourf(
                    YI, ZI, gmag_grid, levels=levels,
                    cmap='hot_r', extend='max')
            # VOF contour
            axes[0, idx].contour(YI, ZI, vof_grid, levels=[0.5],
                                 colors='cyan', linewidths=2)
            # Analytical circle
            theta = np.linspace(0, 2*np.pi, 200)
            axes[0, idx].plot(R_um*np.cos(theta), R_um*np.sin(theta),
                              'w--', linewidth=1.5)

            # Mesh edges near interface
            cell_edges = get_cell_edges(data['sliced'])
            cell_edges_um = [[(p[0]*1e6, p[1]*1e6) for p in seg]
                             for seg in cell_edges]
            R_show = R_um * 1.8
            filtered = [seg for seg in cell_edges_um
                        if np.sqrt((0.5*(seg[0][0]+seg[1][0]))**2
                                   + (0.5*(seg[0][1]+seg[1][1]))**2)
                        < R_show]
            lc = LineCollection(filtered, colors='gray',
                                linewidths=0.3, alpha=0.3)
            axes[0, idx].add_collection(lc)

            axes[0, idx].set_xlim([-150, 150])
            axes[0, idx].set_ylim([-150, 150])
            axes[0, idx].set_aspect('equal')
            axes[0, idx].set_title(
                f'Structured, $t = {t_ms:.2f}$ ms')
            axes[0, idx].set_xlabel(r'$x$ [$\mu$m]')
            if idx == 0:
                axes[0, idx].set_ylabel(r'$y$ [$\mu$m]')

            print(f"  Struct ts={ts_struct[idx]}: "
                  f"grad_T max={data['grad_mag'].max():.2e}")

        # --- Polyhedral ---
        pvtu_p = os.path.join(
            POLY_DIR,
            f"Scriven-Poly-125_dual-ts{ts_poly[idx]:06d}.pvtu")
        if os.path.exists(pvtu_p):
            data = analyze_gradients(pvtu_p, t_anal)
            cx_um = data['cx'] * 1e6
            cy_um = data['cy'] * 1e6
            points = np.column_stack([cx_um, cy_um])

            gmag_grid = griddata(points, data['grad_mag'],
                                 (YI, ZI), method='linear')
            vof_grid = griddata(points, data['vof'],
                                (YI, ZI), method='linear')

            levels = np.linspace(
                0,
                np.percentile(data['grad_mag'][data['grad_mag'] > 0], 99),
                30)
            if levels[-1] > 0:
                cf = axes[1, idx].contourf(
                    YI, ZI, gmag_grid, levels=levels,
                    cmap='hot_r', extend='max')
            axes[1, idx].contour(YI, ZI, vof_grid, levels=[0.5],
                                 colors='cyan', linewidths=2)
            axes[1, idx].plot(R_um*np.cos(theta), R_um*np.sin(theta),
                              'w--', linewidth=1.5)

            cell_edges = get_cell_edges(data['sliced'])
            cell_edges_um = [[(p[0]*1e6, p[1]*1e6) for p in seg]
                             for seg in cell_edges]
            filtered = [seg for seg in cell_edges_um
                        if np.sqrt((0.5*(seg[0][0]+seg[1][0]))**2
                                   + (0.5*(seg[0][1]+seg[1][1]))**2)
                        < R_show]
            lc = LineCollection(filtered, colors='gray',
                                linewidths=0.2, alpha=0.3)
            axes[1, idx].add_collection(lc)

            axes[1, idx].set_xlim([-150, 150])
            axes[1, idx].set_ylim([-150, 150])
            axes[1, idx].set_aspect('equal')
            axes[1, idx].set_title(
                f'Polyhedral, $t = {t_ms:.2f}$ ms')
            axes[1, idx].set_xlabel(r'$x$ [$\mu$m]')
            if idx == 0:
                axes[1, idx].set_ylabel(r'$y$ [$\mu$m]')

            print(f"  Poly   ts={ts_poly[idx]}: "
                  f"grad_T max={data['grad_mag'].max():.2e}")

    if cf is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(cf, cax=cbar_ax, label=r'$|\nabla T|$ [K/m]')

    plt.subplots_adjust(left=0.06, right=0.90, wspace=0.2, hspace=0.25)
    outpath = os.path.join(OUTPUT_DIR, "scriven_grad_T.png")
    plt.savefig(outpath, bbox_inches='tight')
    print(f"\n  Saved: {outpath}")
    plt.close()


def compute_polar_gradient(pvtu_path, t_anal, dx_cell, label=""):
    """Extract gradient polar data near interface on z=0 slice."""
    R_anal = analytical_radius(t_anal)
    mesh = pv.read(pvtu_path)
    sliced = mesh.slice(normal='z', origin=(0, 0, 0))
    centers = sliced.cell_centers()
    cx, cy = centers.points[:, 0], centers.points[:, 1]

    if sliced.n_cells == 0:
        raise RuntimeError(
            f"{pvtu_path} yielded no cells: the .pvtu container is there but "
            f"its Sub/NN/*.vtu pieces are not")
    gradT = sliced.cell_data['Temperature Gradients [K/m]']
    grad_mag = np.sqrt(gradT[:, 0]**2 + gradT[:, 1]**2 + gradT[:, 2]**2)

    r = np.sqrt(cx**2 + cy**2)
    theta = np.arctan2(cy, cx)

    near_interface = (np.abs(r - R_anal) < 2.0 * dx_cell) & (grad_mag > 0)

    theta_if = theta[near_interface]
    gmag_if = grad_mag[near_interface]

    # Bin by angle (5-degree bins)
    n_bins = 72
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_means = np.zeros(n_bins)
    for i in range(n_bins):
        in_bin = (theta_if >= bin_edges[i]) & (theta_if < bin_edges[i+1])
        if in_bin.sum() > 0:
            bin_means[i] = np.mean(gmag_if[in_bin])

    valid = bin_means > 0
    ratio = (bin_means[valid].min() / bin_means[valid].max()
             if valid.sum() > 0 else 0)

    # Fourier decomposition of the binned signal
    mean_val = np.mean(bin_means[valid]) if valid.sum() > 0 else 1.0
    signal = bin_means.copy()
    signal[~valid] = mean_val
    signal_norm = signal / mean_val

    fft = np.fft.rfft(signal_norm)
    amplitudes = np.abs(fft) / (n_bins / 2)
    amplitudes[0] = np.abs(fft[0]) / n_bins

    print(f"\n  {label}:")
    print(f"    Cells near interface: {near_interface.sum()}")
    print(f"    Mean |grad T|: {mean_val:.4e} K/m")
    print(f"    Min/max ratio: {ratio:.3f}")
    for m in range(min(9, len(amplitudes))):
        print(f"      m={m:2d}: {amplitudes[m]*100:.2f}%")

    return {
        'theta_if': theta_if, 'gmag_if': gmag_if,
        'bin_centers': bin_centers, 'bin_means': bin_means,
        'valid': valid, 'ratio': ratio, 'mean': mean_val,
        'fft_amplitudes': amplitudes, 'label': label,
    }


def radial_label_angle(bin_centers, bin_means, ticks, keep_off_deg=15.0,
                       window=1):
    """Angle (degrees) at which the radial tick labels sit farthest from the
    binned-mean curve: over the bin centres, maximise the distance between the
    curve and the nearest labelled ring, the curve being taken over the
    `window` neighbouring bins on either side (a label has angular extent),
    and staying `keep_off_deg` away from the angular labels at multiples of
    45 degrees."""
    ticks = np.asarray(ticks, dtype=float)
    vals = np.asarray(bin_means, dtype=float)
    n = len(vals)
    best_ang, best_gap = 22.5, -1.0
    for i, th in enumerate(bin_centers):
        deg = np.degrees(th) % 360.0
        off = min(abs(((deg - k) + 180.0) % 360.0 - 180.0)
                  for k in range(0, 360, 45))
        if off < keep_off_deg:
            continue
        neigh = vals[[(i + j) % n for j in range(-window, window + 1)]]
        neigh = neigh[neigh > 0]
        if len(neigh) == 0:
            continue
        gap = min(np.min(np.abs(ticks - v)) for v in neigh)
        if gap > best_gap:
            best_gap, best_ang = gap, deg
    return best_ang


def polar_gradient_results(ts_s=100):
    """Load the three meshes at structured step `ts_s` (dt = 1e-5 s, so
    ts_s = 100 is t = 1.0 ms of simulation time, the instant of the second
    column of Table 4; the analytical time t + t_0 is 1.23 ms). The
    polyhedral run uses dt = 2e-6 s, hence five times the step count."""
    POLY_DIR = os.path.join(SCRIVEN_DIR, "Scriven-Poly-125")
    t_sim = ts_s * DT_STRUCT
    t_anal = t_sim + T0
    ts_p = int(round(t_sim / DT_POLY))
    print(f"  polar figure at t = {t_sim*1e3:.2f} ms simulation time "
          f"(t + t_0 = {t_anal*1e3:.2f} ms); structured ts = {ts_s}, "
          f"polyhedral ts = {ts_p}")

    meshes = [
        (os.path.join(STRUCT75_DIR, f"bubble-ts{ts_s:06d}.pvtu"),
         t_anal, 4.0e-6, "Structured 75³"),
        (os.path.join(STRUCT_DIR, f"bubble-ts{ts_s:06d}.pvtu"),
         t_anal, 2.4e-6, "Structured 125³"),
        (os.path.join(POLY_DIR,
                      f"Scriven-Poly-125_dual-ts{ts_p:06d}.pvtu"),
         t_anal, 2.4e-6, "Polyhedral 125³"),
    ]
    results = {}
    for pvtu, t_a, dx, key in meshes:
        if os.path.exists(pvtu):
            results[key] = compute_polar_gradient(pvtu, t_a, dx, key)
        else:
            print(f"\n  {key}: FILE NOT FOUND ({pvtu})")
    return results


def plot_polar_gradients(results=None, outpath=None):
    """2x3 figure: polar plots of |grad T| vs angle (top row) and
    Fourier azimuthal mode decomposition (bottom row) for three meshes.

    Columns: Structured 75³, Structured 125³, Polyhedral 125³.
    `results` (from polar_gradient_results) may be passed in, so that the
    drawing can be checked without the field data.
    """
    if results is None:
        results = polar_gradient_results()
    if outpath is None:
        outpath = os.path.join(OUTPUT_DIR, "scriven_grad_polar.png")

    panels = [
        (r"Structured $75^3$", "Structured 75³"),
        (r"Structured $125^3$", "Structured 125³"),
        (r"Polyhedral $125^3$", "Polyhedral 125³"),
    ]

    fig = plt.figure(figsize=(16, 8.8))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.07,
                        hspace=0.5, wspace=0.28)

    # Top row: polar plots
    for i, (tex_label, key) in enumerate(panels):
        ax = fig.add_subplot(2, 3, i + 1, projection='polar')
        if key in results:
            d = results[key]
            ax.scatter(d['theta_if'], d['gmag_if'], s=3, alpha=0.3,
                       color='gray', label='cells')
            ax.plot(d['bin_centers'], d['bin_means'], 'r-', linewidth=2,
                    label='binned mean')
            ax.set_title(
                f'{tex_label}\n'
                f'mean $|\\nabla T|$ = {d["mean"]:.2e} K/m\n'
                f'min/max ratio = {d["ratio"]:.3f}',
                pad=20, fontsize=10)

            # Radial scale: rings at multiples of 2e4 K/m, labelled in units
            # of 1e4 K/m, placed where they sit farthest from the red curve
            # and clear of the angular labels.
            rmax = 1.06 * float(np.max(d['gmag_if']))
            ax.set_rlim(0.0, rmax)
            # the smallest round step that labels at most four rings
            # (labelled rings stay inside 88% of the radius, clear of the
            # angular labels outside the circle)
            for step in (2.0e4, 2.5e4, 4.0e4, 5.0e4, 1.0e5, 2.0e5):
                ticks = np.arange(step, 0.88 * rmax, step)
                if len(ticks) <= 4:
                    break
            ax.set_rticks(ticks)
            labels = [f"{t/1e4:g}" for t in ticks]
            ang = radial_label_angle(d['bin_centers'], d['bin_means'], ticks)
            ax.set_rlabel_position(ang)
            print(f"  {key}: radial labels {labels} at {ang:.1f} deg")
            # Drawn by hand on the rings.  The radial tick pad displaces a
            # label off the ring it names, outward or inward with the quadrant
            # the label azimuth falls in, so the three panels disagreed by
            # -5.5 to +4.1% of the radius and the polyhedral "10" read as 9.3.
            ax.set_yticklabels([])
            for t, txt in zip(ticks, labels):
                ax.text(np.radians(ang), t, txt, fontsize=8, ha='center',
                        va='center', zorder=5,
                        bbox=dict(facecolor='white', edgecolor='none',
                                  alpha=0.85, pad=0.15))
            # legend below the axes, off the 315 degree label, with the unit
            # of the radial rings under it
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06),
                      ncol=2, fontsize=8, frameon=False)
            ax.text(0.5, -0.19, r'rings: $|\nabla T|$ in $10^{4}$ K/m',
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=8, color='0.25')

    # Bottom row: Fourier spectra (bar charts)
    for i, (tex_label, key) in enumerate(panels):
        ax = fig.add_subplot(2, 3, i + 4)
        if key in results:
            d = results[key]
            amps = d['fft_amplitudes']
            max_mode = min(9, len(amps))
            colors = ['C3' if m == 4 else 'C0'
                      for m in range(1, max_mode)]
            ax.bar(range(1, max_mode), amps[1:max_mode] * 100,
                   color=colors, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Azimuthal mode $m$')
            ax.set_ylabel('Relative amplitude [%]')
            ax.set_title(f'{tex_label}', fontsize=10)
            ax.set_xticks(range(1, max_mode))
            ax.set_ylim([0,
                         max(amps[1:max_mode] * 100) * 1.3 + 0.5])
            ax.axhline(y=0, color='k', linewidth=0.5)
            if 4 < max_mode:
                ax.annotate(
                    f'm=4: {amps[4]*100:.1f}%',
                    xy=(4, amps[4] * 100),
                    xytext=(5.5, amps[4] * 100 * 1.1),
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='->', color='C3'))

    plt.savefig(outpath, bbox_inches='tight')
    plt.savefig(outpath.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n  Saved: {outpath}")
    plt.close()


def plot_gradient_components():
    """Show grad_T_y and grad_T_z separately on the z=0 slice
    for the structured mesh at ts=80, to see direction-dependent
    gradients. Also shows gradient vectors near the interface.
    """
    from scipy.interpolate import griddata

    ts = 80
    t_sim = ts * DT_STRUCT
    t_anal = t_sim + T0
    R_anal = analytical_radius(t_anal)
    R_um = R_anal * 1e6

    pvtu = os.path.join(STRUCT_DIR, f"bubble-ts{ts:06d}.pvtu")
    if not os.path.exists(pvtu):
        print(f"  WARNING: {pvtu} not found")
        return

    data = analyze_gradients(pvtu, t_anal)
    cx_um = data['cx'] * 1e6
    cy_um = data['cy'] * 1e6
    gradT = data['gradT']
    points = np.column_stack([cx_um, cy_um])

    ngrid = 400
    yi = np.linspace(-148, 148, ngrid)
    zi = np.linspace(-148, 148, ngrid)
    YI, ZI = np.meshgrid(yi, zi)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    titles = [r'$\partial T / \partial y$',
              r'$\partial T / \partial z$',
              r'$|\nabla T|$ + vectors']
    fields = [gradT[:, 0], gradT[:, 1], data['grad_mag']]
    cmaps = ['RdBu_r', 'RdBu_r', 'hot_r']

    for i, (ax, title, field, cmap) in enumerate(
            zip(axes, titles, fields, cmaps)):
        f_grid = griddata(points, field, (YI, ZI), method='linear')
        vof_grid = griddata(points, data['vof'],
                            (YI, ZI), method='linear')

        if i < 2:  # Component: symmetric colormap
            vmax = np.percentile(np.abs(field[field != 0]), 99)
            levels = np.linspace(-vmax, vmax, 40)
        else:  # Magnitude: sequential
            vmax = np.percentile(field[field > 0], 99)
            levels = np.linspace(0, vmax, 40)

        cf = ax.contourf(YI, ZI, f_grid, levels=levels, cmap=cmap,
                         extend='both')
        ax.contour(YI, ZI, vof_grid, levels=[0.5], colors='cyan',
                   linewidths=2)

        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(R_um*np.cos(theta), R_um*np.sin(theta),
                'w--', linewidth=1)

        # Gradient vectors for the last panel
        if i == 2:
            from scipy.spatial import cKDTree
            n_arr = 22
            arr_y = np.linspace(-R_um*1.5, R_um*1.5, n_arr)
            arr_z = np.linspace(-R_um*1.5, R_um*1.5, n_arr)
            ay, az = np.meshgrid(arr_y, arr_z)
            tree = cKDTree(points)
            _, nearest = tree.query(
                np.column_stack([ay.ravel(), az.ravel()]))
            gmag_at = data['grad_mag'][nearest]
            sig = gmag_at > 0.1 * data['grad_mag'].max()
            max_g = data['grad_mag'].max()
            ax.quiver(cx_um[nearest[sig]], cy_um[nearest[sig]],
                      gradT[nearest[sig], 0], gradT[nearest[sig], 1],
                      color='white', alpha=0.8, scale=max_g*25,
                      width=0.004, headwidth=3, headlength=4)

        ax.set_xlim([-150, 150])
        ax.set_ylim([-150, 150])
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel(r'$x$ [$\mu$m]')
        if i == 0:
            ax.set_ylabel(r'$y$ [$\mu$m]')
        fig.colorbar(cf, ax=ax, shrink=0.8, label='[K/m]')

    plt.suptitle(f'Structured mesh, $t = {t_anal*1e3:.2f}$ ms',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "scriven_grad_components.png")
    plt.savefig(outpath, bbox_inches='tight')
    print(f"\n  Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    import sys
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  SCRIVEN - Temperature Gradient Analysis")
    print("=" * 60)

    if "polar" in sys.argv[1:]:
        # only Figure 14 of the paper (pics/scriven_grad_polar.{png,pdf})
        print("\n--- Polar gradient analysis ---")
        plot_polar_gradients()
        sys.exit(0)

    print("\n--- 1. Gradient magnitude fields (2x3) ---")
    plot_gradient_fields()

    print("\n--- 2. Polar gradient analysis ---")
    plot_polar_gradients()

    print("\n--- 3. Gradient components (structured, ts=80) ---")
    plot_gradient_components()

    print("\nDone.")
