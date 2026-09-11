#!/usr/bin/env python3
"""
Compare Scriven bubble growth: old (unweighted) vs new (IDW2-weighted)
gradient stencil for mass transfer computation.

Compares:
  1. Bubble radius vs time (bench-data.dat)
  2. Temperature gradient polar anisotropy at ts=120
  3. Gradient magnitude fields side-by-side
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pyvista as pv
import os
import sys

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

SCRIVEN_DIR = os.environ.get("SCRIVEN_DIR",
                          "/home/jan/runs/tflows-vof/Validation/Scriven")
OLD_DIR     = os.path.join(SCRIVEN_DIR, "old_results")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", os.path.join(os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling"), "pics"))

# Scriven problem parameters
BETA    = 4.06022
ALPHA_L = 0.679 / (958.4 * 4216.0)
R0      = 5.0e-5
T0      = R0**2 / (4.0 * BETA**2 * ALPHA_L)
DT      = 1.0e-5


def analytical_radius(t):
    return 2.0 * BETA * np.sqrt(ALPHA_L * t)


def load_bench_data(filepath):
    """Load bench-data.dat: time, surface, volume, radius."""
    data = np.loadtxt(filepath)
    return {
        'time': data[:, 0],
        'surface': data[:, 1],
        'volume': data[:, 2],
        'radius': data[:, 3],
    }


def analyze_slice_gradients(pvtu_path, t_anal):
    """Extract gradient info from z=0 slice."""
    R_anal = analytical_radius(t_anal)

    mesh = pv.read(pvtu_path)
    sliced = mesh.slice(normal='z', origin=(0, 0, 0))
    centers = sliced.cell_centers()
    cx = centers.points[:, 0]
    cy = centers.points[:, 1]

    gradT = sliced.cell_data['Temperature Gradients [K/m]']
    vof = sliced.cell_data['Vof Sharp [1]']
    grad_mag = np.sqrt(gradT[:, 0]**2 + gradT[:, 1]**2
                       + gradT[:, 2]**2)

    r = np.sqrt(cx**2 + cy**2)
    theta = np.arctan2(cy, cx)

    dx = 4.0e-6
    near_interface = (np.abs(r - R_anal) < 2.0 * dx) & (grad_mag > 0)

    return {
        'cx': cx, 'cy': cy,
        'gradT': gradT, 'grad_mag': grad_mag,
        'vof': vof, 'r': r, 'theta': theta,
        'near_interface': near_interface,
        'R_anal': R_anal,
    }


def plot_radius_comparison():
    """Compare bubble radius vs time for old and new methods."""
    old_file = os.path.join(OLD_DIR, "bench-data-old.dat")
    new_file = os.path.join(SCRIVEN_DIR, "bench-data.dat")

    if not os.path.exists(old_file):
        print("  WARNING: old bench-data not found")
        return
    if not os.path.exists(new_file):
        print("  WARNING: new bench-data not found (run not finished?)")
        return

    old = load_bench_data(old_file)
    n_old = len(old['time'])

    # New run appends to old data; extract only the new portion
    all_data = load_bench_data(new_file)
    new = {k: v[n_old:] for k, v in all_data.items()}

    if len(new['time']) == 0:
        print("  WARNING: no new data yet (simulation still running)")
        return

    print(f"  Old data: {n_old} steps, "
          f"t = {old['time'][0]*1e3:.3f} - {old['time'][-1]*1e3:.3f} ms")
    print(f"  New data: {len(new['time'])} steps, "
          f"t = {new['time'][0]*1e3:.3f} - {new['time'][-1]*1e3:.3f} ms")

    # Analytical solution
    t_anal = np.linspace(T0, old['time'].max() + T0, 500)
    R_anal = analytical_radius(t_anal)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Radius vs time
    ax = axes[0]
    ax.plot((t_anal - T0) * 1e3, R_anal * 1e6, 'k-',
            label='Scriven analytical', linewidth=2)
    ax.plot(old['time'] * 1e3, old['radius'] * 1e6, 'r--',
            label='Unweighted (old)', linewidth=1.5)
    ax.plot(new['time'] * 1e3, new['radius'] * 1e6, 'b-',
            label='IDW2-weighted (new)', linewidth=1.5)
    ax.set_xlabel('Simulation time [ms]')
    ax.set_ylabel(r'Bubble radius [$\mu$m]')
    ax.set_title('Bubble radius vs time')
    ax.legend()

    # Panel 2: Relative error
    ax = axes[1]
    # Interpolate analytical to simulation times
    for data, label, color, ls in [
        (old, 'Unweighted (old)', 'red', '--'),
        (new, 'IDW2-weighted (new)', 'blue', '-'),
    ]:
        t_sim = data['time']
        t_phys = t_sim + T0
        R_exact = analytical_radius(t_phys)
        err = (data['radius'] - R_exact) / R_exact * 100
        ax.plot(t_sim * 1e3, err, color=color, ls=ls, label=label)

    ax.set_xlabel('Simulation time [ms]')
    ax.set_ylabel('Radius error [%]')
    ax.set_title('Relative error in bubble radius')
    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.legend()

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "compare_radius_idw2.png")
    plt.savefig(outpath, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()

    # Print summary
    for data, label in [(old, 'Old'), (new, 'New')]:
        t_sim = data['time']
        t_phys = t_sim + T0
        R_exact = analytical_radius(t_phys)
        err = (data['radius'] - R_exact) / R_exact * 100
        print(f"  {label}: max err = {np.max(np.abs(err)):.2f}%, "
              f"mean err = {np.mean(err):.2f}%, "
              f"final R = {data['radius'][-1]*1e6:.2f} um")


def plot_polar_comparison():
    """Compare polar gradient anisotropy: old vs new at ts=120."""
    ts = 120
    t_sim = ts * DT
    t_anal = t_sim + T0

    old_pvtu = os.path.join(OLD_DIR, f"bubble-ts{ts:06d}.pvtu")
    new_pvtu = os.path.join(SCRIVEN_DIR, f"bubble-ts{ts:06d}.pvtu")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             subplot_kw={'projection': 'polar'})

    results = {}
    for ax, (pvtu, label, color) in zip(axes, [
        (old_pvtu, 'Unweighted (old)', 'red'),
        (new_pvtu, 'IDW2-weighted (new)', 'blue'),
    ]):
        if not os.path.exists(pvtu):
            ax.set_title(f'{label}\n(data not available)')
            print(f"  WARNING: {pvtu} not found")
            continue

        data = analyze_slice_gradients(pvtu, t_anal)
        mask = data['near_interface']

        if mask.sum() == 0:
            ax.set_title(f'{label}\n(no cells near interface)')
            continue

        theta_if = data['theta'][mask]
        gmag_if = data['grad_mag'][mask]

        # Bin by angle
        n_bins = 72
        bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_means = np.zeros(n_bins)
        for i in range(n_bins):
            in_bin = ((theta_if >= bin_edges[i])
                      & (theta_if < bin_edges[i+1]))
            if in_bin.sum() > 0:
                bin_means[i] = np.mean(gmag_if[in_bin])

        ax.scatter(theta_if, gmag_if, s=3, alpha=0.3, color='gray')
        ax.plot(bin_centers, bin_means, color=color, linewidth=2,
                label='binned mean')

        mean_g = np.mean(gmag_if)
        valid = bin_means[bin_means > 0]
        ratio = valid.min() / valid.max() if len(valid) > 0 else 0

        ax.set_title(
            f'{label}\n'
            f'mean $|\\nabla T|$ = {mean_g:.2e} K/m\n'
            f'min/max ratio = {ratio:.3f}',
            pad=20)
        ax.legend(loc='lower right', fontsize=8)

        results[label] = {
            'mean': mean_g,
            'min_max_ratio': ratio,
            'cells': mask.sum(),
        }

        # Print angular stats
        print(f"\n  {label} (ts={ts}):")
        print(f"    Cells near interface: {mask.sum()}")
        print(f"    Mean |grad T|: {mean_g:.4e}")
        print(f"    min/max ratio: {ratio:.4f}")
        for angle_deg in [0, 45, 90, 135]:
            angle_rad = np.radians(angle_deg)
            close = np.abs(theta_if - angle_rad) < np.radians(10)
            if close.sum() > 0:
                print(f"    At {angle_deg:3d} deg: "
                      f"mean={np.mean(gmag_if[close]):.4e}"
                      f"  (n={close.sum()})")

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "compare_polar_idw2.png")
    plt.savefig(outpath, bbox_inches='tight')
    print(f"\n  Saved: {outpath}")
    plt.close()

    return results


def plot_gradient_field_comparison():
    """Side-by-side gradient magnitude fields at ts=120."""
    ts = 120
    t_sim = ts * DT
    t_anal = t_sim + T0
    R_anal = analytical_radius(t_anal)
    R_um = R_anal * 1e6

    old_pvtu = os.path.join(OLD_DIR, f"bubble-ts{ts:06d}.pvtu")
    new_pvtu = os.path.join(SCRIVEN_DIR, f"bubble-ts{ts:06d}.pvtu")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ngrid = 400
    yi = np.linspace(-148, 148, ngrid)
    zi = np.linspace(-148, 148, ngrid)
    YI, ZI = np.meshgrid(yi, zi)

    theta_circle = np.linspace(0, 2*np.pi, 200)

    for ax, (pvtu, label) in zip(axes, [
        (old_pvtu, 'Unweighted (old)'),
        (new_pvtu, 'IDW2-weighted (new)'),
    ]):
        if not os.path.exists(pvtu):
            ax.set_title(f'{label}\n(data not available)')
            continue

        data = analyze_slice_gradients(pvtu, t_anal)
        cx_um = data['cx'] * 1e6
        cy_um = data['cy'] * 1e6
        points = np.column_stack([cx_um, cy_um])

        gmag_grid = griddata(points, data['grad_mag'],
                             (YI, ZI), method='linear')
        vof_grid = griddata(points, data['vof'],
                            (YI, ZI), method='linear')

        vmax = np.percentile(
            data['grad_mag'][data['grad_mag'] > 0], 99)
        levels = np.linspace(0, vmax, 30)

        if levels[-1] > 0:
            cf = ax.contourf(YI, ZI, gmag_grid, levels=levels,
                             cmap='hot_r', extend='max')
            fig.colorbar(cf, ax=ax, shrink=0.8,
                         label=r'$|\nabla T|$ [K/m]')

        ax.contour(YI, ZI, vof_grid, levels=[0.5],
                   colors='cyan', linewidths=2)
        ax.plot(R_um*np.cos(theta_circle),
                R_um*np.sin(theta_circle),
                'w--', linewidth=1.5)

        ax.set_xlim([-150, 150])
        ax.set_ylim([-150, 150])
        ax.set_aspect('equal')
        ax.set_title(label)
        ax.set_xlabel(r'$y$ [$\mu$m]')

    axes[0].set_ylabel(r'$z$ [$\mu$m]')

    plt.suptitle(
        f'$|\\nabla T|$ at $t = {t_anal*1e3:.2f}$ ms (structured mesh)',
        fontsize=13, y=1.02)
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "compare_grad_field_idw2.png")
    plt.savefig(outpath, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  IDW2 GRADIENT WEIGHTING - COMPARISON")
    print("=" * 60)

    # Check what data is available
    new_bench = os.path.join(SCRIVEN_DIR, "bench-data.dat")
    new_ts120 = os.path.join(SCRIVEN_DIR, "bubble-ts000120.pvtu")

    if os.path.exists(new_bench):
        n_lines = sum(1 for _ in open(new_bench))
        print(f"\n  New bench-data.dat: {n_lines} time steps")
    else:
        print("\n  New bench-data.dat: not yet available")

    print("\n--- 1. Bubble radius comparison ---")
    plot_radius_comparison()

    print("\n--- 2. Polar gradient anisotropy ---")
    if os.path.exists(new_ts120):
        plot_polar_comparison()
    else:
        print("  Waiting for new ts=120 data...")

    print("\n--- 3. Gradient field comparison ---")
    if os.path.exists(new_ts120):
        plot_gradient_field_comparison()
    else:
        print("  Waiting for new ts=120 data...")

    print("\nDone.")
