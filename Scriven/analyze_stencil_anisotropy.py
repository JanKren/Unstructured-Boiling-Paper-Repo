#!/usr/bin/env python3
"""
Analysis of the least-squares gradient stencil anisotropy
caused by the front-modified gradient (Calculate_Grad_Matrix_With_Front).

Demonstrates why a structured (hex) mesh produces 4-fold gradient
anisotropy for a spherically symmetric temperature field when the
LSQ stencil is modified at the interface, and evaluates potential
fixes (inverse distance weighting, extended stencil).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'savefig.dpi': 300,
    'axes.grid': False,
    'lines.linewidth': 1.5,
})

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling"), "pics"))

# =======================================================================
#  1. MATHEMATICAL FRAMEWORK
# =======================================================================
#
#  The LSQ gradient at cell c minimizes:
#
#    J = sum_f [ w_f * (grad(phi).delta_f - dphi_f)^2 ]
#
#  where:
#    delta_f = (dx, dy, dz) is the displacement vector for face f
#    dphi_f  = phi(neighbor) - phi(c)
#    w_f     = weight for face f
#
#  The solution is:  grad(phi) = G^{-1} . b
#
#  where the 3x3 matrix G and RHS vector b are:
#
#    G = sum_f [ w_f * delta_f (x) delta_f ]     (outer product)
#    b = sum_f [ w_f * dphi_f * delta_f ]
#
#  For UNWEIGHTED LSQ (w_f = 1): G = sum [ delta delta^T ]
#  For IDW-weighted  (w_f = 1/|delta|^2): normalized directions
# =======================================================================


def temperature_scriven(y, z, R, T_sat, T_inf, k=3.0):
    """Scriven-like temperature field: T = T_sat inside bubble,
    thin thermal boundary layer outside.

    Uses a smooth approximation: T = T_inf - (T_inf-T_sat)*R/r
    which has dT/dr = (T_inf - T_sat) * R / r^2
    """
    r = np.sqrt(y**2 + z**2)
    r = np.maximum(r, 1e-12)  # avoid division by zero
    T = np.where(r <= R, T_sat,
                 T_inf - (T_inf - T_sat) * (R / r)**k)
    return T


def analytical_gradient(y, z, R, T_sat, T_inf, k=3.0):
    """Analytical gradient of the Scriven-like temperature field."""
    r = np.sqrt(y**2 + z**2)
    r = np.maximum(r, 1e-12)
    dTdr = np.where(r <= R, 0.0,
                    k * (T_inf - T_sat) * R**k / r**(k+1))
    dTdy = dTdr * y / r
    dTdz = dTdr * z / r
    return dTdy, dTdz


def compute_lsq_gradient_2d(yc, zc, neighbors, phi_vals, phi_c,
                            weighting='none'):
    """Compute 2D LSQ gradient at cell center (yc, zc).

    Parameters:
      neighbors: list of (y, z) positions of neighbors/front points
      phi_vals:  corresponding phi values at those positions
      phi_c:     phi value at cell center
      weighting: 'none', 'idw' (inverse distance), or 'idw2'

    Returns:
      (dphidy, dphidz) gradient components
    """
    G = np.zeros((2, 2))
    b = np.zeros(2)

    for (yn, zn), phi_n in zip(neighbors, phi_vals):
        dy = yn - yc
        dz = zn - zc
        dphi = phi_n - phi_c
        delta = np.array([dy, dz])
        dist2 = dy**2 + dz**2

        if dist2 < 1e-30:
            continue

        if weighting == 'none':
            w = 1.0
        elif weighting == 'idw':
            w = 1.0 / np.sqrt(dist2)
        elif weighting == 'idw2':
            w = 1.0 / dist2
        else:
            w = 1.0

        G += w * np.outer(delta, delta)
        b += w * dphi * delta

    if np.linalg.det(G) < 1e-30:
        return 0.0, 0.0

    grad = np.linalg.solve(G, b)
    return grad[0], grad[1]


def circle_face_intersection(y1, z1, y2, z2, R):
    """Find intersection of a line segment (y1,z1)-(y2,z2)
    with circle of radius R centered at origin.
    Returns (y_int, z_int) or None.
    """
    dy = y2 - y1
    dz = z2 - z1
    a = dy**2 + dz**2
    b = 2 * (y1*dy + z1*dz)
    c = y1**2 + z1**2 - R**2
    disc = b**2 - 4*a*c
    if disc < 0:
        return None

    t1 = (-b - np.sqrt(disc)) / (2*a)
    t2 = (-b + np.sqrt(disc)) / (2*a)

    for t in [t1, t2]:
        if 0 < t < 1:
            return (y1 + t*dy, z1 + t*dz)
    return None


def analyze_cell_gradient(yc, zc, h, R, T_sat, T_inf,
                          use_front=True, weighting='none',
                          extended_stencil=False):
    """Analyze the LSQ gradient for a single cell on a 2D Cartesian mesh.

    Returns: (grad_y, grad_z, grad_y_anal, grad_z_anal)
    """
    # Face-connected neighbors (Cartesian 2D: 4 neighbors)
    face_neighbors = [
        (yc + h, zc),  # +y
        (yc - h, zc),  # -y
        (yc, zc + h),  # +z
        (yc, zc - h),  # -z
    ]
    # Face midpoints (for intersection detection)
    face_midpoints = [
        (yc + h/2, zc),
        (yc - h/2, zc),
        (yc, zc + h/2),
        (yc, zc - h/2),
    ]

    if extended_stencil:
        # Add vertex (diagonal) neighbors
        face_neighbors += [
            (yc + h, zc + h),
            (yc + h, zc - h),
            (yc - h, zc + h),
            (yc - h, zc - h),
        ]
        face_midpoints += [
            (yc + h/2, zc + h/2),
            (yc + h/2, zc - h/2),
            (yc - h/2, zc + h/2),
            (yc - h/2, zc - h/2),
        ]

    neighbors = []
    phi_vals = []

    for (yn, zn), (ym, zm) in zip(face_neighbors, face_midpoints):
        if use_front:
            # Check if the line from cell center to neighbor
            # crosses the front
            intersection = circle_face_intersection(yc, zc, yn, zn, R)
            if intersection is not None:
                yi, zi = intersection
                # Use front intersection point with T_sat
                neighbors.append((yi, zi))
                phi_vals.append(T_sat)
                continue

        # No front intersection: use neighbor cell center
        neighbors.append((yn, zn))
        phi_vals.append(temperature_scriven(yn, zn, R, T_sat, T_inf))

    phi_c = temperature_scriven(yc, zc, R, T_sat, T_inf)
    grad_y, grad_z = compute_lsq_gradient_2d(
        yc, zc, neighbors, phi_vals, phi_c, weighting)

    grad_y_a, grad_z_a = analytical_gradient(yc, zc, R, T_sat, T_inf)

    return grad_y, grad_z, grad_y_a, grad_z_a


def plot_stencil_analysis():
    """Main analysis: compute gradient error vs angle for different
    configurations and show the 4-fold anisotropy pattern.
    """
    R = 1.0e-4      # 100 um bubble radius
    h = 4.0e-6      # 4 um cell size
    T_sat = 373.15   # K
    T_inf = 374.40   # K (1.25 K superheat)

    # Place cells in a ring just outside the bubble
    n_angles = 360
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    r_cell = R + 0.7 * h  # Just outside the interface

    configs = [
        ('Standard LSQ (no front)',  False, 'none', False),
        ('Front-modified (current)', True,  'none', False),
        ('Front + IDW weighting',    True,  'idw',  False),
        ('Front + IDW2 weighting',   True,  'idw2', False),
        ('Front + extended stencil', True,  'none', True),
    ]

    results = {}
    anal_mags = np.zeros(n_angles)
    for label, use_front, weight, ext_stencil in configs:
        grad_mags = np.zeros(n_angles)
        errors = np.zeros(n_angles)
        for i, theta in enumerate(angles):
            yc = r_cell * np.cos(theta)
            zc = r_cell * np.sin(theta)
            gy, gz, gy_a, gz_a = analyze_cell_gradient(
                yc, zc, h, R, T_sat, T_inf,
                use_front=use_front, weighting=weight,
                extended_stencil=ext_stencil)
            grad_mags[i] = np.sqrt(gy**2 + gz**2)
            anal_mag = np.sqrt(gy_a**2 + gz_a**2)
            anal_mags[i] = anal_mag
            if anal_mag > 0:
                errors[i] = (grad_mags[i] - anal_mag) / anal_mag
        results[label] = {
            'grad_mag': grad_mags,
            'error': errors,
        }

    # ---- Figure 1: Polar plot of |grad T| / |grad T|_analytical ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                             subplot_kw={'projection': 'polar'})

    # Panel 1: Standard vs front-modified
    ax = axes[0]
    for label, color, ls in [
        ('Standard LSQ (no front)', 'blue', '--'),
        ('Front-modified (current)', 'red', '-'),
    ]:
        gm = results[label]['grad_mag'] / anal_mags
        ax.plot(angles, gm, color=color, ls=ls, label=label)
    ax.plot(angles, np.ones_like(angles), 'k:', alpha=0.4, lw=0.8)
    ax.set_title('Standard vs Front-modified', pad=20)
    ax.legend(loc='center', bbox_to_anchor=(0.5, 0.35),
              fontsize=14, framealpha=0.9)

    # Panel 2: Front-modified vs weighted fixes
    ax = axes[1]
    for label, color, ls in [
        ('Front-modified (current)', 'red', '-'),
        ('Front + IDW weighting', 'green', '-'),
        ('Front + IDW2 weighting', 'purple', '--'),
    ]:
        gm = results[label]['grad_mag'] / anal_mags
        ax.plot(angles, gm, color=color, ls=ls, label=label)
    ax.plot(angles, np.ones_like(angles), 'k:', alpha=0.4, lw=0.8)
    ax.set_title('Weighting schemes', pad=20)
    ax.legend(loc='center', bbox_to_anchor=(0.5, 0.35),
              fontsize=14, framealpha=0.9)

    # Panel 3: Front-modified vs extended stencil
    ax = axes[2]
    for label, color, ls in [
        ('Front-modified (current)', 'red', '-'),
        ('Front + extended stencil', 'orange', '-'),
    ]:
        gm = results[label]['grad_mag'] / anal_mags
        ax.plot(angles, gm, color=color, ls=ls, label=label)
    ax.plot(angles, np.ones_like(angles), 'k:', alpha=0.4, lw=0.8)
    ax.set_title('Extended stencil', pad=20)
    ax.legend(loc='center', bbox_to_anchor=(0.5, 0.35),
              fontsize=14, framealpha=0.9)

    plt.tight_layout()
    outpath = f"{OUTPUT_DIR}/stencil_polar_comparison.png"
    plt.savefig(outpath, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()

    # ---- Figure 2: Gradient error vs angle (Cartesian) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    angles_deg = np.degrees(angles)
    for label, color, ls in [
        ('Standard LSQ (no front)', 'blue', '--'),
        ('Front-modified (current)', 'red', '-'),
        ('Front + IDW weighting', 'green', '-'),
        ('Front + IDW2 weighting', 'purple', '--'),
        ('Front + extended stencil', 'orange', '-'),
    ]:
        err = results[label]['error'] * 100  # percent
        ax.plot(angles_deg, err, color=color, ls=ls, label=label)
    ax.set_xlabel('Angle [deg]')
    ax.set_ylabel('Gradient magnitude error [%]')
    ax.set_title('Relative error in $|\\nabla T|$')
    ax.set_xlim([0, 360])
    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Print summary statistics
    ax = axes[1]
    labels_short = ['No front', 'Current', 'IDW', 'IDW2', 'Extended']
    labels_full = list(results.keys())
    min_max_ratios = []
    rms_errors = []
    for label in labels_full:
        gm = results[label]['grad_mag']
        valid = gm[gm > 0]
        ratio = valid.min() / valid.max() if len(valid) > 0 else 0
        min_max_ratios.append(ratio)
        rms_errors.append(np.sqrt(np.mean(results[label]['error']**2)) * 100)

    x = np.arange(len(labels_short))
    width = 0.35
    bars1 = ax.bar(x - width/2, min_max_ratios, width,
                   label='min/max ratio', color='steelblue')
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, rms_errors, width,
                    label='RMS error [%]', color='salmon')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=30, ha='right')
    ax.set_ylabel('min/max ratio (closer to 1 = better)')
    ax2.set_ylabel('RMS error [%]')
    ax.set_title('Isotropy metrics')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.set_ylim([0, 1.1])
    ax.axhline(1.0, color='gray', ls=':', lw=0.5)

    plt.tight_layout()
    outpath = f"{OUTPUT_DIR}/stencil_error_comparison.png"
    plt.savefig(outpath, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()

    # Print table
    print("\n  Stencil anisotropy summary:")
    print(f"  {'Config':<30s} {'min/max':>8s} {'RMS err %':>10s}")
    print("  " + "-" * 50)
    for label, ratio, rms in zip(labels_full, min_max_ratios, rms_errors):
        print(f"  {label:<30s} {ratio:8.4f} {rms:10.2f}")


def plot_stencil_geometry():
    """Visualize the stencil geometry for cells at 0, 22.5, and 45 deg.
    Shows displacement vectors and how the front modifies them.
    """
    R = 1.0e-4
    h = 4.0e-6
    # 0.7 h put the 45 deg cell's -y and -z neighbours at r = 100.01 um
    # against R = 100, a hundredth of a micron outside the interface, so
    # neither face was cut and the panel the caption calls the two-face
    # case showed none.  At 0.5 h the axis cell still cuts one face and
    # the diagonal cell cuts two, which is the contrast the figure exists
    # to draw.
    r_cell = R + 0.5 * h

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, angle_deg in zip(axes, [0, 22.5, 45]):
        theta = np.radians(angle_deg)
        yc = r_cell * np.cos(theta)
        zc = r_cell * np.sin(theta)

        # Draw the bubble
        circle = Circle((0, 0), R*1e6, fill=False,
                         color='cyan', linewidth=2, ls='--')
        ax.add_patch(circle)

        # Draw the Cartesian grid near this cell
        y_lo = (yc - 2*h) * 1e6
        y_hi = (yc + 2*h) * 1e6
        z_lo = (zc - 2*h) * 1e6
        z_hi = (zc + 2*h) * 1e6

        # Grid lines
        ny = int(round((y_hi - y_lo) / (h*1e6))) + 1
        nz = int(round((z_hi - z_lo) / (h*1e6))) + 1
        for i in range(ny + 1):
            yy = y_lo + i * h * 1e6
            ax.plot([yy, yy], [z_lo, z_hi], 'gray', lw=0.5, alpha=0.5)
        for j in range(nz + 1):
            zz = z_lo + j * h * 1e6
            ax.plot([y_lo, y_hi], [zz, zz], 'gray', lw=0.5, alpha=0.5)

        # Cell center
        ax.plot(yc*1e6, zc*1e6, 'ko', ms=8, zorder=5)

        # Face-connected neighbors
        face_neighbors = [
            (yc + h, zc), (yc - h, zc),
            (yc, zc + h), (yc, zc - h),
        ]

        for yn, zn in face_neighbors:
            # Check for front intersection
            intersection = circle_face_intersection(yc, zc, yn, zn, R)

            if intersection is not None:
                yi, zi = intersection
                # Draw shortened vector (to front)
                ax.annotate('', xy=(yi*1e6, zi*1e6),
                            xytext=(yc*1e6, zc*1e6),
                            arrowprops=dict(arrowstyle='->', color='red',
                                            lw=2.5))
                ax.plot(yi*1e6, zi*1e6, 'r*', ms=12, zorder=5)
                # Draw the would-be full vector as dashed
                ax.plot([yc*1e6, yn*1e6], [zc*1e6, zn*1e6],
                        'r--', lw=0.8, alpha=0.4)
            else:
                # Draw normal vector (to neighbor)
                ax.annotate('', xy=(yn*1e6, zn*1e6),
                            xytext=(yc*1e6, zc*1e6),
                            arrowprops=dict(arrowstyle='->', color='blue',
                                            lw=2))
                ax.plot(yn*1e6, zn*1e6, 'bs', ms=6, zorder=5)

        ax.set_aspect('equal')
        ax.set_title(f'Cell at {angle_deg}$^\\circ$')
        ax.set_xlabel(r'$y$ [$\mu$m]')
        if angle_deg == 0:
            ax.set_ylabel(r'$z$ [$\mu$m]')

        # Zoom to cell neighborhood
        margin = 2.5 * h * 1e6
        ax.set_xlim([yc*1e6 - margin, yc*1e6 + margin])
        ax.set_ylim([zc*1e6 - margin, zc*1e6 + margin])

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, marker='s', ms=6,
               label='Full vector (unmodified face)'),
        Line2D([0], [0], color='red', lw=2.5, marker='*', ms=12,
               label='Shortened vector (front intersection)'),
        Line2D([0], [0], color='cyan', ls='--', lw=2,
               label='Bubble interface'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    outpath = f"{OUTPUT_DIR}/stencil_geometry.png"
    plt.savefig(outpath, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()


def plot_condition_number():
    """Plot the condition number of the gradient matrix G
    as a function of angle around the interface.
    Shows how the front modification worsens conditioning at 0/90 deg.
    """
    R = 1.0e-4
    h = 4.0e-6
    r_cell = R + 0.7 * h

    n_angles = 360
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)

    cond_standard = np.zeros(n_angles)
    cond_front = np.zeros(n_angles)
    cond_front_idw = np.zeros(n_angles)

    for i, theta in enumerate(angles):
        yc = r_cell * np.cos(theta)
        zc = r_cell * np.sin(theta)

        face_neighbors = [
            (yc + h, zc), (yc - h, zc),
            (yc, zc + h), (yc, zc - h),
        ]

        for mode in ['standard', 'front', 'front_idw']:
            G = np.zeros((2, 2))

            for yn, zn in face_neighbors:
                if mode in ('front', 'front_idw'):
                    intersection = circle_face_intersection(
                        yc, zc, yn, zn, R)
                    if intersection is not None:
                        yn, zn = intersection

                dy = yn - yc
                dz = zn - zc
                delta = np.array([dy, dz])
                dist2 = dy**2 + dz**2
                if dist2 < 1e-30:
                    continue

                if mode == 'front_idw':
                    w = 1.0 / dist2
                else:
                    w = 1.0
                G += w * np.outer(delta, delta)

            if np.linalg.det(G) > 1e-30:
                cond = np.linalg.cond(G)
            else:
                cond = np.inf

            if mode == 'standard':
                cond_standard[i] = cond
            elif mode == 'front':
                cond_front[i] = cond
            else:
                cond_front_idw[i] = cond

    fig, ax = plt.subplots(figsize=(10, 5))
    angles_deg = np.degrees(angles)
    ax.semilogy(angles_deg, cond_standard, 'b--',
                label='Standard (no front)')
    ax.semilogy(angles_deg, cond_front, 'r-',
                label='Front-modified (current)')
    ax.semilogy(angles_deg, cond_front_idw, 'g-',
                label='Front + IDW2 weighting')
    ax.set_xlabel('Angle [deg]')
    ax.set_ylabel('Condition number of G')
    ax.set_title('Gradient matrix conditioning vs angle around interface')
    ax.set_xlim([0, 360])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add vertical lines at 0, 45, 90 etc
    for a in [0, 45, 90, 135, 180, 225, 270, 315]:
        ax.axvline(a, color='gray', ls=':', lw=0.5, alpha=0.5)

    plt.tight_layout()
    outpath = f"{OUTPUT_DIR}/stencil_condition_number.png"
    plt.savefig(outpath, bbox_inches='tight')
    print(f"  Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  STENCIL ANISOTROPY ANALYSIS")
    print("=" * 60)

    print("\n--- 1. Stencil geometry visualization ---")
    plot_stencil_geometry()

    print("\n--- 2. Gradient magnitude vs angle ---")
    plot_stencil_analysis()

    print("\n--- 3. Condition number analysis ---")
    plot_condition_number()

    print("\nDone.")
