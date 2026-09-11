#!/usr/bin/env python3
"""
Extract temperature and velocity profiles from Sucking problem VTU files
and compare with analytical solution.

YS review comment: Add instantaneous temperature and velocity profiles.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import pyvista as pv
import os
import glob

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.figsize': (12, 5),
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

# ===========================================================================
#  Physical properties (matching Sucking_Problem.f90)
# ===========================================================================
RHO_G   = 0.597
RHO_L   = 958.4
K_G     = 0.025
K_L     = 0.679
CP_G    = 2030.0
CP_L    = 4216.0
H_LG    = 2.26e6
T_WALL  = 10.0
T_SAT   = 10.0
T_INF   = 15.0
ALPHA_G = K_G / (RHO_G * CP_G)
ALPHA_L = K_L / (RHO_L * CP_L)

SUCKING_DIR = os.environ.get("SUCKING_DIR", "/home/jan/runs/tflows-vof/Validation/Sucking")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", os.path.join(os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling"), "pics"))


def compute_beta():
    """Solve the transcendental equation for beta (bisection method)."""
    def transc(beta):
        inb = beta**2 * (ALPHA_G * RHO_G**2) / (ALPHA_L * RHO_L**2)
        return (beta
                - ((T_INF - T_SAT) * CP_G * K_L * np.sqrt(ALPHA_G) * np.exp(-inb))
                / (H_LG * K_G * np.sqrt(np.pi * ALPHA_L) * erfc(np.sqrt(inb))))

    # Bisection
    a, b = 0.0, 1.0
    for _ in range(200):
        m = 0.5 * (a + b)
        if transc(m) < 0:
            a = m
        else:
            b = m
    beta = 0.5 * (a + b)
    print(f"  Beta = {beta:.8f}")
    return beta


def analytical_interface_position(beta, t):
    """Interface position at time t."""
    return 2.0 * beta * np.sqrt(ALPHA_G * t)


def analytical_temperature(beta, x, t):
    """Analytical temperature distribution T(x) at time t."""
    ipos = analytical_interface_position(beta, t)
    T = np.full_like(x, T_SAT)
    mask = x > ipos
    if np.any(mask):
        arg_erfc_denom = beta * (RHO_G * np.sqrt(ALPHA_G)) / (RHO_L * np.sqrt(ALPHA_L))
        arg_erfc_num = (x[mask] / (2.0 * np.sqrt(ALPHA_L * t))
                        + beta * (RHO_G - RHO_L) / RHO_L
                        * np.sqrt(ALPHA_G / ALPHA_L))
        T[mask] = T_INF - (T_INF - T_SAT) / erfc(arg_erfc_denom) * erfc(arg_erfc_num)
    return T


def analytical_velocity(beta, x, t):
    """Analytical velocity for the 1D Sucking problem.

    With wall at x=0 (u=0) and incompressible phases:
    - Vapor (0 < x < x_interface): u = 0 (div(u)=0 with wall BC)
    - Liquid (x > x_interface): u = v_interface * (1 - rho_g/rho_l)
      where v_interface = dx_interface/dt is the interface velocity
    """
    ipos = analytical_interface_position(beta, t)
    v_int = beta * np.sqrt(ALPHA_G / t)  # interface velocity dx/dt

    u = np.zeros_like(x)
    liquid = x > ipos

    # Liquid pushed at uniform velocity by the expanding vapor
    u[liquid] = v_int * (1.0 - RHO_G / RHO_L)

    return u


def extract_1d_profile(vtu_path):
    """Extract 1D profile along x-axis from VTU file."""
    mesh = pv.read(vtu_path)
    centers = mesh.cell_centers()
    x = centers.points[:, 0]
    y = centers.points[:, 1]
    z = centers.points[:, 2]

    # Select cells near axis (single row in y-z)
    mask = (np.abs(y) < 1e-4) & (np.abs(z) < 1e-4)

    x_1d = x[mask]
    T_1d = mesh.cell_data['Temperature [K]'][mask]
    u_1d = mesh.cell_data['Velocity [m/s]'][mask, 0]  # x-component
    vof_1d = mesh.cell_data['Vof Sharp [1]'][mask]

    # Sort by x
    idx = x_1d.argsort()
    return x_1d[idx], T_1d[idx], u_1d[idx], vof_1d[idx]


def main():
    print("=" * 60)
    print("  SUCKING PROBLEM - Temperature & Velocity Profiles")
    print("=" * 60)

    beta = compute_beta()

    # Simulation time step and start time
    dt_sim = 2.5e-5   # s
    t_start_analytical = 0.1  # analytical solution starts at t=0.1s

    # Select 3 time instants for comparison
    # The simulation initial condition corresponds to t_analytical = 0.1 s
    # VTU file ts_XXXXXX corresponds to time step XXXXXX
    # t_analytical = t_start_analytical + ts * dt_sim

    # Choose time steps that correspond to nice analytical times
    time_steps = [4000, 10000, 16000]  # ts numbers
    t_analytical = [t_start_analytical + ts * dt_sim for ts in time_steps]
    # t_analytical = [0.2, 0.35, 0.5] s

    print(f"  Analytical times: {[f'{t:.2f}' for t in t_analytical]} s")

    colors = ['C0', 'C1', 'C2']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, (ts, t_anal) in enumerate(zip(time_steps, t_analytical)):
        vtu_path = os.path.join(SUCKING_DIR, f"sucking-ts{ts:06d}.vtu")

        if not os.path.exists(vtu_path):
            print(f"  WARNING: {vtu_path} not found, skipping")
            continue

        print(f"  Loading ts={ts} (t_anal={t_anal:.3f} s)...")
        x_num, T_num, u_num, vof_num = extract_1d_profile(vtu_path)

        # Compute analytical solution at same x coordinates
        x_fine = np.linspace(0, 0.008, 2000)
        T_anal = analytical_temperature(beta, x_fine, t_anal)
        u_anal = analytical_velocity(beta, x_fine, t_anal)
        ipos = analytical_interface_position(beta, t_anal)

        # (a) Temperature profiles (as Delta T = T - T_sat)
        ax = axes[0]
        ax.plot(x_fine * 1e3, T_anal - T_SAT, color=colors[i], linestyle='-',
                linewidth=2, alpha=0.6,
                label=f'Analytical, $t = {t_anal:.2f}$ s')
        ax.plot(x_num * 1e3, T_num - T_SAT, color=colors[i], linestyle='--',
                marker='o', markevery=20, markersize=3,
                label=f'T-Flows, $t = {t_anal:.2f}$ s')

        # (b) Velocity profiles
        ax = axes[1]
        ax.plot(x_fine * 1e3, u_anal * 1e3, color=colors[i], linestyle='-',
                linewidth=2, alpha=0.6,
                label=f'Analytical, $t = {t_anal:.2f}$ s')
        ax.plot(x_num * 1e3, u_num * 1e3, color=colors[i], linestyle='--',
                marker='o', markevery=20, markersize=3,
                label=f'T-Flows, $t = {t_anal:.2f}$ s')

        print(f"    Interface position: analytical={ipos*1e3:.3f} mm")

    axes[0].set_xlabel('Position $x$ [mm]')
    axes[0].set_ylabel(r'$\Delta T = T - T_{\mathrm{sat}}$ [K]')
    axes[0].set_title('(a) Temperature profiles')
    axes[0].set_xlim([0, 8])
    axes[0].set_ylim([-0.5, 5.5])

    axes[1].set_xlabel('Position $x$ [mm]')
    axes[1].set_ylabel('Velocity $U_x$ [mm/s]')
    axes[1].set_title('(b) Velocity profiles')
    axes[1].set_xlim([0, 8])

    # Reorder legend: left column = Analytical, right column = T-Flows
    # Plot order is [A1, TF1, A2, TF2, A3, TF3] (indices 0-5)
    # Reorder to [A1, A2, A3, TF1, TF2, TF3] so each row has matching colors
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        order = [0, 2, 4, 1, 3, 5]
        ax.legend([handles[i] for i in order],
                  [labels[i] for i in order],
                  fontsize=8, ncol=2)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "sucking_profiles.png")
    plt.savefig(outpath, bbox_inches='tight')
    plt.savefig(outpath.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n  Saved: {outpath}")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
    print("\nDone.")
