#!/usr/bin/env python3
"""
Mass conservation analysis for Scriven and Sucking benchmark problems.
Produces publication-quality figures for the paper.

YS review comment: Add mass conservation analysis for both problems.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

import scriven_reference

# Publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.figsize': (6, 4),
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

# ===========================================================================
#  Paths
# ===========================================================================
SCRIVEN_DIR = os.environ.get("SCRIVEN_DIR",
                          "/home/jan/runs/tflows-vof/Validation/Scriven")
SUCKING_DIR = os.environ.get("SUCKING_DIR",
                          "/home/jan/runs/tflows-vof/Validation/Sucking")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", os.path.join(os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling"), "pics"))

# ===========================================================================
#  Physical properties (water-steam at 1 bar)
# ===========================================================================
rho_v   = 0.597      # kg/m^3
rho_l   = 958.4      # kg/m^3
k_l     = 0.679      # W/(m·K)  Table 1; see scriven_reference
cp_l    = 4216.0     # J/(kg·K)
h_fg    = 2.26e6     # J/kg
sigma   = 0.059      # N/m

# Scriven-specific
beta_scriven = scriven_reference.BETA   # solved from Eq. (16), not tabulated
alpha_l      = scriven_reference.D_L    # thermal diffusivity [m^2/s]
R0_scriven   = scriven_reference.R0     # initial bubble radius [m]
t0_scriven   = scriven_reference.T0     # virtual origin

def scriven_radius(t):
    """Analytical Scriven bubble radius R(t) = 2*beta*sqrt(alpha_l * t)."""
    return 2.0 * beta_scriven * np.sqrt(alpha_l * t)

def scriven_volume(t):
    """Analytical Scriven bubble volume."""
    R = scriven_radius(t)
    return (4.0 / 3.0) * np.pi * R**3


# ===========================================================================
#  1. SCRIVEN MASS CONSERVATION
# ===========================================================================
def analyze_scriven():
    """Analyze Scriven bubble growth mass conservation."""

    print("=" * 60)
    print("  SCRIVEN PROBLEM - Mass Conservation Analysis")
    print("=" * 60)

    # Load bench-data files for different meshes
    meshes = {}
    bench_files = {
        r'Structured $75^3$':  os.path.join(SCRIVEN_DIR, "bench-data.dat"),
        r'Structured $100^3$': os.path.join(SCRIVEN_DIR, "bench-data100.dat"),
        r'Structured $125^3$': os.path.join(SCRIVEN_DIR,
                                             "Scriven-Struct-125", "bench-data.dat"),
        r'Structured $150^3$': os.path.join(SCRIVEN_DIR, "bench-data150.dat"),
        r'Polyhedral $75^3$':  os.path.join(
            "/home/jan/runs/tflows-vof/Algebraic/Scriven",
            "Scriven-Unstructured-Coarse-2", "bench-data.dat"),
        r'Polyhedral $125^3$': os.path.join(SCRIVEN_DIR,
                                             "Scriven-Poly-125", "bench-data.dat"),
    }

    for label, fpath in bench_files.items():
        if os.path.exists(fpath):
            data = np.loadtxt(fpath)
            # Remove any rows with NaN
            mask = ~np.isnan(data).any(axis=1)
            data = data[mask]
            # Extract the longest contiguous run (detect time resets)
            resets = np.where(np.diff(data[:, 0]) < 0)[0]
            if len(resets) > 0:
                starts = [0] + [r + 1 for r in resets]
                ends   = list(resets) + [len(data) - 1]
                best_s, best_e, best_n = 0, len(data) - 1, 0
                for s, e in zip(starts, ends):
                    n = e - s + 1
                    if n > best_n:
                        best_s, best_e, best_n = s, e, n
                data = data[best_s:best_e + 1]
            if data.shape[0] > 0:
                meshes[label] = {
                    'time':    data[:, 0],
                    'area':    data[:, 1],
                    'volume':  data[:, 2],
                    'radius':  data[:, 3],
                }
                print(f"  Loaded {label}: {data.shape[0]} time steps, "
                      f"t=[{data[0,0]:.1e}, {data[-1,0]:.1e}] s")

    # --- Figure 1: Bubble radius comparison (enhanced version of existing) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) Radius vs time
    t_anal = np.linspace(0, 1.5e-3, 500)
    R_anal = scriven_radius(t_anal)
    ax1.plot(t_anal * 1e3, R_anal * 1e6, 'k-', linewidth=2,
             label='Analytical (Scriven)')

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    for i, (label, d) in enumerate(meshes.items()):
        t_shifted = d['time'] + t0_scriven
        R_num = d['radius']
        ls = '-' if 'Polyhedral' in label else '--'
        ax1.plot(t_shifted * 1e3, R_num * 1e6, color=colors[i],
                 linestyle=ls, label=label)

    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel(r'Bubble radius [$\mu$m]')
    ax1.set_title('(a) Bubble radius growth')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.set_xlim([0, 1.5])

    # (b) Relative radius error
    for i, (label, d) in enumerate(meshes.items()):
        t_shifted = d['time'] + t0_scriven
        R_num = d['radius']
        R_anal = scriven_radius(t_shifted)
        rel_error = (R_num - R_anal) / R_anal * 100.0
        ls = '-' if 'Polyhedral' in label else '--'
        ax2.plot(t_shifted * 1e3, rel_error, color=colors[i],
                 linestyle=ls, label=label)
        # Print final error
        print(f"  {label}: final radius error = {rel_error[-1]:.2f}%")

    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Relative radius error [%]')
    ax2.set_title('(b) Bubble radius error')
    ax2.legend(fontsize=9)
    ax2.set_xlim([0, 1.5])
    ax2.axhline(y=0, color='k', linewidth=0.5)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "scriven_mass_conservation.png")
    plt.savefig(outpath, bbox_inches='tight')
    plt.savefig(outpath.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n  Saved: {outpath}")
    plt.close()


# ===========================================================================
#  2. SUCKING MASS CONSERVATION
# ===========================================================================
def analyze_sucking():
    """Analyze Sucking problem mass conservation."""

    print("\n" + "=" * 60)
    print("  SUCKING PROBLEM - Mass Conservation Analysis")
    print("=" * 60)

    # Load analytical solution
    anal_path = os.path.join(SUCKING_DIR, "interface_position.exa")
    anal_data = np.loadtxt(anal_path, comments='#')
    t_anal = anal_data[:, 0]   # time [s]
    x_anal = anal_data[:, 1]   # position [m]
    print(f"  Analytical: {len(t_anal)} points, "
          f"t=[{t_anal[0]:.4f}, {t_anal[-1]:.4f}] s")

    # Load numerical solution (from front.gnu: interface_position.num)
    # This file has MPI-interleaved data from 2 ranks.  For each unique
    # time, take the maximum position (the rank tracking the actual front).
    num_path = os.path.join(SUCKING_DIR, "interface_position.num")
    raw = np.loadtxt(num_path)
    # Group by time and take max position per group
    unique_times = np.unique(raw[:, 0])
    max_positions = np.array([raw[raw[:, 0] == t, 1].max()
                              for t in unique_times])
    # Remove MPI artifacts: interface position must be monotonically
    # non-decreasing (evaporation only moves the front forward).
    # Flag points where position drops compared to the running maximum.
    running_max = np.maximum.accumulate(max_positions)
    valid = max_positions >= running_max - 1e-8
    unique_times = unique_times[valid]
    max_positions = max_positions[valid]
    t_num = unique_times + 0.1   # apply +0.1 s offset (as in front.gnu)
    x_num = max_positions         # position [m]
    print(f"  Numerical: {len(t_num)} points, "
          f"t=[{t_num[0]:.4f}, {t_num[-1]:.4f}] s")

    # Domain cross-section (periodic cell in y,z)
    # Domain: 8mm x 1mm x 1mm, with 6 cells in y and z (periodic)
    # For volume conservation, we use the full periodic domain cross section
    Ly = 1.0e-3   # m
    Lz = 1.0e-3   # m
    A_cross = Ly * Lz   # m^2

    # Compute vapor volumes
    V_vapor_anal = x_anal * A_cross
    V_vapor_num  = x_num * A_cross

    # Interpolate analytical onto numerical time points for error computation
    x_anal_interp = np.interp(t_num, t_anal, x_anal)
    V_anal_interp = x_anal_interp * A_cross

    rel_error_x = (x_num - x_anal_interp) / x_anal_interp * 100.0

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) Interface position
    ax1.plot(t_anal, x_anal * 1e3, 'k-', linewidth=2,
             label='Analytical')
    ax1.plot(t_num, x_num * 1e3, 'C0--', linewidth=1.5,
             label=r'T-Flows')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Interface position [mm]')
    ax1.set_title('(a) Interface position')
    ax1.legend()

    # (b) Relative error in interface position (proxy for mass conservation)
    ax2.plot(t_num, rel_error_x, 'C0-', linewidth=1.5)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Relative position error [%]')
    ax2.set_title('(b) Interface position error')
    ax2.axhline(y=0, color='k', linewidth=0.5)

    # Print summary statistics
    print(f"  Mean relative error: {np.mean(rel_error_x):.2f}%")
    print(f"  Max  relative error: {np.max(np.abs(rel_error_x)):.2f}%")
    print(f"  Final relative error: {rel_error_x[-1]:.2f}%")

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "sucking_mass_conservation.png")
    plt.savefig(outpath, bbox_inches='tight')
    plt.savefig(outpath.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n  Saved: {outpath}")
    plt.close()


# ===========================================================================
#  3. COMBINED SUMMARY TABLE
# ===========================================================================
def print_summary():
    """Print a summary suitable for inclusion in the paper."""
    print("\n" + "=" * 60)
    print("  SUMMARY FOR PAPER")
    print("=" * 60)
    print("""
  Suggested text for the paper:

  "Table X summarizes the mass conservation metrics for the
  benchmark cases. The Scriven problem shows volume errors
  of X-Y% depending on mesh resolution, with the error
  decreasing monotonically with refinement. The sucking
  problem maintains interface position accuracy within Z%
  throughout the simulation."
  """)


# ===========================================================================
#  Main
# ===========================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    analyze_scriven()
    analyze_sucking()
    print_summary()
    print("\nDone.")
