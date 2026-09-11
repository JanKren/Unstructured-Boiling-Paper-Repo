#!/usr/bin/env python3
"""
Stefan and Sucking problem: interface position and relative error.
Produces a 2x2 figure for the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

STEFAN_DIR  = os.environ.get("STEFAN_DIR", "/home/jan/runs/tflows-vof/Validation/Stefan")
SUCKING_DIR = os.environ.get("SUCKING_DIR", "/home/jan/runs/tflows-vof/Validation/Sucking")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", os.path.join(os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling"), "pics"))


# ===================================================================
#  Stefan problem
# ===================================================================
def load_stefan():
    """Load Stefan simulation and analytical data."""
    sim = np.loadtxt(os.path.join(STEFAN_DIR, "stefans_solution.dat"))
    t_sim = sim[:, 0]
    x_sim = sim[:, 1]

    ana = np.loadtxt(os.path.join(STEFAN_DIR, "interface_position.dat"),
                     comments='#')
    t_ana = ana[:, 0]
    x_ana = ana[:, 1]

    return t_sim, x_sim, t_ana, x_ana


# ===================================================================
#  Sucking problem
# ===================================================================
def load_sucking():
    """Load Sucking simulation and analytical data.

    The file interface_position.num contains MPI-interleaved data from
    2 ranks.  For each unique time, take the maximum position (the rank
    that tracks the actual front).  Time is shifted by +0.1 s as in the
    original gnuplot script (front.gnu).
    """
    sim = np.loadtxt(os.path.join(SUCKING_DIR, "interface_position.num"))
    # Group by time and take max position per group
    unique_times = np.unique(sim[:, 0])
    max_positions = np.array([sim[sim[:, 0] == t, 1].max()
                              for t in unique_times])
    # Remove MPI artifacts: position must be monotonically non-decreasing
    running_max = np.maximum.accumulate(max_positions)
    valid = max_positions >= running_max - 1e-8
    unique_times  = unique_times[valid]
    max_positions = max_positions[valid]

    t_sim = unique_times + 0.1
    x_sim = max_positions

    ana = np.loadtxt(os.path.join(SUCKING_DIR, "interface_position.exa"),
                     comments='#')
    t_ana = ana[:, 0]
    x_ana = ana[:, 1]

    return t_sim, x_sim, t_ana, x_ana


# ===================================================================
#  Plot
# ===================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t_stef_sim, x_stef_sim, t_stef_ana, x_stef_ana = load_stefan()
    t_suck_sim, x_suck_sim, t_suck_ana, x_suck_ana = load_sucking()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # --- (a) Stefan interface position ---
    ax = axes[0, 0]
    ax.plot(t_stef_ana, x_stef_ana * 1e3, 'k-', linewidth=2,
            label='Analytical')
    ax.plot(t_stef_sim, x_stef_sim * 1e3, 'C0--', linewidth=1.5,
            label='T-Flows')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Interface position [mm]')
    ax.set_title('(a) Stefan problem')
    ax.legend(loc='lower right')

    # --- (b) Stefan relative error ---
    ax = axes[0, 1]
    x_stef_ana_interp = np.interp(t_stef_sim, t_stef_ana, x_stef_ana)
    rel_err_stef = (x_stef_sim - x_stef_ana_interp) / x_stef_ana_interp * 100
    ax.plot(t_stef_sim, rel_err_stef, 'C0-', linewidth=1.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Relative error [%]')
    ax.set_title('(b) Stefan position error')
    ax.axhline(y=0, color='k', linewidth=0.5)
    print(f"Stefan:  mean error = {np.mean(rel_err_stef):.2f}%, "
          f"max |error| = {np.max(np.abs(rel_err_stef)):.2f}%")

    # --- (c) Sucking interface position ---
    ax = axes[1, 0]
    ax.plot(t_suck_ana, x_suck_ana * 1e3, 'k-', linewidth=2,
            label='Analytical')
    step = max(1, len(t_suck_sim) // 500)
    ax.plot(t_suck_sim[::step], x_suck_sim[::step] * 1e3,
            'C0--', linewidth=1.5, label='T-Flows')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Interface position [mm]')
    ax.set_title('(c) Sucking problem')
    ax.legend(loc='lower right')

    # --- (d) Sucking relative error ---
    ax = axes[1, 1]
    x_suck_ana_interp = np.interp(t_suck_sim, t_suck_ana, x_suck_ana)
    rel_err_suck = (x_suck_sim - x_suck_ana_interp) / x_suck_ana_interp * 100
    ax.plot(t_suck_sim[::step], rel_err_suck[::step], 'C0-', linewidth=1.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Relative error [%]')
    ax.set_title('(d) Sucking position error')
    ax.axhline(y=0, color='k', linewidth=0.5)
    print(f"Sucking: mean error = {np.mean(rel_err_suck):.2f}%, "
          f"max |error| = {np.max(np.abs(rel_err_suck)):.2f}%")

    plt.tight_layout()

    for ext in ['.png', '.pdf']:
        out = os.path.join(OUTPUT_DIR, f"stefan_sucking_interface{ext}")
        plt.savefig(out, bbox_inches='tight')
    print(f"Saved: {os.path.join(OUTPUT_DIR, 'stefan_sucking_interface.png')}")
    plt.close()


if __name__ == "__main__":
    main()
