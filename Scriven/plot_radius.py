#!/usr/bin/env python3
"""
Scriven bubble growth: bubble radius comparison across meshes.

Plots the numerical bubble radius on all available meshes against the
analytical Scriven solution, including reference data from
Bureš & Sato (2021).

Usage:
    cd Scriven
    python3 plot_radius.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
    'xtick.minor.size': 2.0,
    'ytick.minor.size': 2.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'lines.linewidth': 1.2,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'legend.fontsize': 8,
    'legend.handlelength': 2.2,
})

# Curated palette (colorblind-friendly, print-safe)
PAL = {
    'blue':   '#2171b5',
    'orange': '#d94801',
    'green':  '#238b45',
    'red':    '#cb181d',
    'grey':   '#737373',
    'purple': '#6a51a3',
}

# ===========================================================================
#  Physical parameters (water-steam at 1 bar, dT = 1.25 K)
# ===========================================================================
rho_l   = 958.4
cp_l    = 4216.0
k_l     = 0.677
beta    = 4.06022
alpha_l = k_l / (rho_l * cp_l)
R0      = 5.0e-5
t0      = R0**2 / (4.0 * beta**2 * alpha_l)


def scriven_radius(t):
    """Analytical Scriven radius R(t) = 2 beta sqrt(alpha_l t)."""
    return 2.0 * beta * np.sqrt(alpha_l * t)


def load_bench_data(fpath):
    """Load bench-data file, handling NaN rows and time resets."""
    data = np.loadtxt(fpath)
    mask = ~np.isnan(data).any(axis=1)
    data = data[mask]
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
    return data


def load_bures_sato_data(fpath):
    """Load Bureš & Sato (2021) volume data and compute bubble radius."""
    times = []
    radii = []
    with open(fpath) as f:
        for line in f:
            parts = line.strip().split('=')
            if len(parts) == 2:
                vals = parts[1].split()
                if len(vals) == 3:
                    t = float(vals[0])
                    phisum = float(vals[2])
                    R = 2.0 * (3.0 * phisum / (4.0 * np.pi))**(1.0 / 3.0)
                    times.append(t)
                    radii.append(R)
    return np.array(times), np.array(radii)


# ===========================================================================
#  Main
# ===========================================================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Structured bench-data files
    struct_bench_files = {
        r'Struct. $75^3$':   "bench-data-struct-075.dat",
        r'Struct. $125^3$':  "bench-data-struct-125.dat",
    }

    struct_meshes = {}
    for label, fname in struct_bench_files.items():
        fpath = os.path.join(script_dir, fname)
        if os.path.exists(fpath):
            data = load_bench_data(fpath)
            struct_meshes[label] = {
                'time':   data[:, 0],
                'radius': data[:, 3],
            }
            print(f"  Loaded {label}: {data.shape[0]} steps")
        else:
            print(f"  WARNING: {fpath} not found")

    # Paths — override via environment variables, e.g.:
    #   export MERLIN_DIR=/path/to/Scriven-Merlin
    #   export DATA_DIR=/path/to/UnstructuredBoiling/Data
    #   export OUTPUT_DIR=/path/to/paper/pics
    merlin_dir = os.environ.get("MERLIN_DIR", "")
    if not merlin_dir:
        raise RuntimeError("MERLIN_DIR environment variable is not set. "
                           "Point it to the Scriven-Merlin data directory.")
    poly_front_files = {
        r'Poly. $75^3$':  "Scriven-Poly-75",
        r'Poly. $125^3$': "Scriven-Poly-125",
    }

    poly_meshes = {}
    for label, dname in poly_front_files.items():
        fpath = os.path.join(merlin_dir, dname, "front-radius.dat")
        if os.path.exists(fpath):
            data = np.loadtxt(fpath)
            poly_meshes[label] = {
                'time':   data[:, 0],
                'radius': data[:, 1],
            }
            print(f"  Loaded {label}: {data.shape[0]} steps")
        else:
            print(f"  WARNING: {fpath} not found")

    # Bureš & Sato (2021) reference data
    data_dir = os.environ.get("DATA_DIR", "")
    if not data_dir:
        raise RuntimeError("DATA_DIR environment variable is not set. "
                           "Point it to the UnstructuredBoiling/Data directory.")
    bures_sato_files = {
        r'Bureš & Sato, $h = 3.91~\mu$m':  "vol-cart2.out",
        r'Bureš & Sato, $h = 1.95~\mu$m':  "vol-cart4.out",
    }
    bures_sato = {}
    for label, fname in bures_sato_files.items():
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            t_bs, R_bs = load_bures_sato_data(fpath)
            R0_bs = R_bs[0]
            t0_bs = R0_bs**2 / (4.0 * beta**2 * alpha_l)
            bures_sato[label] = {
                'time': t_bs, 'radius': R_bs, 't0': t0_bs,
            }
            print(f"  Loaded {label}: {len(t_bs)} steps")
        else:
            print(f"  WARNING: {fpath} not found")

    # --- Figure ---
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))

    # Analytical
    t_anal = np.linspace(0, 1.5e-3, 500)
    R_anal = scriven_radius(t_anal)
    ax.plot(t_anal * 1e3, R_anal * 1e6, color='k', linestyle='-',
            linewidth=1.5, label='Analytical')

    # Structured (blue tones, dashed)
    struct_styles = [
        {'color': PAL['blue'],   'linestyle': '--',  'linewidth': 1.0},
        {'color': PAL['blue'],   'linestyle': '-',   'linewidth': 1.2},
    ]
    for i, (label, d) in enumerate(struct_meshes.items()):
        t_shifted = d['time'] + t0
        ax.plot(t_shifted * 1e3, d['radius'] * 1e6,
                label=label, **struct_styles[i])

    # Polyhedral (orange tones, dash-dot)
    poly_styles = [
        {'color': PAL['orange'], 'linestyle': '--',  'linewidth': 1.0},
        {'color': PAL['orange'], 'linestyle': '-',   'linewidth': 1.2},
    ]
    for i, (label, d) in enumerate(poly_meshes.items()):
        t_shifted = d['time'] + t0
        ax.plot(t_shifted * 1e3, d['radius'] * 1e6,
                label=label, **poly_styles[i])

    # Bureš & Sato (green, dotted)
    bs_styles = [
        {'color': PAL['green'],  'linestyle': ':',   'linewidth': 1.0},
        {'color': PAL['green'],  'linestyle': '-.',   'linewidth': 1.0},
    ]
    for i, (label, d) in enumerate(bures_sato.items()):
        t_shifted = d['time'] + d['t0']
        ax.plot(t_shifted * 1e3, d['radius'] * 1e6,
                label=label, **bs_styles[i])

    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'Bubble radius [$\mu$m]')
    ax.set_xlim([0, 1.5])
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    out_dir = os.environ.get("OUTPUT_DIR", script_dir)
    for ext in ['.png', '.pdf']:
        plt.savefig(os.path.join(out_dir,
                                 f"scriven_mass_conservation{ext}"),
                    bbox_inches='tight')
        plt.savefig(os.path.join(script_dir, f"scriven_radius{ext}"),
                    bbox_inches='tight')
    print(f"\n  Saved: scriven_mass_conservation and scriven_radius")
    plt.close()
