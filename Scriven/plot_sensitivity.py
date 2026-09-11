#!/usr/bin/env python3
"""
Scriven bubble growth: 4-panel sensitivity figure.

Panels:
  (a) Polyhedral mesh convergence at dt = 2 us
  (b) Timestep sensitivity on Poly-75^3
  (c) Structured hex mesh: temporal convergence and overshoot
  (d) Ablation: front-modified vs standard LSQ gradient

Usage:
    cd Scriven
    python3 plot_sensitivity.py
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

# ===========================================================================
#  Physical parameters (water-steam at 1 bar, dT = 1.25 K)
# ===========================================================================
rho_l   = 958.4
cp_l    = 4216.0
k_l     = 0.677
beta    = 4.06022
alpha_l = k_l / (rho_l * cp_l)
R0      = 5.0e-5                        # initial bubble radius [m]
t0      = R0**2 / (4.0 * beta**2 * alpha_l)  # virtual origin [s]

T_MAX_MS = 1.5   # x-axis limit [ms]


def scriven_radius(t_sim):
    """Analytical Scriven radius at simulation time t_sim."""
    return 2.0 * beta * np.sqrt(alpha_l * (t_sim + t0))


def load_bench_segment(fpath, seg_index=-1, usecols=None):
    """Load a specific segment from a bench-data file with time resets.

    seg_index=-1 gives the last segment; 0 gives the first, etc.
    """
    with open(fpath) as f:
        lines = f.readlines()

    segments = []
    seg_start = 0
    t_prev = -1
    for i, line in enumerate(lines):
        parts = line.split()
        t = float(parts[0])
        if t < t_prev:
            segments.append((seg_start, i - 1))
            seg_start = i
        t_prev = t
    segments.append((seg_start, len(lines) - 1))

    s, e = segments[seg_index]
    seg_lines = lines[s:e + 1]

    ncols = len(seg_lines[0].split())
    data = np.zeros((len(seg_lines), ncols))
    for j, line in enumerate(seg_lines):
        data[j] = [float(x) for x in line.split()[:ncols]]

    if usecols is not None:
        data = data[:, usecols]
    return data


def load_front_radius(fpath):
    """Load front-radius.dat from Merlin.

    Columns: time  R_front  R_area  R_anal  err_front  err_area  ...
    """
    data = np.loadtxt(fpath, comments='#')
    return data


# ===========================================================================
#  Data paths — all data in Scriven/data/ (no external paths needed)
# ===========================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(script_dir, "data")
out_dir    = os.environ.get("OUTPUT_DIR", script_dir)

if not os.path.isdir(data_dir):
    raise RuntimeError(f"Data directory not found: {data_dir}\n"
                       f"Run the data extraction script first.")

# ===========================================================================
#  Panel (a): Polyhedral mesh convergence at dt = 2 us
# ===========================================================================
poly_meshes_a = {}
for label, mesh in [
    (r'$75^3$',  '75'),
    (r'$100^3$', '100'),
    (r'$125^3$', '125'),
    (r'$150^3$', '150'),
]:
    fpath = os.path.join(data_dir, f"poly-{mesh}-dt2us-front-radius.dat")
    if os.path.exists(fpath):
        fr = load_front_radius(fpath)
        poly_meshes_a[label] = {
            'time':    fr[:, 0],
            'R_front': fr[:, 1],
            'R_anal':  fr[:, 3],
        }
        print(f"  [a] Loaded {label}: {fr.shape[0]} pts, "
              f"t=[{fr[0,0]:.2e}, {fr[-1,0]:.2e}]")
    else:
        print(f"  [a] WARNING: {fpath} not found, skipping {label}")

# ===========================================================================
#  Panel (b): Timestep sensitivity on Poly-75^3
# ===========================================================================
poly_dt_b = {}

# dt = 1 us: 4-col bench-data (rad_volume), corrected to rad_front
# using the ratio from the dt=10us file which has both metrics.
fpath_1us  = os.path.join(data_dir, "poly-75-dt1us-bench.dat")
fpath_10us = os.path.join(data_dir, "poly-75-dt10us-bench.dat")
if os.path.exists(fpath_1us):
    data_1us = np.loadtxt(fpath_1us, comments='#')
    if os.path.exists(fpath_10us):
        data_10us = np.loadtxt(fpath_10us, comments='#')
        if data_10us.shape[1] >= 5:
            ratio_t = data_10us[:, 0]
            ratio_v = data_10us[:, 4] / data_10us[:, 3]  # rad_front / rad_volume
            correction = np.interp(data_1us[:, 0], ratio_t, ratio_v,
                                   left=ratio_v[0], right=ratio_v[-1])
            corrected_radius = data_1us[:, 3] * correction
            print(f"  [b] Applied rad_front/rad_volume correction "
                  f"(mean={correction.mean():.4f}) to dt=1us data")
        else:
            corrected_radius = data_1us[:, 3]
    else:
        corrected_radius = data_1us[:, 3]
    poly_dt_b[r'$\Delta t = 1~\mu$s'] = {
        'time': data_1us[:, 0], 'radius': corrected_radius}
    print(f"  [b] Loaded Poly-75 dt=1us: {data_1us.shape[0]} pts")

# dt = 2 us: front-radius (R_front)
fpath = os.path.join(data_dir, "poly-75-dt2us-front-radius.dat")
if os.path.exists(fpath):
    fr = load_front_radius(fpath)
    poly_dt_b[r'$\Delta t = 2~\mu$s'] = {
        'time': fr[:, 0], 'radius': fr[:, 1]}
    print(f"  [b] Loaded Poly-75 dt=2us: {fr.shape[0]} pts")

# dt = 10 us: 5-col bench-data (col 4 = rad_front)
if os.path.exists(fpath_10us):
    data = np.loadtxt(fpath_10us, comments='#')
    if data.shape[1] >= 5:
        poly_dt_b[r'$\Delta t = 10~\mu$s'] = {
            'time': data[:, 0], 'radius': data[:, 4]}
        print(f"  [b] Loaded Poly-75 dt=10us (rad_front): {data.shape[0]} pts")
    else:
        poly_dt_b[r'$\Delta t = 10~\mu$s'] = {
            'time': data[:, 0], 'radius': data[:, 3]}
        print(f"  [b] Loaded Poly-75 dt=10us (rad_volume): {data.shape[0]} pts")

# ===========================================================================
#  Panel (c): Structured hex overshoot
# ===========================================================================
struct_c = {}

# Struct-75, dt = 1 us
fpath = os.path.join(data_dir, "struct-75-dt1us-bench.dat")
if os.path.exists(fpath):
    data = np.loadtxt(fpath, comments='#')
    struct_c[r'$\Delta t = 1~\mu$s'] = {
        'time': data[:, 0], 'radius': data[:, 3]}
    print(f"  [c] Loaded Struct-75 dt=1us: {data.shape[0]} pts")

# Struct-75, dt = 10 us
fpath = os.path.join(data_dir, "struct-75-dt10us-bench.dat")
if os.path.exists(fpath):
    data = np.loadtxt(fpath, comments='#')
    struct_c[r'$\Delta t = 10~\mu$s'] = {
        'time': data[:, 0], 'radius': data[:, 3]}
    print(f"  [c] Loaded Struct-75 dt=10us: {data.shape[0]} pts")

# Poly-75 reference (front-radius, dt = 2 us, consistent with panel a)
poly_ref = None
fpath = os.path.join(data_dir, "poly-75-dt2us-front-radius.dat")
if os.path.exists(fpath):
    fr = load_front_radius(fpath)
    poly_ref = {'time': fr[:, 0], 'R_front': fr[:, 1], 'R_anal': fr[:, 3]}
    print(f"  [c/d] Loaded Poly-75 front-radius ref: {fr.shape[0]} pts")

# ===========================================================================
#  Panel (d): Ablation test
# ===========================================================================
ablation_d = {}

# With interface-modified gradient (same as panel c, dt = 1 us)
fpath = os.path.join(data_dir, "struct-75-dt1us-bench.dat")
if os.path.exists(fpath):
    data = np.loadtxt(fpath, comments='#')
    ablation_d[r'Interface-modified'] = {
        'time': data[:, 0], 'radius': data[:, 3]}

# Without interface-modified gradient (standard LSQ)
fpath = os.path.join(data_dir, "struct-75-nomod-bench.dat")
if os.path.exists(fpath):
    data = np.loadtxt(fpath, comments='#')
    ablation_d[r'Standard'] = {
        'time': data[:, 0], 'radius': data[:, 3]}
    print(f"  [d] Loaded Struct-75 noFrontGrad: {data.shape[0]} pts")


# ===========================================================================
#  Color palette and figure
# ===========================================================================
# Curated palette (colorblind-friendly, print-safe)
PAL = {
    'blue':   '#2171b5',
    'orange': '#d94801',
    'green':  '#238b45',
    'red':    '#cb181d',
    'grey':   '#737373',
    'purple': '#6a51a3',
}

fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5))
plt.subplots_adjust(hspace=0.38, wspace=0.35)

# Common formatting for all panels
for ax in axes.flat:
    ax.set_xlim([0, T_MAX_MS])
    ax.axhline(y=1.0, color='k', linewidth=0.5, linestyle='-', alpha=0.4)
    ax.set_xlabel(r'Time [ms]')
    ax.set_ylabel(r'$R\,/\,R_{\mathrm{Scriven}}$')
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# --- Panel (a): Polyhedral convergence ---
ax = axes[0, 0]
colors_a = [PAL['blue'], PAL['orange'], PAL['green'], PAL['red']]
styles_a = ['-', '--', '-.', ':']
for i, (label, d) in enumerate(poly_meshes_a.items()):
    R_norm = d['R_front'] / d['R_anal']
    mask = d['time'] * 1e3 <= T_MAX_MS
    ax.plot(d['time'][mask] * 1e3, R_norm[mask],
            color=colors_a[i], linestyle=styles_a[i], label=label)
ax.set_ylim([0.905, 1.005])
ax.legend(loc='lower right', ncol=2)
ax.text(0.03, 0.95, r'(a)', transform=ax.transAxes,
        fontsize=11, fontweight='bold', va='top')

# --- Panel (b): Timestep sensitivity ---
ax = axes[0, 1]
styles_b = [
    {'color': PAL['blue'],   'linestyle': '-'},
    {'color': PAL['orange'], 'linestyle': '--'},
    {'color': PAL['green'],  'linestyle': '-.'},
]
for i, (label, d) in enumerate(poly_dt_b.items()):
    R_anal = scriven_radius(d['time'])
    R_norm = d['radius'] / R_anal
    mask = d['time'] * 1e3 <= T_MAX_MS
    ax.plot(d['time'][mask] * 1e3, R_norm[mask],
            label=label, **styles_b[i])
ax.set_ylim([0.82, 1.01])
ax.legend(loc='lower right')
ax.text(0.03, 0.95, r'(b)', transform=ax.transAxes,
        fontsize=11, fontweight='bold', va='top')

# --- Panel (c): Structured overshoot ---
ax = axes[1, 0]
styles_c = [
    {'color': PAL['blue'],   'linestyle': '-'},
    {'color': PAL['orange'], 'linestyle': '--'},
]
for i, (label, d) in enumerate(struct_c.items()):
    R_anal = scriven_radius(d['time'])
    R_norm = d['radius'] / R_anal
    mask = d['time'] * 1e3 <= T_MAX_MS
    ax.plot(d['time'][mask] * 1e3, R_norm[mask],
            label=r'Struct. $75^3$, ' + label, **styles_c[i])
# Poly-75 reference (front-radius, dt = 2 us)
if poly_ref is not None:
    R_norm = poly_ref['R_front'] / poly_ref['R_anal']
    mask = poly_ref['time'] * 1e3 <= T_MAX_MS
    ax.plot(poly_ref['time'][mask] * 1e3, R_norm[mask],
            color=PAL['grey'], linestyle=':', linewidth=1.0,
            label=r'Poly. $75^3$, $\Delta t = 1~\mu$s')
ax.set_ylim([0.88, 1.05])
ax.legend(loc='lower right', fontsize=7.5)
ax.text(0.03, 0.95, r'(c)', transform=ax.transAxes,
        fontsize=11, fontweight='bold', va='top')

# --- Panel (d): Ablation test ---
ax = axes[1, 1]
styles_d = [
    {'color': PAL['blue'],  'linestyle': '-'},
    {'color': PAL['red'],   'linestyle': '--'},
]
for i, (label, d) in enumerate(ablation_d.items()):
    R_anal = scriven_radius(d['time'])
    R_norm = d['radius'] / R_anal
    mask = d['time'] * 1e3 <= T_MAX_MS
    ax.plot(d['time'][mask] * 1e3, R_norm[mask],
            label=label, **styles_d[i])
# Poly-75 reference (front-radius, dt = 2 us)
if poly_ref is not None:
    R_norm = poly_ref['R_front'] / poly_ref['R_anal']
    mask = poly_ref['time'] * 1e3 <= T_MAX_MS
    ax.plot(poly_ref['time'][mask] * 1e3, R_norm[mask],
            color=PAL['grey'], linestyle=':', linewidth=1.0,
            label=r'Poly. $75^3$ ref.')
ax.set_ylim([0.82, 1.05])
ax.legend(loc='center right', fontsize=7.5)
ax.text(0.03, 0.95, r'(d)', transform=ax.transAxes,
        fontsize=11, fontweight='bold', va='top')

# Save
for ext in ['.png', '.pdf']:
    fout = os.path.join(out_dir, f"scriven_paper_figure{ext}")
    plt.savefig(fout, bbox_inches='tight')
    print(f"  Saved: {fout}")
    fout2 = os.path.join(script_dir, f"scriven_sensitivity{ext}")
    plt.savefig(fout2, bbox_inches='tight')

plt.close()
print("\nDone.")
