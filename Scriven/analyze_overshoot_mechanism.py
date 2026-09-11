#!/usr/bin/env python3
"""
Systematic analysis of the structured-mesh overshoot mechanism.

Investigates whether the Scriven overshoot on structured meshes is caused by:
  (a) Net gradient overestimation (angular-mean gradient > analytical), or
  (b) Coherent anisotropy + shape feedback (gradient unbiased on average,
      but coherent m=4 pattern drives lobes that amplify mass transfer).

Diagnostics:
  1. R_vol / R_anal  over time  — does the overshoot accelerate?
  2. R_front / R_vol  over time — does the bubble become more non-spherical?
  3. A_front / A_sphere(R_vol)  — surface area excess
  4. d(R_vol)/dt  vs  d(R_anal)/dt — instantaneous growth rate ratio
  5. Comparison: front-modified vs no-front-grad vs polyhedral

Usage:
    cd Scriven
    python3 analyze_overshoot_mechanism.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.figsize': (14, 10),
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
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

T_MAX_MS = 1.5


def scriven_radius(t_sim):
    """Analytical Scriven radius at simulation time t_sim."""
    return 2.0 * beta * np.sqrt(alpha_l * (t_sim + t0))


def scriven_area(t_sim):
    """Analytical surface area of Scriven bubble."""
    R = scriven_radius(t_sim)
    return 4.0 * np.pi * R**2


def scriven_dRdt(t_sim):
    """Analytical dR/dt for Scriven bubble."""
    return beta * np.sqrt(alpha_l / (t_sim + t0))


def sphere_area_from_vol(vol):
    """Surface area of a sphere with the given volume."""
    R = (3.0 * vol / (4.0 * np.pi))**(1.0 / 3.0)
    return 4.0 * np.pi * R**2


def load_bench_segment(fpath, seg_index=-1):
    """Load a specific segment from a bench-data file with time resets."""
    with open(fpath) as f:
        lines = f.readlines()

    segments = []
    seg_start = 0
    t_prev = -1
    for i, line in enumerate(lines):
        t = float(line.split()[0])
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
        vals = line.split()[:ncols]
        data[j] = [float(x) for x in vals]
    return data


# ===========================================================================
#  Load data
# ===========================================================================
# Paths — override via environment variables, e.g.:
#   export SCRIVEN_DIR=/path/to/Scriven
#   export MERLIN_DIR=/path/to/Scriven-Merlin
#   export OUTPUT_DIR=/path/to/paper/pics
scriven_dir = os.environ.get("SCRIVEN_DIR", "")
if not scriven_dir:
    raise RuntimeError("SCRIVEN_DIR environment variable is not set. "
                       "Point it to the T-Flows Scriven case directory.")
merlin_dir  = os.environ.get("MERLIN_DIR", "")
if not merlin_dir:
    raise RuntimeError("MERLIN_DIR environment variable is not set. "
                       "Point it to the Scriven-Merlin data directory.")
script_dir  = os.path.dirname(os.path.abspath(__file__))
out_dir     = os.environ.get("OUTPUT_DIR", script_dir)

cases = {}

# --- Struct-75, dt=1us, front-modified gradient (5 cols) ---
fpath = os.path.join(scriven_dir, "Scriven-Struct-75-smallDt", "bench-data.dat")
if os.path.exists(fpath):
    data = np.loadtxt(fpath)
    cases['Struct-75, front-mod'] = {
        'time': data[:, 0],
        'area': data[:, 1],
        'vol':  data[:, 2],
        'R_vol': data[:, 3],
        'R_front': data[:, 4],
        'dt': 1e-6,
        'color': 'C0',
        'ls': '-',
    }
    print(f"  Loaded Struct-75 front-mod: {data.shape[0]} pts, "
          f"t=[{data[0,0]:.2e}, {data[-1,0]:.2e}]")

# --- Struct-75, dt=1us, NO front-modified gradient (5 cols) ---
fpath = os.path.join(scriven_dir, "Scriven-Struct-75-noFrontGrad", "bench-data.dat")
if os.path.exists(fpath):
    data = np.loadtxt(fpath)
    cases['Struct-75, no front'] = {
        'time': data[:, 0],
        'area': data[:, 1],
        'vol':  data[:, 2],
        'R_vol': data[:, 3],
        'R_front': data[:, 4],
        'dt': 1e-6,
        'color': 'C3',
        'ls': '--',
    }
    print(f"  Loaded Struct-75 no-front: {data.shape[0]} pts")

# --- Struct-75, dt=10us (5 cols) ---
fpath = os.path.join(scriven_dir, "Scriven-Struct-75-largeDt", "bench-data.dat")
if os.path.exists(fpath):
    data = np.loadtxt(fpath)
    cases['Struct-75, dt=10us'] = {
        'time': data[:, 0],
        'area': data[:, 1],
        'vol':  data[:, 2],
        'R_vol': data[:, 3],
        'R_front': data[:, 4],
        'dt': 10e-6,
        'color': 'C4',
        'ls': ':',
    }
    print(f"  Loaded Struct-75 dt=10us: {data.shape[0]} pts")

# --- Poly-75, dt=1us (from bench-data.dat, segment index 2 = 5-col rerun) ---
fpath = os.path.join(scriven_dir, "Scriven-Poly-75", "bench-data.dat")
if os.path.exists(fpath):
    data = load_bench_segment(fpath, seg_index=2)  # 5-col rerun (1650 rows)
    if data.shape[1] >= 5:
        cases['Poly-75, dt=1us'] = {
            'time': data[:, 0],
            'area': data[:, 1],
            'vol':  data[:, 2],
            'R_vol': data[:, 3],
            'R_front': data[:, 4],
            'dt': 1e-6,
            'color': 'C1',
            'ls': '-',
        }
        print(f"  Loaded Poly-75 dt=1us: {data.shape[0]} pts, "
              f"t=[{data[0,0]:.2e}, {data[-1,0]:.2e}], ncols={data.shape[1]}")

# --- Poly-75, dt=2us (from Merlin front-radius.dat) ---
fpath = os.path.join(merlin_dir, "Scriven-Poly-75", "front-radius.dat")
if os.path.exists(fpath):
    data = np.loadtxt(fpath, comments='#')
    # Cols: time, R_front, R_area, R_anal, err_front, err_area, n_elems, surface_area
    cases['Poly-75, dt=2us (Merlin)'] = {
        'time': data[:, 0],
        'area': data[:, 7],    # surface area
        'R_front': data[:, 1],
        'R_area': data[:, 2],
        'R_anal_merlin': data[:, 3],
        'dt': 2e-6,
        'color': 'C2',
        'ls': '-.',
    }
    print(f"  Loaded Poly-75 dt=2us Merlin: {data.shape[0]} pts, "
          f"t=[{data[0,0]:.2e}, {data[-1,0]:.2e}]")


# ===========================================================================
#  Compute derived quantities
# ===========================================================================
for name, c in cases.items():
    t = c['time']
    R_anal = scriven_radius(t)
    c['R_anal'] = R_anal

    if 'R_vol' in c:
        c['R_vol_norm'] = c['R_vol'] / R_anal
    if 'R_front' in c:
        c['R_front_norm'] = c['R_front'] / R_anal
    if 'R_vol' in c and 'R_front' in c:
        c['R_front_over_R_vol'] = c['R_front'] / c['R_vol']
    if 'area' in c and 'vol' in c:
        A_sphere = sphere_area_from_vol(c['vol'])
        c['area_excess'] = c['area'] / A_sphere
    elif 'area' in c and 'R_front' in c:
        # For Merlin data, approximate volume from R_front
        A_sphere = 4.0 * np.pi * c['R_front']**2
        c['area_excess'] = c['area'] / A_sphere

    # Instantaneous growth rate (numerical derivative)
    if 'R_vol' in c:
        dR = np.gradient(c['R_vol'], t)
        dR_anal = scriven_dRdt(t)
        c['growth_rate_ratio'] = dR / dR_anal


# ===========================================================================
#  Figure 1: 4-panel diagnostic
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel (a): R_vol / R_anal over time
ax = axes[0, 0]
ax.set_title(r'(a) Volume-equivalent radius ratio $R_\mathrm{vol}/R_\mathrm{Scriven}$',
             fontsize=11)
for name, c in cases.items():
    if 'R_vol_norm' in c:
        mask = c['time'] * 1e3 <= T_MAX_MS
        ax.plot(c['time'][mask] * 1e3, c['R_vol_norm'][mask],
                color=c['color'], linestyle=c['ls'], label=name)
ax.axhline(y=1.0, color='k', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Time [ms]')
ax.set_ylabel(r'$R_\mathrm{vol} / R_\mathrm{Scriven}$')
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim([0, T_MAX_MS])

# Panel (b): R_front / R_vol (non-sphericity)
ax = axes[0, 1]
ax.set_title(r'(b) Non-sphericity: $R_\mathrm{front}/R_\mathrm{vol}$', fontsize=11)
for name, c in cases.items():
    if 'R_front_over_R_vol' in c:
        mask = c['time'] * 1e3 <= T_MAX_MS
        ax.plot(c['time'][mask] * 1e3, c['R_front_over_R_vol'][mask],
                color=c['color'], linestyle=c['ls'], label=name)
ax.axhline(y=1.0, color='k', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Time [ms]')
ax.set_ylabel(r'$R_\mathrm{front} / R_\mathrm{vol}$')
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim([0, T_MAX_MS])

# Panel (c): Surface area excess A / A_sphere(R_vol)
ax = axes[1, 0]
ax.set_title(r'(c) Surface area excess: $A_\mathrm{front}/A_\mathrm{sphere}(R_\mathrm{vol})$',
             fontsize=11)
for name, c in cases.items():
    if 'area_excess' in c:
        mask = c['time'] * 1e3 <= T_MAX_MS
        ax.plot(c['time'][mask] * 1e3, c['area_excess'][mask],
                color=c['color'], linestyle=c['ls'], label=name)
ax.axhline(y=1.0, color='k', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Time [ms]')
ax.set_ylabel(r'$A_\mathrm{front} / A_\mathrm{sphere}$')
ax.legend(fontsize=8, loc='upper left')
ax.set_xlim([0, T_MAX_MS])

# Panel (d): Instantaneous growth rate ratio
ax = axes[1, 1]
ax.set_title(r'(d) Growth rate ratio $\dot{R}_\mathrm{vol}/\dot{R}_\mathrm{Scriven}$',
             fontsize=11)
for name, c in cases.items():
    if 'growth_rate_ratio' in c:
        mask = (c['time'] * 1e3 <= T_MAX_MS) & (c['time'] > 20 * c['dt'])
        # Smooth with running average (10-point window)
        ratio = c['growth_rate_ratio'][mask]
        t_ms = c['time'][mask] * 1e3
        if len(ratio) > 20:
            window = min(20, len(ratio) // 5)
            ratio_smooth = np.convolve(ratio, np.ones(window)/window, mode='valid')
            t_smooth = t_ms[:len(ratio_smooth)]
            ax.plot(t_smooth, ratio_smooth,
                    color=c['color'], linestyle=c['ls'], label=name)
        else:
            ax.plot(t_ms, ratio,
                    color=c['color'], linestyle=c['ls'], label=name)
ax.axhline(y=1.0, color='k', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Time [ms]')
ax.set_ylabel(r'$\dot{R}_\mathrm{vol} / \dot{R}_\mathrm{Scriven}$')
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim([0, T_MAX_MS])

plt.tight_layout()
for ext in ['.png', '.pdf']:
    plt.savefig(os.path.join(script_dir, f"overshoot_mechanism{ext}"),
                bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, f"overshoot_mechanism{ext}"),
                bbox_inches='tight')
print("\n  Saved: overshoot_mechanism.png/.pdf")
plt.close()


# ===========================================================================
#  Print quantitative summary at key time points
# ===========================================================================
print("\n" + "="*80)
print("  QUANTITATIVE SUMMARY")
print("="*80)

for t_target_ms in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
    t_target = t_target_ms * 1e-3
    print(f"\n  --- t = {t_target_ms:.2f} ms ---")
    R_anal_target = scriven_radius(t_target)
    A_anal_target = scriven_area(t_target)
    print(f"  R_anal = {R_anal_target*1e6:.1f} um, "
          f"A_anal = {A_anal_target*1e9:.2f} x10^-9 m^2")

    for name, c in cases.items():
        idx = np.argmin(np.abs(c['time'] - t_target))
        if abs(c['time'][idx] - t_target) > 2 * c['dt']:
            continue

        line = f"  {name:30s}: "
        if 'R_vol_norm' in c:
            line += f"R_vol/R_anal={c['R_vol_norm'][idx]:.4f} "
            line += f"({(c['R_vol_norm'][idx]-1)*100:+.1f}%) "
        if 'R_front_norm' in c:
            line += f"R_front/R_anal={c['R_front_norm'][idx]:.4f} "
            line += f"({(c['R_front_norm'][idx]-1)*100:+.1f}%) "
        if 'R_front_over_R_vol' in c:
            line += f"R_fr/R_vol={c['R_front_over_R_vol'][idx]:.4f} "
        if 'area_excess' in c:
            line += f"A/A_sph={c['area_excess'][idx]:.4f} "
        print(line)


# ===========================================================================
#  Key diagnostic: Does overshoot ACCELERATE?
# ===========================================================================
print("\n" + "="*80)
print("  ACCELERATION DIAGNOSTIC")
print("  If overshoot accelerates, d(R_vol/R_anal)/dt > 0 and increasing.")
print("="*80)

for name in ['Struct-75, front-mod', 'Poly-75, dt=1us']:
    if name not in cases:
        continue
    c = cases[name]
    if 'R_vol_norm' not in c:
        continue

    t = c['time']
    ratio = c['R_vol_norm']
    mask = (t * 1e3 >= 0.2) & (t * 1e3 <= T_MAX_MS)
    t_m = t[mask]
    r_m = ratio[mask]

    # Fit R_vol/R_anal = a + b*t + c*t^2 (quadratic)
    # If c > 0, the overshoot accelerates (positive feedback)
    from numpy.polynomial import polynomial as P
    coeffs = P.polyfit(t_m * 1e3, r_m, 2)  # fit in ms
    # coeffs are [c0, c1, c2] where f(x) = c0 + c1*x + c2*x^2
    print(f"\n  {name}:")
    print(f"    Quadratic fit: R/R_anal = {coeffs[0]:.6f} "
          f"+ {coeffs[1]:.6f}*t_ms + {coeffs[2]:.6f}*t_ms^2")
    if coeffs[2] > 0:
        print(f"    >>> ACCELERATING (c2 = {coeffs[2]:.6f} > 0)")
        print(f"    >>> Consistent with positive feedback mechanism")
    else:
        print(f"    >>> DECELERATING (c2 = {coeffs[2]:.6f} < 0)")
        print(f"    >>> Overshoot is linear or saturating, not accelerating")

    # Also report the slope d(R/R_anal)/dt at early and late times
    # Using finite differences on the smoothed ratio
    window = 50 if len(r_m) > 100 else max(5, len(r_m) // 10)
    r_smooth = np.convolve(r_m, np.ones(window)/window, mode='valid')
    t_smooth = t_m[:len(r_smooth)]
    dr_dt = np.gradient(r_smooth, t_smooth * 1e3)  # per ms

    n4 = len(dr_dt) // 4
    slope_early = np.mean(dr_dt[:n4])
    slope_late  = np.mean(dr_dt[-n4:])
    print(f"    Slope d(R/R_anal)/dt: early={slope_early:.4f}/ms, "
          f"late={slope_late:.4f}/ms")
    if abs(slope_late) > abs(slope_early) and slope_late > 0:
        print(f"    >>> Late slope > early slope: POSITIVE FEEDBACK")
    else:
        print(f"    >>> No clear acceleration in slope")

print("\nDone.")
