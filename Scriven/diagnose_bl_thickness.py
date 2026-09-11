#!/usr/bin/env python3
"""
Diagnose the Scriven thermal boundary layer thickness vs cell size.

For the Scriven self-similar solution, the effective BL thickness is:
    lambda = sqrt(alpha_l * t) / beta

This script computes lambda/h as a function of time and correlates with
the effective gradient ratio from the bench-data.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# Physical parameters
rho_l   = 958.4
rho_v   = 0.597
cp_l    = 4216.0
k_l     = 0.677
L       = 2.257e6
alpha_l = k_l / (rho_l * cp_l)
beta    = 4.06022
R0      = 5.0e-5
t0      = R0**2 / (4.0 * beta**2 * alpha_l)
h       = 300e-6 / 75  # cell size for 75^3 mesh [m] = 4 um

def scriven_R(t_sim):
    return 2.0 * beta * np.sqrt(alpha_l * (t_sim + t0))

def scriven_dRdt(t_sim):
    return beta * np.sqrt(alpha_l / (t_sim + t0))

def load_bench_segment(fpath, seg_index=-1):
    with open(fpath) as f:
        lines = f.readlines()
    segments, seg_start, t_prev = [], 0, -1
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 4: continue
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
    return data


# Load bench-data for Struct-75 front-modified (dt=1us)
# Paths — override via environment variables, e.g.:
#   export SCRIVEN_DIR=/path/to/Scriven
scriven_dir = os.environ.get("SCRIVEN_DIR", "")
if not scriven_dir:
    raise RuntimeError("SCRIVEN_DIR environment variable is not set. "
                       "Point it to the T-Flows Scriven case directory.")
fpath = os.path.join(scriven_dir, "Scriven-Struct-75-smallDt", "bench-data.dat")
data = load_bench_segment(fpath, seg_index=-1)
t_sim = data[:, 0]
area   = data[:, 1]
vol    = data[:, 2]
R_vol  = data[:, 3]

# Compute effective gradient ratio
dVdt = np.gradient(vol, t_sim)
R_anal = scriven_R(t_sim)
dRdt_anal = scriven_dRdt(t_sim)
dVdt_anal = 4.0 * np.pi * R_anal**2 * dRdt_anal
g_anal = rho_v * L * dRdt_anal / k_l

vol_factor = 1.0/rho_v - 1.0/rho_l
m_dot = dVdt / vol_factor
g_eff = m_dot * L / (k_l * area)

# BL thickness
t_total = t_sim + t0
lam = np.sqrt(alpha_l * t_total) / beta  # BL thickness [m]
lam_over_h = lam / h

# Smoothed growth rate ratio
from scipy.ndimage import uniform_filter1d
window = max(1, int(50))  # 50 time steps smoothing
g_ratio_smooth = uniform_filter1d(g_eff / g_anal, window)

print("="*70)
print("  BL THICKNESS vs CELL SIZE and GRADIENT RATIO")
print("="*70)
for t_ms in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5]:
    t_val = t_ms * 1e-3
    idx = np.argmin(np.abs(t_sim - t_val))
    if abs(t_sim[idx] - t_val) > 0.1e-3:
        continue
    print(f"  t = {t_ms:5.2f} ms: "
          f"lambda/h = {lam_over_h[idx]:.3f}, "
          f"g_eff/g_anal = {g_ratio_smooth[idx]:.3f}, "
          f"R/R_anal = {R_vol[idx]/R_anal[idx]:.4f}")

# ===== Figure =====
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# Top: lambda/h vs time
ax1.plot(t_sim * 1e3, lam_over_h, 'b-', lw=2)
ax1.axhline(1.0, color='r', ls='--', lw=1, label=r'$\lambda = h$')
ax1.set_ylabel(r'$\lambda / h$ (BL cells)')
ax1.set_title('Scriven thermal BL thickness / cell size')
ax1.legend()
ax1.set_xlim([0, 1.5])

# Bottom: effective gradient ratio vs time
ax2.plot(t_sim * 1e3, g_ratio_smooth, 'C0-', lw=2, label='Smoothed')
ax2.axhline(1.0, color='k', ls='-', lw=0.8, alpha=0.5)
ax2.set_xlabel('Time [ms]')
ax2.set_ylabel(r'$g_\mathrm{eff} / g_\mathrm{Scriven}$')
ax2.set_title('Effective gradient ratio')
ax2.legend()
ax2.set_xlim([0, 1.5])

plt.tight_layout()
script_dir = os.path.dirname(os.path.abspath(__file__))
for ext in ['.png', '.pdf']:
    plt.savefig(os.path.join(script_dir, f"bl_thickness_vs_gradient{ext}"),
                bbox_inches='tight', dpi=200)
plt.close()
print(f"\n  Saved: bl_thickness_vs_gradient.png/.pdf")

# ===== Correlation plot: g_eff/g_anal vs lambda/h =====
fig, ax = plt.subplots(figsize=(6, 5))
mask = (t_sim > 0.02e-3) & (t_sim * 1e3 <= 1.5)
sc = ax.scatter(lam_over_h[mask], g_ratio_smooth[mask],
                c=t_sim[mask]*1e3, cmap='viridis', s=3, alpha=0.7)
plt.colorbar(sc, ax=ax, label='Time [ms]')
ax.axhline(1.0, color='k', ls='-', lw=0.8, alpha=0.5)
ax.axvline(1.0, color='r', ls='--', lw=1, alpha=0.5)
ax.set_xlabel(r'$\lambda / h$')
ax.set_ylabel(r'$g_\mathrm{eff} / g_\mathrm{Scriven}$')
ax.set_title(r'Gradient accuracy vs BL resolution ($75^3$ hex)')
plt.tight_layout()
for ext in ['.png', '.pdf']:
    plt.savefig(os.path.join(script_dir, f"gradient_vs_bl_resolution{ext}"),
                bbox_inches='tight', dpi=200)
plt.close()
print(f"  Saved: gradient_vs_bl_resolution.png/.pdf")
