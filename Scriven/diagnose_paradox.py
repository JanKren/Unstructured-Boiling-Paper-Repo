#!/usr/bin/env python3
"""
Diagnose the overshoot paradox: compare effective mass transfer
from volume growth (bench-data) with the analytical Scriven prediction.

The key question: if the gradient is ~86% of analytical (from VTU),
why does the bubble grow FASTER than Scriven?
"""
import numpy as np
import matplotlib.pyplot as plt
import os

# Physical parameters
rho_l   = 958.4       # kg/m^3
rho_v   = 0.597       # kg/m^3
cp_l    = 4216.0       # J/(kg K)
k_l     = 0.677        # W/(m K)
L       = 2.257e6      # J/kg  (latent heat)
alpha_l = k_l / (rho_l * cp_l)
beta    = 4.06022      # Scriven growth constant
R0      = 5.0e-5       # initial radius [m]
t0      = R0**2 / (4.0 * beta**2 * alpha_l)  # virtual origin
dT_sup  = 1.25         # K (superheat)

def scriven_R(t_sim):
    """Scriven radius at simulation time."""
    return 2.0 * beta * np.sqrt(alpha_l * (t_sim + t0))

def scriven_dRdt(t_sim):
    """Scriven dR/dt at simulation time."""
    t_total = t_sim + t0
    return beta * np.sqrt(alpha_l / t_total)

def scriven_g_interface(t_sim):
    """Analytical interface gradient from Stefan condition."""
    return rho_v * L * scriven_dRdt(t_sim) / k_l

def load_bench_segment(fpath, seg_index=-1):
    """Load last segment from bench-data file."""
    with open(fpath) as f:
        lines = f.readlines()
    segments = []
    seg_start = 0
    t_prev = -1
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 4:
            continue
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
        vals = line.split()[:ncols]
        data[j] = [float(x) for x in vals]
    return data


# =====================================================================
#   Load bench-data
# =====================================================================
# Paths — override via environment variables, e.g.:
#   export SCRIVEN_DIR=/path/to/Scriven
scriven_dir = os.environ.get("SCRIVEN_DIR", "")
if not scriven_dir:
    raise RuntimeError("SCRIVEN_DIR environment variable is not set. "
                       "Point it to the T-Flows Scriven case directory.")
cases = {
    'Struct-75 front-mod (dt=1us)':
        os.path.join(scriven_dir, "Scriven-Struct-75-smallDt", "bench-data.dat"),
    'Struct-75 no-front-grad (dt=1us)':
        os.path.join(scriven_dir, "Scriven-Struct-75-noFrontGrad", "bench-data.dat"),
}

results = {}
for label, fpath in cases.items():
    if not os.path.exists(fpath):
        print(f"  WARNING: {fpath} not found")
        continue
    data = load_bench_segment(fpath, seg_index=-1)
    t = data[:, 0]
    area = data[:, 1]
    vol = data[:, 2]
    R_vol = data[:, 3]
    R_front = data[:, 4] if data.shape[1] > 4 else None

    # Compute dV/dt from central differences
    dVdt = np.gradient(vol, t)

    # Analytical values
    R_anal = scriven_R(t)
    dRdt_anal = scriven_dRdt(t)
    dVdt_anal = 4.0 * np.pi * R_anal**2 * dRdt_anal

    # Effective mass transfer: m_dot_total = dV/dt / (1/rho_v - 1/rho_l)
    vol_factor = 1.0/rho_v - 1.0/rho_l
    m_dot_total = dVdt / vol_factor
    m_dot_anal  = dVdt_anal / vol_factor

    # Effective gradient from mass transfer and surface area:
    # Q = m_dot * L = k_l * g_eff * A
    # g_eff = m_dot * L / (k_l * A)
    g_eff = m_dot_total * L / (k_l * area)
    g_anal_interface = scriven_g_interface(t)

    # Also compute from a perfect sphere area
    A_sphere = 4.0 * np.pi * R_vol**2
    g_eff_sphere = m_dot_total * L / (k_l * A_sphere)

    results[label] = {
        't': t, 'area': area, 'vol': vol, 'R_vol': R_vol,
        'R_front': R_front, 'R_anal': R_anal,
        'dVdt': dVdt, 'dVdt_anal': dVdt_anal,
        'g_eff': g_eff, 'g_eff_sphere': g_eff_sphere,
        'g_anal': g_anal_interface,
        'A_sphere': A_sphere,
    }
    print(f"  Loaded {label}: {data.shape[0]} pts")

# =====================================================================
#   Print diagnostic table
# =====================================================================
print("\n" + "="*80)
print("  OVERSHOOT PARADOX DIAGNOSTICS")
print("="*80)

for label, d in results.items():
    print(f"\n  --- {label} ---")

    # Sample at a few times
    for t_ms in [0.1, 0.5, 1.0, 1.5]:
        t_val = t_ms * 1e-3
        idx = np.argmin(np.abs(d['t'] - t_val))
        if abs(d['t'][idx] - t_val) > 0.05e-3:
            continue

        R_ratio = d['R_vol'][idx] / d['R_anal'][idx]
        dVdt_ratio = d['dVdt'][idx] / d['dVdt_anal'][idx]
        g_ratio = d['g_eff'][idx] / d['g_anal'][idx]
        g_ratio_sphere = d['g_eff_sphere'][idx] / d['g_anal'][idx]
        A_ratio = d['area'][idx] / d['A_sphere'][idx]

        print(f"  t = {t_ms:.1f} ms:")
        print(f"    R_vol/R_anal     = {R_ratio:.4f}  ({(R_ratio-1)*100:+.1f}%)")
        print(f"    dV/dt ratio      = {dVdt_ratio:.4f}  ({(dVdt_ratio-1)*100:+.1f}%)")
        print(f"    g_eff/g_anal     = {g_ratio:.4f}  ({(g_ratio-1)*100:+.1f}%) [using A_front]")
        print(f"    g_eff/g_anal     = {g_ratio_sphere:.4f}  ({(g_ratio_sphere-1)*100:+.1f}%) [using A_sphere]")
        print(f"    A_front/A_sphere = {A_ratio:.4f}  ({(A_ratio-1)*100:+.1f}%)")
        print(f"    g_anal           = {d['g_anal'][idx]:.0f} K/m")
        print(f"    g_eff            = {d['g_eff'][idx]:.0f} K/m")

# =====================================================================
#   Figure
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# Panel (a): R_vol / R_anal
ax = axes[0, 0]
for label, d in results.items():
    mask = d['t'] * 1e3 <= 1.5
    ax.plot(d['t'][mask]*1e3, d['R_vol'][mask]/d['R_anal'][mask], label=label)
ax.axhline(1.0, color='k', lw=0.8, ls='-', alpha=0.5)
ax.set_xlabel('Time [ms]')
ax.set_ylabel(r'$R_\mathrm{vol} / R_\mathrm{Scriven}$')
ax.set_title('(a) Volume-equivalent radius ratio')
ax.legend(fontsize=8)

# Panel (b): dV/dt ratio
ax = axes[0, 1]
for label, d in results.items():
    mask = d['t'] * 1e3 <= 1.5
    ratio = d['dVdt'][mask] / d['dVdt_anal'][mask]
    ax.plot(d['t'][mask]*1e3, ratio, label=label)
ax.axhline(1.0, color='k', lw=0.8, ls='-', alpha=0.5)
ax.set_xlabel('Time [ms]')
ax.set_ylabel(r'$(\mathrm{d}V/\mathrm{d}t)_\mathrm{num} / (\mathrm{d}V/\mathrm{d}t)_\mathrm{anal}$')
ax.set_title('(b) Volume growth rate ratio')
ax.legend(fontsize=8)

# Panel (c): Effective gradient ratio
ax = axes[1, 0]
for label, d in results.items():
    mask = d['t'] * 1e3 <= 1.5
    ax.plot(d['t'][mask]*1e3, d['g_eff'][mask]/d['g_anal'][mask],
            label=f'{label} (A_front)', ls='-')
    ax.plot(d['t'][mask]*1e3, d['g_eff_sphere'][mask]/d['g_anal'][mask],
            label=f'{label} (A_sphere)', ls='--')
ax.axhline(1.0, color='k', lw=0.8, ls='-', alpha=0.5)
ax.set_xlabel('Time [ms]')
ax.set_ylabel(r'$g_\mathrm{eff} / g_\mathrm{Scriven}$')
ax.set_title('(c) Effective gradient ratio')
ax.legend(fontsize=7)

# Panel (d): A_front / A_sphere
ax = axes[1, 1]
for label, d in results.items():
    mask = d['t'] * 1e3 <= 1.5
    ax.plot(d['t'][mask]*1e3, d['area'][mask]/d['A_sphere'][mask], label=label)
ax.axhline(1.0, color='k', lw=0.8, ls='-', alpha=0.5)
ax.set_xlabel('Time [ms]')
ax.set_ylabel(r'$A_\mathrm{front} / A_\mathrm{sphere}$')
ax.set_title('(d) Front surface area excess')
ax.legend(fontsize=8)

plt.tight_layout()
script_dir = os.path.dirname(os.path.abspath(__file__))
for ext in ['.png', '.pdf']:
    plt.savefig(os.path.join(script_dir, f"overshoot_paradox{ext}"),
                bbox_inches='tight', dpi=200)
plt.close()
print("\n  Saved: overshoot_paradox.png/.pdf")
