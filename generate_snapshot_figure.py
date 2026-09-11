#!/usr/bin/env python3
"""
Standalone script to regenerate only the annular_snapshot figure (Figure 17).
Extracted from generate_paper_figures_v2.py (PART 2 only).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.colors import PowerNorm, SymLogNorm
from pathlib import Path
import os
import sys

def printf(*args):
    print(*args)
    sys.stdout.flush()

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Paths — override via environment variables, e.g.:
#   export OUTPUT_DIR=/path/to/paper/pics
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR",
                                 os.path.dirname(os.path.abspath(__file__))))

R_INNER_DATA = 0.00943    # 9.43 mm - actual wall position
R_INNER_DISPLAY = 0.0095  # 9.50 mm - shifted for display
DELTA_INITIAL = 0.25e-3

ANNULAR_NPZ_FILES = [Path("Annular/annular_xplane_chunk_01_gap.npz")]
ANNULAR_NPZ_FILES += [Path(f"Annular/annular_xplane_chunk_{i:02d}.npz") for i in range(2, 21)]

def load_all_manifests(npz_files):
    tag_to_file, all_manifests = {}, {}
    for npz_path in npz_files:
        if not npz_path.exists(): continue
        with np.load(npz_path, allow_pickle=False) as z:
            manifest = json.loads(str(z["_MANIFEST_JSON"]))
        for tag in manifest:
            tag_to_file[tag] = npz_path
            all_manifests[tag] = manifest[tag]
    return tag_to_file, sorted(tag_to_file.keys(), key=lambda s: int(s)), all_manifests

def load_field(npz_path, tag, field):
    with np.load(npz_path, allow_pickle=False) as z:
        return z[f"{tag}__{field}"]

def interpolate_to_grid(r, y, values, r_grid, y_grid, method='linear'):
    points = np.column_stack([y, r])
    grid_y, grid_r = np.meshgrid(y_grid, r_grid)
    grid_values = griddata(points, values, (grid_y, grid_r), method=method)
    if np.any(np.isnan(grid_values)):
        grid_values_nn = griddata(points, values, (grid_y, grid_r), method='nearest')
        grid_values = np.where(np.isnan(grid_values), grid_values_nn, grid_values)
    return grid_values

# Load manifests
printf("Loading manifests...")
tag_to_file, all_tags, all_manifests = load_all_manifests(ANNULAR_NPZ_FILES)
printf(f"Total timesteps: {len(all_tags)}")

# Load single timestep
printf("Loading single timestep for contour plots...")
tag = '029841'  # Same timestep as original figure (from chunks up to 13)
printf(f"Using timestep: {tag}")
fields = all_manifests[tag]
npz_path = tag_to_file[tag]

vof_key = 'VOF' if 'VOF' in fields else 'VOF_Sharp'
vof = load_field(npz_path, tag, vof_key).flatten()
coords = load_field(npz_path, tag, f'RY_{vof_key}')
Uy = load_field(npz_path, tag, 'Uy').flatten()
mtr_data = load_field(npz_path, tag, 'Mass_Transfer_Rate').flatten()
mtr_coords = load_field(npz_path, tag, 'RY_Mass_Transfer_Rate')

r = coords[:, 0] * 1000  # mm
y = coords[:, 1] * 1000  # mm

r_grid = np.linspace(9.5, 11.5, 100)
y_grid = np.linspace(0, 52, 200)

# ============================================================================
# Generate 4-panel snapshot figure
# ============================================================================
printf("Generating snapshot figure...")
fig, axes = plt.subplots(3, 1, figsize=(8, 10))

# (a) VOF field
ax = axes[0]
vof_grid = interpolate_to_grid(r, y, vof, r_grid, y_grid)
vof_grid = gaussian_filter(vof_grid, sigma=1)
im = ax.contourf(y_grid, r_grid, vof_grid, levels=np.linspace(0, 1, 21), cmap='RdBu')
ax.contour(y_grid, r_grid, vof_grid, levels=[0.5], colors='black', linewidths=2)
plt.colorbar(im, ax=ax, label='VOF')
ax.axhline(R_INNER_DISPLAY * 1000, color='red', ls='-', lw=2, label='Wall')
ax.set_ylabel('Radial position $r$ [mm]')
ax.set_title('(a) Phase distribution ($\\alpha$)')
ax.legend(loc='upper right')
ax.tick_params(labelbottom=False)

# (b) Velocity field
ax = axes[1]
Uy_grid = interpolate_to_grid(r, y, Uy, r_grid, y_grid)
Uy_grid = gaussian_filter(Uy_grid, sigma=1)
Uy_grid = np.clip(Uy_grid, 0, 22)
im = ax.contourf(y_grid, r_grid, Uy_grid, levels=np.linspace(0, 21, 22), cmap='viridis', extend='max')
ax.contour(y_grid, r_grid, vof_grid, levels=[0.5], colors='white', linewidths=1.5)
plt.colorbar(im, ax=ax, label='$U_y$ [m/s]')
ax.axhline(R_INNER_DISPLAY * 1000, color='red', ls='-', lw=2)
ax.set_ylabel('Radial position $r$ [mm]')
ax.set_title('(b) Axial velocity field')
ax.tick_params(labelbottom=False)

ax.text(25, 11.2, '$U_{y,v} \\approx 18.5$ m/s', color='white', fontsize=15, ha='center',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
ax.annotate('$U_{y,l} \\approx 0.6$ m/s', xy=(25, 9.65), xytext=(25, 9.85),
            color='black', fontsize=15, ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# (c) Mass transfer rate
ax = axes[2]
r_mtr = mtr_coords[:, 0] * 1000
y_mtr = mtr_coords[:, 1] * 1000
mtr_plot = mtr_data * 1e9

interface_mask = (r_mtr > 9.5) & (r_mtr < 10.8) & np.isfinite(mtr_plot)
if interface_mask.sum() > 100:
    r_mtr_filt = r_mtr[interface_mask]
    y_mtr_filt = y_mtr[interface_mask]
    mtr_filt = mtr_plot[interface_mask]

    vmin = np.percentile(mtr_filt, 2)
    vmax = np.percentile(mtr_filt, 98)

    r_grid_mtr = np.linspace(9.5, 10.6, 60)
    y_grid_mtr = np.linspace(0, 52, 200)
    mtr_grid = interpolate_to_grid(r_mtr_filt, y_mtr_filt, mtr_filt, r_grid_mtr, y_grid_mtr)
    mtr_grid = gaussian_filter(mtr_grid, sigma=1)

    if vmin < 0 < vmax:
        linthresh = max(abs(vmin), abs(vmax)) * 0.05
        norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
        im = ax.contourf(y_grid_mtr, r_grid_mtr, mtr_grid, levels=30, cmap='RdBu_r', norm=norm)
    else:
        mtr_min = np.nanmin(mtr_grid)
        mtr_shifted = mtr_grid - mtr_min + 1e-10
        vmin_shifted = vmin - mtr_min + 1e-10
        vmax_shifted = vmax - mtr_min + 1e-10
        norm = PowerNorm(gamma=0.4, vmin=vmin_shifted, vmax=vmax_shifted)
        im = ax.contourf(y_grid_mtr, r_grid_mtr, mtr_shifted, levels=30, cmap='YlOrRd', norm=norm)
    plt.colorbar(im, ax=ax, label='MTR [ng/(m³·s)]')
    ax.contour(y_grid, r_grid, vof_grid, levels=[0.5], colors='black', linewidths=1.5)

ax.axhline(R_INNER_DISPLAY * 1000, color='red', ls='-', lw=2)
ax.set_xlabel('Axial position $y$ [mm]')
ax.set_ylabel('Radial position $r$ [mm]')
ax.set_title('(c) Mass transfer rate')
ax.set_ylim([9.5, 10.6])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'annular_snapshot.pdf')
plt.savefig(OUTPUT_DIR / 'annular_snapshot.png')
plt.close()
printf(f"Saved: {OUTPUT_DIR / 'annular_snapshot.pdf'}")
