#!/usr/bin/env python3
"""
Vectorized quantitative figure generation - uses np.digitize for fast binning
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
import os
import sys

def printf(*args):
    print(*args)
    sys.stdout.flush()

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Paths — override via environment variables, e.g.:
#   export OUTPUT_DIR=/path/to/paper/pics
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR",
                                 os.path.dirname(os.path.abspath(__file__))))
R_INNER = 0.00943
DT = 1e-6
RHO_L, RHO_G = 943.9, 1.076
SIGMA = 0.0553
G = 9.81

ANNULAR_NPZ_FILES = [Path("Annular/annular_xplane_chunk_01_gap.npz")]
ANNULAR_NPZ_FILES += [Path(f"Annular/annular_xplane_chunk_{i:02d}.npz") for i in range(2, 21)]

# ============================================================================
# Load interface data
# ============================================================================
printf("Loading interface data...")
hov = np.load('publication_figures/hovmoller_data_extended.npz')
R_interface = hov['R_interface']
y_bins = hov['y_bins']
time_ms = hov['time_ms']
tags = list(hov['tags'])

n_t, n_y = R_interface.shape
delta = R_interface - R_INNER
R_mean_per_t = np.nanmean(R_interface, axis=1, keepdims=True)
eta = R_interface - R_mean_per_t

printf(f"Shape: {n_t} timesteps × {n_y} y-bins")

tag_to_idx = {tag: i for i, tag in enumerate(tags)}

# ============================================================================
# Vectorized MTR extraction
# ============================================================================
printf("\nExtracting MTR (vectorized)...")

# Convert y_bins from mm to meters (hovmoller_data.npz stores mm)
y_bins_m = y_bins * 1e-3

# Pre-compute bin edges for np.digitize (in meters)
bin_edges = np.concatenate([[y_bins_m[0] - 0.5*(y_bins_m[1]-y_bins_m[0])],
                            0.5*(y_bins_m[:-1] + y_bins_m[1:]),
                            [y_bins_m[-1] + 0.5*(y_bins_m[-1]-y_bins_m[-2])]])

mtr_mean = np.full((n_t, n_y), np.nan)
mtr_count = np.zeros((n_t, n_y), dtype=int)

for npz_path in ANNULAR_NPZ_FILES:
    if not npz_path.exists():
        continue

    printf(f"  Processing {npz_path.name}...")

    with np.load(npz_path, allow_pickle=False) as z:
        manifest = json.loads(str(z["_MANIFEST_JSON"]))
        file_tags = [t for t in manifest if t in tag_to_idx]

        for tag in file_tags:
            n = tag_to_idx[tag]
            fields = manifest[tag]
            if 'Mass_Transfer_Rate' not in fields:
                continue

            mtr_data = z[f"{tag}__Mass_Transfer_Rate"].flatten()
            mtr_coords = z[f"{tag}__RY_Mass_Transfer_Rate"]

            y_pts = mtr_coords[:, 1]
            r_pts = mtr_coords[:, 0]

            # Filter interface region AND non-zero MTR only!
            mtr_abs = np.abs(mtr_data)
            interface_mask = (r_pts > 0.0094) & (r_pts < 0.0105) & (mtr_abs > 1e-15)
            if interface_mask.sum() < 5:
                continue

            y_int = y_pts[interface_mask]
            mtr_int = mtr_abs[interface_mask]

            # Vectorized binning with np.bincount
            bin_idx = np.digitize(y_int, bin_edges) - 1
            valid_bins = (bin_idx >= 0) & (bin_idx < n_y)
            bin_idx_v = bin_idx[valid_bins]
            mtr_v = mtr_int[valid_bins]

            # Sum and count per bin using bincount
            counts = np.bincount(bin_idx_v, minlength=n_y)
            sums = np.bincount(bin_idx_v, weights=mtr_v, minlength=n_y)

            # Compute mean where count > 0 (we already filtered non-zero)
            good = counts > 0
            mtr_mean[n, good] = sums[good] / counts[good]
            mtr_count[n, :] = counts

printf(f"MTR mean: {np.nanmean(mtr_mean):.3e} kg/(m³·s)")
printf(f"Coverage: {(~np.isnan(mtr_mean)).sum() / mtr_mean.size * 100:.1f}%")

# Remove outlier timesteps (MTR > 1e-4 is clearly anomalous)
outlier_ts = np.any(mtr_mean > 1e-4, axis=1)
printf(f"Removing {outlier_ts.sum()} outlier timestep(s)")
mtr_mean[outlier_ts, :] = np.nan

# ============================================================================
# Generate Figure
# ============================================================================
printf("\nGenerating figure...")
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

delta_mm = delta.flatten() * 1000
eta_mm = eta.flatten() * 1000
mtr_flat = mtr_mean.flatten()

valid = (~np.isnan(delta_mm) & ~np.isnan(mtr_flat) &
         (delta_mm > 0.01) & (delta_mm < 1.0) & (mtr_flat > 0))
printf(f"Valid points: {valid.sum()}")

# Normalize MTR to mean for readable axis values
mtr_ref = np.nanmean(mtr_flat[valid])
printf(f"MTR reference (mean): {mtr_ref:.3e} kg/(m³·s)")

# (a) Film thickness vs MTR
ax = axes[0]
if valid.sum() > 100:
    n_sample = min(15000, valid.sum())
    idx = np.random.choice(np.where(valid)[0], n_sample, replace=False)
    mtr_norm = mtr_flat[idx] / mtr_ref  # Normalized to mean

    ax.scatter(delta_mm[idx], mtr_norm, s=1, alpha=0.15, c='steelblue', rasterized=True)

    # Binned mean
    bins = np.linspace(0.03, 0.55, 15)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_means = []
    for i in range(len(bins)-1):
        mask = valid & (delta_mm >= bins[i]) & (delta_mm < bins[i+1])
        if mask.sum() > 50:
            bin_means.append(np.mean(mtr_flat[mask]) / mtr_ref)
        else:
            bin_means.append(np.nan)

    ax.plot(bin_centers, bin_means, 'r-o', lw=2.5, markersize=5,
            label='Binned mean', zorder=5)

    # 1/δ theory curve anchored to data
    ref_idx = 3  # Anchor at a thick-film bin
    if not np.isnan(bin_means[ref_idx]):
        delta_theory = np.linspace(0.03, 0.55, 100)
        mtr_theory = bin_means[ref_idx] * (bin_centers[ref_idx] / delta_theory)
        ax.plot(delta_theory, mtr_theory, 'k--', lw=1.5, alpha=0.6,
                label='$\\propto 1/\\delta$')

    rho, _ = pearsonr(delta_mm[valid], mtr_flat[valid])
    ax.text(0.95, 0.78, f'$\\rho = {rho:.2f}$', transform=ax.transAxes,
            ha='right', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.set_xlabel('Film thickness $\\delta$ [mm]')
ax.set_ylabel('MTR / $\\langle$MTR$\\rangle$')
ax.set_title('(a) Film thickness vs mass transfer rate')
ax.legend(loc='upper right')
ax.set_xlim([0, 0.6])
ax.set_ylim([0, np.nanmax(bin_means) * 1.3 if bin_means else 5])

# (b) MTR by wave phase
ax = axes[1]
valid_eta = ~np.isnan(eta_mm) & ~np.isnan(mtr_flat) & (mtr_flat > 0)
eta_v = eta_mm[valid_eta]
mtr_v = mtr_flat[valid_eta] / mtr_ref  # Normalized

sigma_eta = np.nanstd(eta_v)
trough_mask = eta_v < -sigma_eta
crest_mask = eta_v > sigma_eta
neutral_mask = np.abs(eta_v) <= sigma_eta

printf(f"Phase counts: trough={trough_mask.sum()}, neutral={neutral_mask.sum()}, crest={crest_mask.sum()}")

categories = ['Trough\n(thin film)', 'Neutral', 'Crest\n(thick film)']
means = [np.mean(mtr_v[trough_mask]), np.mean(mtr_v[neutral_mask]), np.mean(mtr_v[crest_mask])]
stds = [np.std(mtr_v[trough_mask])/np.sqrt(trough_mask.sum()),
        np.std(mtr_v[neutral_mask])/np.sqrt(neutral_mask.sum()),
        np.std(mtr_v[crest_mask])/np.sqrt(crest_mask.sum())]
colors = ['#e74c3c', '#f39c12', '#3498db']

bars = ax.bar(categories, means, yerr=stds, color=colors, edgecolor='black', capsize=5, alpha=0.8)

for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(stds)*0.3,
            f'{mean:.2f}', ha='center', va='bottom', fontsize=10)

if means[2] > 0:
    ratio = means[0] / means[2]
    ax.text(0.5, 0.95, f'Trough/Crest = {ratio:.1f}', transform=ax.transAxes,
            ha='center', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.set_ylabel('MTR / $\\langle$MTR$\\rangle$')
ax.set_title('(b) MTR by wave phase')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'annular_quantitative.pdf')
plt.savefig(OUTPUT_DIR / 'annular_quantitative.png')
printf(f"\nSaved: {OUTPUT_DIR / 'annular_quantitative.pdf'}")
plt.close()

np.savez('publication_figures/mtr_data.npz',
         mtr_mean=mtr_mean, y_bins=y_bins, tags=tags,
         delta=delta, eta=eta, time_ms=time_ms)
printf("Saved: publication_figures/mtr_data.npz")
