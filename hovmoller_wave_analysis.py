#!/usr/bin/env python3
"""
Hovmöller Diagram and Wave Velocity Analysis for Annular Flow.

Correctly uses:
- DT = 1e-6 s (1 μs between output timesteps)
- Radial filter for film region (r = 9.4-10.5 mm)
- All available NPZ chunk files
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.ndimage import gaussian_filter

# Import phase velocity module from cfd-postprocessing
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent / 'cfd-postprocessing'))
from PhaseVelocity import estimate_phase_speed_from_hovmoller

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

OUTPUT_DIR = Path("publication_figures")
ANNULAR_DIR = Path("Annular")

# CORRECT TIME PARAMETERS
# Simulation dt = 0.1 μs, output every 10 steps → DT = 1 μs between outputs
DT = 1e-6  # 1 μs between consecutive timesteps

# Spatial resolution
Y_MIN, Y_MAX = 0.0, 0.052  # meters
DY = 2.6e-4  # ~0.26 mm

# VOF threshold and radial filter
VOF_LEVEL = 0.5
R_MIN_FILM = 0.0094  # 9.4 mm - lower bound of film region
R_MAX_FILM = 0.0105  # 10.5 mm - upper bound of film region
R_INNER = 0.00943   # Wall radius (9.43 mm)


def get_all_npz_files():
    """Find all NPZ chunk files in Annular directory, including gap chunks."""
    files = sorted(ANNULAR_DIR.glob("annular_xplane_chunk_*.npz"))
    # Also include gap chunk if it exists
    gap_chunk = ANNULAR_DIR / "annular_xplane_chunk_01_gap.npz"
    if gap_chunk.exists() and gap_chunk not in files:
        files.append(gap_chunk)
        files = sorted(files)
    return files


def load_all_manifests(npz_files):
    """Load manifests from all NPZ files and build tag-to-file mapping."""
    tag_to_file = {}
    all_manifests = {}

    for npz_path in npz_files:
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=False) as z:
            manifest = json.loads(str(z["_MANIFEST_JSON"]))
        for tag in manifest:
            tag_to_file[tag] = npz_path
            all_manifests[tag] = manifest[tag]

    all_tags = sorted(tag_to_file.keys(), key=lambda s: int(s))
    return tag_to_file, all_tags, all_manifests


def load_field(npz_path, tag, field):
    """Load a single field from an NPZ file."""
    with np.load(npz_path, allow_pickle=False) as z:
        return z[f"{tag}__{field}"]


def find_interface_filtered(r_coords, vof, vof_level=0.5):
    """
    Find interface position with radial filter.
    Uses linear interpolation to find VOF=0.5 crossing.
    Only considers crossings in the film region (R_MIN_FILM to R_MAX_FILM).
    """
    if len(r_coords) < 2:
        return np.nan

    order = np.argsort(r_coords)
    r_sorted = r_coords[order]
    vof_sorted = vof[order]

    # Find all crossings
    s = vof_sorted - vof_level
    crossings = np.where(s[:-1] * s[1:] < 0)[0]

    # Filter to film region and use first valid crossing
    for j in crossings:
        denom = vof_sorted[j+1] - vof_sorted[j]
        if abs(denom) < 1e-10:
            continue
        t = (vof_level - vof_sorted[j]) / denom
        r_cross = r_sorted[j] + t * (r_sorted[j+1] - r_sorted[j])

        if R_MIN_FILM <= r_cross <= R_MAX_FILM:
            return r_cross

    return np.nan


def extract_interface_profile(coords, vof, y_bins):
    """Extract R(y) interface profile for a single timestep."""
    interface_r = np.full(len(y_bins), np.nan)

    for i, y_c in enumerate(y_bins):
        y_mask = np.abs(coords[:, 1] - y_c) < DY / 2
        if y_mask.sum() < 10:
            continue

        r_local = coords[y_mask, 0]
        vof_local = vof[y_mask]

        interface_r[i] = find_interface_filtered(r_local, vof_local)

    return interface_r


def main(suffix=''):
    print("="*60)
    print("HOVMÖLLER DIAGRAM AND WAVE VELOCITY ANALYSIS")
    print("="*60)
    if suffix:
        print(f"Output suffix: {suffix}")
    print(f"\nTime step between outputs: DT = {DT*1e6:.1f} μs")
    print(f"Spatial resolution: DY = {DY*1e3:.2f} mm")

    # Find and load all NPZ files
    print("\nLoading data from NPZ files...")
    npz_files = get_all_npz_files()
    print(f"Found {len(npz_files)} NPZ files")

    tag_to_file, all_tags, all_manifests = load_all_manifests(npz_files)
    print(f"Total timesteps available: {len(all_tags)}")

    if len(all_tags) == 0:
        print("ERROR: No data found!")
        return

    # Show timestep range
    first_ts = int(all_tags[0])
    last_ts = int(all_tags[-1])
    print(f"Timestep range: {first_ts} -> {last_ts}")

    # Check for gaps
    tags_int = [int(t) for t in all_tags]
    diffs = np.diff(tags_int)
    gap_indices = np.where(diffs > 10)[0]
    if len(gap_indices) > 0:
        print(f"\nWARNING: {len(gap_indices)} gaps detected in data:")
        for gi in gap_indices[:5]:  # Show first 5 gaps
            print(f"  {tags_int[gi]} -> {tags_int[gi+1]} (missing {diffs[gi]//10 - 1} steps)")

    # Select timesteps for analysis (use last contiguous block or all)
    # For wave analysis, we want a continuous sequence
    print("\nSelecting data for analysis...")

    # Find longest contiguous block
    blocks = [[0]]
    for i, d in enumerate(diffs):
        if d == 10:
            blocks[-1].append(i + 1)
        else:
            blocks.append([i + 1])

    # Use longest block
    longest_block = max(blocks, key=len)
    block_start = longest_block[0]
    block_end = longest_block[-1]

    tags = all_tags[block_start:block_end + 1]
    print(f"Using {len(tags)} consecutive timesteps: {tags[0]} -> {tags[-1]}")

    # Y-grid
    y_bins = np.arange(Y_MIN, Y_MAX + DY, DY)
    n_y = len(y_bins)
    n_t = len(tags)

    # Total time span
    total_time_ms = n_t * DT * 1000
    print(f"Total time span: {total_time_ms:.2f} ms")

    # Extract R(y,t) matrix
    print(f"\nExtracting interface R(y,t): {n_y} y-bins × {n_t} timesteps...")
    R_interface = np.full((n_t, n_y), np.nan)

    for n, tag in enumerate(tags):
        if n % 200 == 0:
            print(f"  Timestep {n}/{n_t} ({tag})")

        fields = all_manifests[tag]
        npz_path = tag_to_file[tag]

        # Get VOF field name
        vof_key = 'VOF' if 'VOF' in fields else 'VOF_Sharp' if 'VOF_Sharp' in fields else None
        if vof_key is None:
            continue
        coord_key = f'RY_{vof_key}'
        if coord_key not in fields:
            coord_key = 'RY_VOF'
        if coord_key not in fields:
            continue

        vof = load_field(npz_path, tag, vof_key)
        coords = load_field(npz_path, tag, coord_key)

        if vof.ndim == 2:
            vof = vof.flatten()

        R_interface[n, :] = extract_interface_profile(coords, vof, y_bins)

    # Data quality check
    valid_frac = np.sum(~np.isnan(R_interface)) / R_interface.size
    print(f"\nValid data fraction: {valid_frac*100:.1f}%")

    # Compute perturbation (film thickness fluctuation from time-mean at each y)
    R_mean_t = np.nanmean(R_interface, axis=0)  # Mean at each y over time
    eta = R_interface - R_mean_t  # Perturbation η(t, y)

    # Also compute mean film thickness
    delta = (R_interface - R_INNER) * 1000  # Film thickness in mm
    delta_mean = np.nanmean(delta)

    print(f"\nFilm thickness statistics:")
    print(f"  Mean δ: {delta_mean:.2f} mm")
    print(f"  Min δ:  {np.nanmin(delta):.2f} mm")
    print(f"  Max δ:  {np.nanmax(delta):.2f} mm")
    print(f"\nPerturbation η statistics:")
    print(f"  RMS η: {np.nanstd(eta)*1e6:.1f} μm")

    # =========================================================================
    # HOVMÖLLER DIAGRAM
    # =========================================================================
    print("\nGenerating Hovmöller diagram...")

    # Time and space axes
    t_ms = np.arange(n_t) * DT * 1000  # Time in milliseconds
    y_mm = y_bins * 1000               # Position in mm

    fig, ax = plt.subplots(figsize=(12, 6))

    eta_um = eta.T * 1e6  # Transpose to (y, t) and convert to μm
    vmax = np.nanpercentile(np.abs(eta_um), 99)

    im = ax.pcolormesh(t_ms, y_mm, eta_um, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, shading='auto')

    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Axial position $y$ [mm]')
    ax.set_title(f'Hovmöller Diagram: Interface Perturbation η(y,t)\n'
                 f'Timesteps {tags[0]} → {tags[-1]}, DT = {DT*1e6:.0f} μs')

    cbar = plt.colorbar(im, ax=ax, label='η [μm]', pad=0.02)

    # Add reference lines for measured wave velocities
    # slope on (t_ms, y_mm) plot: v [m/s] = v [mm/ms]
    ref_lines = [
        (1.6, 'white', 'Slow waves: $c \\approx 1.6$ m/s'),
        (12.0, 'lime', 'Fast waves: $c \\approx 12$ m/s'),
    ]
    for v_ref, color, label in ref_lines:
        y_start = 5.0  # mm
        t_start = 0.0
        t_end = min(total_time_ms, (52 - y_start) / v_ref) if v_ref > 0 else total_time_ms
        y_end = y_start + v_ref * t_end
        ax.plot([t_start, t_end], [y_start, y_end], '--', color=color,
                linewidth=2.5, alpha=1.0, label=label)

    ax.legend(loc='upper left', fontsize=9)
    ax.set_xlim([0, total_time_ms])
    ax.set_ylim([0, 52])

    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIR / f'hovmoller_diagram{suffix}.pdf')
    plt.savefig(OUTPUT_DIR / f'hovmoller_diagram{suffix}.png')
    print(f"Saved: {OUTPUT_DIR}/hovmoller_diagram{suffix}.pdf")
    plt.close()

    # Save Hovmöller data for further analysis
    np.savez(OUTPUT_DIR / f'hovmoller_data{suffix}.npz',
             R_interface=R_interface,  # (n_t, n_y) interface position
             eta=eta,                   # (n_t, n_y) perturbation
             y_bins=y_bins * 1000,      # y coordinates in mm
             time_ms=t_ms,              # time in ms
             tags=np.array(tags),       # timestep tags
             DT=DT, DY=DY, R_INNER=R_INNER)
    print(f"Saved: {OUTPUT_DIR}/hovmoller_data{suffix}.npz")

    # =========================================================================
    # WAVE VELOCITY ANALYSIS
    # =========================================================================
    print("\n" + "="*60)
    print("WAVE VELOCITY ANALYSIS")
    print("="*60)

    # Fill NaNs for FFT (linear interpolation)
    eta_filled = eta.copy()
    for i in range(n_t):
        row = eta_filled[i, :]
        valid = ~np.isnan(row)
        if np.sum(valid) > 10 and np.sum(~valid) > 0:
            eta_filled[i, ~valid] = np.interp(
                np.where(~valid)[0],
                np.where(valid)[0],
                row[valid]
            )
    eta_filled = np.nan_to_num(eta_filled, nan=0.0)

    # 2D FFT analysis
    print("\nPerforming 2D FFT analysis...")

    # Apply window
    window_t = np.hanning(n_t)
    window_y = np.hanning(n_y)
    window_2d = np.outer(window_t, window_y)
    eta_windowed = eta_filled * window_2d

    # 2D FFT
    fft_2d = np.fft.fft2(eta_windowed)
    fft_2d_shifted = np.fft.fftshift(fft_2d)
    power = np.abs(fft_2d_shifted)**2

    # Frequency axes
    freq_t = np.fft.fftshift(np.fft.fftfreq(n_t, DT))  # Hz
    freq_y = np.fft.fftshift(np.fft.fftfreq(n_y, DY))  # 1/m
    omega = 2 * np.pi * freq_t  # rad/s
    k_y = 2 * np.pi * freq_y    # rad/m

    # Wave velocity: v = ω/k = f/k_y * 2π
    # Integrate power along lines ω = v * k (or f = v * k_y / 2π)
    v_test = np.linspace(-2, 20, 2201)
    v_power = np.zeros_like(v_test)

    # Only use positive wavenumbers in reasonable range (wavelengths 2-60 mm)
    k_positive = (k_y > 100) & (k_y < 3000)
    k_idx = np.where(k_positive)[0]

    for i, v in enumerate(v_test):
        power_sum = 0
        count = 0
        for ki in k_idx:
            omega_expected = v * k_y[ki]
            omega_idx = np.argmin(np.abs(omega - omega_expected))
            if 5 < omega_idx < len(omega) - 5:
                power_sum += np.sum(power[omega_idx-2:omega_idx+3, ki])
                count += 1
        if count > 0:
            v_power[i] = power_sum / count

    # Smooth and find peak
    v_power_smooth = gaussian_filter(v_power, sigma=5)
    peak_idx = np.argmax(v_power_smooth)
    v_fft = v_test[peak_idx]

    # Estimate uncertainty from FWHM
    half_max = v_power_smooth[peak_idx] / 2
    above_half = v_power_smooth > half_max
    if np.sum(above_half) > 1:
        v_fwhm = v_test[above_half][-1] - v_test[above_half][0]
    else:
        v_fwhm = 0.5

    print(f"\nFFT Wave Velocity: {v_fft:.2f} m/s (FWHM: {v_fwhm:.2f} m/s)")

    # ===================================================================
    # CROSS-CORRELATION: Station-to-station (PhaseVelocity module)
    # Uses the same method as the paper: cross-correlate η(t) time
    # series at pairs of y-stations, find time lag → v = Δy / Δt
    # ===================================================================
    print("\nPerforming station-to-station cross-correlation...")

    # PhaseVelocity expects eta shape (Ny, Nt); ours is (Nt, Ny)
    eta_T = eta_filled.T
    t_seconds = np.arange(n_t) * DT

    # max_row_gap covers station separations up to ~5 mm
    max_gap = int(round(5e-3 / DY))

    cp_est, cp_samples, cp_weights = estimate_phase_speed_from_hovmoller(
        eta_T, y_bins, t_seconds,
        max_row_gap=max_gap,
        max_lag_frac=0.5,
        min_peak_r=0.15
    )

    cp_samples = np.asarray(cp_samples)
    cp_weights = np.asarray(cp_weights)

    # PhaseVelocity convention: R(L) = Σ a(t+L)*b(t), so upward-moving
    # waves (feature arrives at lower y first) give negative cp.
    # Negate to get physical wave velocity in the +y (streamwise) direction.
    cp_est = -cp_est
    cp_samples = -cp_samples

    if np.isfinite(cp_est) and len(cp_samples) > 0:
        v_xcorr = cp_est
        v_xcorr_std = np.std(cp_samples)

        # High-coherence subset → fast waves
        high_coh_mask = cp_weights > 0.5
        if np.sum(high_coh_mask) > 3:
            # Weighted median of high-coherence samples
            order = np.argsort(cp_samples[high_coh_mask])
            c_s = cp_samples[high_coh_mask][order]
            w_s = cp_weights[high_coh_mask][order]
            cum = np.cumsum(w_s) / np.sum(w_s)
            idx = np.searchsorted(cum, 0.5)
            v_fast = float(c_s[min(idx, len(c_s)-1)])
            v_fast_std = np.std(cp_samples[high_coh_mask])
            coh_fast = np.mean(cp_weights[high_coh_mask])
        else:
            v_fast = v_xcorr
            v_fast_std = v_xcorr_std
            coh_fast = np.mean(cp_weights)

        print(f"XCorr Wave Velocity (weighted median): {v_xcorr:.2f} m/s")
        print(f"  N samples: {len(cp_samples)}, mean coherence: {np.mean(cp_weights):.2f}")
        print(f"  Fast waves (coh > 0.5): {v_fast:.2f} ± {v_fast_std:.2f} m/s "
              f"(coh: {coh_fast:.2f}, N={np.sum(high_coh_mask)})")
    else:
        v_xcorr = np.nan
        v_xcorr_std = np.nan
        v_fast = np.nan
        v_fast_std = np.nan
        coh_fast = np.nan
        print("XCorr: No valid results")

    # ===================================================================
    # BANDPASS-FILTERED ANALYSIS: Slow waves (c < 3 m/s)
    # Apply 2D FFT bandpass filter to keep only slow-propagating modes,
    # then re-run station-to-station xcorr on filtered field
    # ===================================================================
    print("\nPerforming bandpass-filtered analysis for slow waves (c < 3 m/s)...")

    fft_2d_raw = np.fft.fft2(eta_filled)
    freq_t_raw = np.fft.fftfreq(n_t, DT)
    freq_y_raw = np.fft.fftfreq(n_y, DY)

    # Phase velocity filter: v_phase = f_t / f_y
    FT, FY = np.meshgrid(freq_t_raw, freq_y_raw, indexing='ij')
    with np.errstate(divide='ignore', invalid='ignore'):
        v_phase_abs = np.where(np.abs(FY) > 1e-6, np.abs(FT / FY), np.inf)

    # Keep modes with 0.1 < |v_phase| < 3 m/s
    mask_slow = (v_phase_abs < 3.0) & (v_phase_abs > 0.1)
    mask_slow[0, 0] = True  # Keep DC

    fft_filtered = fft_2d_raw * mask_slow
    eta_slow = np.real(np.fft.ifft2(fft_filtered))

    # Energy fraction
    energy_total = np.sum(np.abs(fft_2d_raw)**2)
    energy_slow = np.sum(np.abs(fft_filtered)**2)
    print(f"  Energy fraction in slow modes: {energy_slow/energy_total*100:.1f}%")

    # Re-run phase velocity estimation on filtered data
    eta_slow_T = eta_slow.T
    cp_slow, cp_slow_samples, cp_slow_weights = estimate_phase_speed_from_hovmoller(
        eta_slow_T, y_bins, t_seconds,
        max_row_gap=max_gap,
        max_lag_frac=0.5,
        min_peak_r=0.10
    )

    cp_slow = -cp_slow
    cp_slow_samples = [-s for s in cp_slow_samples]

    if np.isfinite(cp_slow) and len(cp_slow_samples) > 0:
        v_slow = cp_slow
        v_slow_std = np.std(cp_slow_samples)
        coh_slow = np.mean(cp_slow_weights)
        print(f"  Slow waves (filtered): {v_slow:.2f} ± {v_slow_std:.2f} m/s "
              f"(coh: {coh_slow:.2f}, N={len(cp_slow_samples)})")
    else:
        v_slow = np.nan
        v_slow_std = np.nan
        coh_slow = np.nan
        print("  Slow waves: No valid results from filtered analysis")

    # =========================================================================
    # SUMMARY FIGURE
    # =========================================================================
    print("\nGenerating summary figure...")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # (a) Hovmöller diagram
    ax = axes[0, 0]
    im = ax.pcolormesh(t_ms, y_mm, eta_um, cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax, shading='auto')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Axial position $y$ [mm]')
    ax.set_title('(a) Interface perturbation η(y,t)')
    plt.colorbar(im, ax=ax, label='η [μm]')

    # (b) 2D Power spectrum
    ax = axes[0, 1]
    f_max = 5000  # Hz (much higher now with DT = 1 μs)
    k_max = 2000  # 1/m
    f_mask = np.abs(freq_t) < f_max
    k_mask = np.abs(freq_y) < k_max

    power_plot = np.log10(power[np.ix_(f_mask, k_mask)] + 1)
    im = ax.pcolormesh(freq_y[k_mask], freq_t[f_mask], power_plot,
                       cmap='viridis', shading='auto')
    ax.set_xlabel('Wavenumber $k_y$ [1/m]')
    ax.set_ylabel('Frequency $f$ [Hz]')
    ax.set_title('(b) 2D Power Spectrum')
    plt.colorbar(im, ax=ax, label='log₁₀(Power)')

    # Plot dispersion lines
    for v in [1.0, 2.0, 5.0, 11.0]:
        k_line = np.linspace(0, k_max, 100)
        f_line = v * k_line / (2 * np.pi)
        mask_f = f_line < f_max
        if np.sum(mask_f) > 1:
            ax.plot(k_line[mask_f], f_line[mask_f], 'w--', alpha=0.5, linewidth=1)
            ax.text(k_line[mask_f][-1]*0.7, f_line[mask_f][-1]*0.9, f'{v} m/s',
                    color='white', fontsize=8)

    # (c) Velocity spectrum
    ax = axes[0, 2]
    ax.plot(v_test, v_power_smooth / np.max(v_power_smooth), 'b-', linewidth=2)
    ax.axvline(v_fft, color='r', linestyle='--', linewidth=2,
               label=f'Peak: {v_fft:.2f} m/s')
    ax.axvspan(0.5, 1.7, alpha=0.2, color='green', label='Barbosa range')
    ax.set_xlabel('Wave velocity [m/s]')
    ax.set_ylabel('Normalized power')
    ax.set_title('(c) Velocity Spectrum (2D FFT)')
    ax.legend()
    ax.set_xlim([-2, 18])
    ax.grid(True, alpha=0.3)

    # (d) Sample time series
    ax = axes[1, 0]
    y_samples = [10, 20, 30, 40]  # mm
    colors = plt.cm.viridis(np.linspace(0, 1, len(y_samples)))
    n_plot = min(500, n_t)  # Plot first 500 timesteps
    for i, y_target in enumerate(y_samples):
        y_idx = np.argmin(np.abs(y_mm - y_target))
        eta_ts = eta[:n_plot, y_idx] * 1e6
        ax.plot(t_ms[:n_plot], eta_ts, '-', color=colors[i],
                linewidth=0.8, label=f'y={y_target} mm')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('η [μm]')
    ax.set_title('(d) Sample Time Series')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (e) Film thickness profile
    ax = axes[1, 1]
    delta_mean_y = np.nanmean(delta, axis=0)
    delta_std_y = np.nanstd(delta, axis=0)

    ax.fill_between(y_mm, delta_mean_y - delta_std_y, delta_mean_y + delta_std_y,
                    alpha=0.3, color='blue')
    ax.plot(y_mm, delta_mean_y, 'b-', linewidth=2)
    ax.axhline(np.nanmean(delta_mean_y), color='r', linestyle='--',
               label=f'Mean: {np.nanmean(delta_mean_y):.2f} mm')
    ax.set_xlabel('Axial position $y$ [mm]')
    ax.set_ylabel('Film thickness δ [mm]')
    ax.set_title('(e) Mean Film Thickness Profile')
    ax.legend()
    ax.set_xlim([0, 52])
    ax.grid(True, alpha=0.3)

    # (f) Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = f"""
    WAVE VELOCITY SUMMARY
    {'='*35}

    Station-to-station XCorr (Δy=5mm):
      Fast waves:  {v_fast:.1f} ± {v_fast_std:.1f} m/s
        coherence: {coh_fast:.2f}
      All stations: {v_xcorr:.1f} ± {v_xcorr_std:.1f} m/s

    Bandpass filtered (c < 3 m/s):
      Slow waves:  {v_slow:.1f} ± {v_slow_std:.1f} m/s

    2D FFT peak:   {v_fft:.1f} m/s

    Reference (Barbosa et al.):
      Expected: 0.5 - 1.7 m/s

    {'='*35}
    Data: {n_t} steps, {total_time_ms:.1f} ms
    Mean δ: {delta_mean:.2f} mm, η RMS: {np.nanstd(eta)*1e6:.0f} μm
    """

    ax.text(0.1, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'wave_velocity_analysis{suffix}.pdf')
    plt.savefig(OUTPUT_DIR / f'wave_velocity_analysis{suffix}.png')
    print(f"Saved: {OUTPUT_DIR}/wave_velocity_analysis{suffix}.pdf")
    plt.close()

    # Final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Station-to-station XCorr (Δy = 5 mm):")
    print(f"  Fast waves (coh > 0.5): {v_fast:.2f} ± {v_fast_std:.2f} m/s (coh: {coh_fast:.2f})")
    print(f"  All stations:           {v_xcorr:.2f} ± {v_xcorr_std:.2f} m/s")
    print(f"Bandpass filtered (c < 3 m/s):")
    print(f"  Slow waves:             {v_slow:.2f} ± {v_slow_std:.2f} m/s")
    print(f"2D FFT peak:              {v_fft:.2f} m/s (FWHM: {v_fwhm:.2f} m/s)")
    print(f"Barbosa reference:        0.5 - 1.7 m/s")
    print(f"\nData parameters: DT = {DT*1e6:.0f} μs, span = {total_time_ms:.1f} ms")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', default='', help='Suffix for output filenames')
    args = parser.parse_args()
    main(suffix=args.suffix)
