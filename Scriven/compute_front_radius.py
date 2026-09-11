#!/usr/bin/env python3
"""
Compute bubble radius from front element positions in T-Flows PVTU files.

For each front snapshot, computes three radius measures:
  - R_vol:   equivalent spherical radius from VOF volume = (3V/4pi)^(1/3)
  - R_area:  radius from surface area assuming sphere    = sqrt(A/4pi)
  - R_front: area-weighted average distance from bubble centre to element
             centroids (most robust for non-spherical / wrinkled fronts)

Usage:
    python3 compute_front_radius.py <case_dir> [--dt DT]

Example:
    python3 compute_front_radius.py /home/jan/Scriven-Merlin/Scriven-Poly-150
"""

import numpy as np
import pyvista as pv
import glob
import os
import re
import argparse

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


def compute_front_radius(pvtu_path):
    """Read a front PVTU file and compute radius metrics.

    Returns dict with: n_elems, total_area, total_volume,
                        R_area, R_front, center_x/y/z
    """
    mesh = pv.read(pvtu_path)

    # Element centroids and areas from cell data
    centroids = mesh.cell_data['ElementCoordinates']  # (N, 3)
    areas = mesh.cell_data['ElementArea']              # (N,)

    total_area = np.sum(areas)
    n_elems = len(areas)

    if total_area < 1e-30:
        return None

    # Area-weighted bubble centre
    cx = np.sum(centroids[:, 0] * areas) / total_area
    cy = np.sum(centroids[:, 1] * areas) / total_area
    cz = np.sum(centroids[:, 2] * areas) / total_area

    # Area-weighted average distance from centre to element centroids
    dist = np.sqrt((centroids[:, 0] - cx)**2
                 + (centroids[:, 1] - cy)**2
                 + (centroids[:, 2] - cz)**2)
    R_front = np.sum(dist * areas) / total_area

    # R from surface area (assuming sphere)
    R_area = np.sqrt(total_area / (4.0 * np.pi))

    return {
        'n_elems': n_elems,
        'total_area': total_area,
        'R_area': R_area,
        'R_front': R_front,
        'center': (cx, cy, cz),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compute bubble radius from front PVTU files')
    parser.add_argument('case_dir',
                        help='Path to the T-Flows case directory')
    parser.add_argument('--dt', type=float, default=None,
                        help='Time step [s]. Auto-detected from control file '
                             'if not specified.')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file path (default: front-radius.dat '
                             'in case_dir)')
    args = parser.parse_args()

    case_dir = args.case_dir

    # Auto-detect dt from control file
    dt = args.dt
    if dt is None:
        control_path = os.path.join(case_dir, 'control')
        if os.path.exists(control_path):
            with open(control_path) as f:
                for line in f:
                    if 'TIME_STEP' in line and 'NUMBER' not in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == 'TIME_STEP':
                                dt = float(parts[i + 1])
                                break
                        if dt is not None:
                            break
        if dt is None:
            print("  WARNING: Could not detect TIME_STEP from control file.")
            print("           Specify with --dt. Using dt=2e-6 as fallback.")
            dt = 2.0e-6

    print(f"  Time step: dt = {dt:.2e} s")

    # Find all front PVTU files
    pattern = os.path.join(case_dir, '*front-ts*.pvtu')
    pvtu_files = sorted(glob.glob(pattern))

    if not pvtu_files:
        print(f"  ERROR: No front PVTU files found matching {pattern}")
        return

    print(f"  Found {len(pvtu_files)} front PVTU files")

    # Extract timestep number from filename
    ts_pattern = re.compile(r'front-ts(\d+)\.pvtu$')

    results = []
    for fpath in pvtu_files:
        m = ts_pattern.search(os.path.basename(fpath))
        if not m:
            continue
        ts = int(m.group(1))
        t_sim = ts * dt

        try:
            r = compute_front_radius(fpath)
        except Exception as e:
            print(f"  WARNING: Could not read {os.path.basename(fpath)}: {e}")
            continue

        if r is None:
            continue

        t_anal = t_sim + t0
        R_anal = scriven_radius(t_anal)

        # Also compute R_vol from bench-data if available (for comparison)
        err_front = abs(r['R_front'] - R_anal) / R_anal * 100
        err_area = abs(r['R_area'] - R_anal) / R_anal * 100

        results.append({
            'ts': ts,
            't_sim': t_sim,
            'R_front': r['R_front'],
            'R_area': r['R_area'],
            'R_anal': R_anal,
            'err_front': err_front,
            'err_area': err_area,
            'n_elems': r['n_elems'],
            'total_area': r['total_area'],
            'center': r['center'],
        })

        print(f"  ts={ts:6d}  t={t_sim*1e3:7.3f} ms  "
              f"R_front={r['R_front']*1e6:7.1f} um  "
              f"R_anal={R_anal*1e6:7.1f} um  "
              f"err={err_front:5.2f}%  "
              f"n_elem={r['n_elems']}")

    if not results:
        print("  No results computed.")
        return

    # Write output file
    out_path = args.output or os.path.join(case_dir, 'front-radius.dat')
    header = (f"# Front radius analysis for {os.path.basename(case_dir)}\n"
              f"# dt = {dt:.2e} s\n"
              f"# Columns: time[s]  R_front[m]  R_area[m]  R_anal[m]  "
              f"err_front[%]  err_area[%]  n_elems  surface_area[m2]\n")

    with open(out_path, 'w') as f:
        f.write(header)
        for r in results:
            f.write(f"{r['t_sim']:16.8e}  {r['R_front']:16.8e}  "
                    f"{r['R_area']:16.8e}  {r['R_anal']:16.8e}  "
                    f"{r['err_front']:8.4f}  {r['err_area']:8.4f}  "
                    f"{r['n_elems']:6d}  {r['total_area']:16.8e}\n")

    print(f"\n  Saved: {out_path}")

    # Summary
    last = results[-1]
    print(f"\n  === Summary at t = {last['t_sim']*1e3:.3f} ms ===")
    print(f"  R_front = {last['R_front']*1e6:.2f} um  "
          f"(err = {last['err_front']:.2f}%)")
    print(f"  R_area  = {last['R_area']*1e6:.2f} um  "
          f"(err = {last['err_area']:.2f}%)")
    print(f"  R_anal  = {last['R_anal']*1e6:.2f} um")
    print(f"  Front elements: {last['n_elems']}")


if __name__ == '__main__':
    main()
