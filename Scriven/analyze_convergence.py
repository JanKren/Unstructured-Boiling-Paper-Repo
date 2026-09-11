#!/usr/bin/env python3
"""
Grid convergence analysis for the Scriven bubble growth problem.

Computes:
  - L2 and Linf error norms for bubble radius on 3 structured meshes
  - Observed convergence order via Richardson extrapolation
  - Grid Convergence Index (GCI) per Roache/Celik et al. (2008)
  - Publication-quality convergence plot

Reviewer comment: Convergence order not quantified.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.figsize': (6, 4),
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
})

# ===========================================================================
#  Paths
# ===========================================================================
SCRIVEN_DIR = os.environ.get("SCRIVEN_DIR",
                          "/home/jan/runs/tflows-vof/Validation/Scriven")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR", os.path.join(os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling"), "pics"))

# ===========================================================================
#  Physical parameters
# ===========================================================================
rho_l   = 958.4
cp_l    = 4216.0
k_l     = 0.679
beta    = 4.06022
alpha_l = k_l / (rho_l * cp_l)
R0      = 5.0e-5
t0      = R0**2 / (4.0 * beta**2 * alpha_l)  # virtual origin


def scriven_radius(t):
    """Analytical Scriven radius R(t) = 2*beta*sqrt(alpha_l * t)."""
    return 2.0 * beta * np.sqrt(alpha_l * t)


def load_bench_data(fpath):
    """Load bench-data file, handling NaN rows and time resets.

    When a file contains multiple concatenated runs (detected by
    time going backwards), take the last segment that has more
    than 50 rows — this skips trailing incomplete restarts.
    """
    data = np.loadtxt(fpath)
    # Remove NaN rows
    mask = ~np.isnan(data).any(axis=1)
    data = data[mask]
    # Detect time resets (multiple concatenated runs)
    resets = np.where(np.diff(data[:, 0]) < 0)[0]
    if len(resets) == 0:
        return data
    # Build segments and pick the longest one
    starts = [0] + [r + 1 for r in resets]
    ends   = list(resets) + [len(data) - 1]
    best_s, best_e, best_n = 0, len(data) - 1, 0
    for s, e in zip(starts, ends):
        n = e - s + 1
        if n > best_n:
            best_s, best_e, best_n = s, e, n
    return data[best_s:best_e + 1]


def compute_errors(data, label=""):
    """Compute error metrics for bubble radius vs analytical."""
    t_sim = data[:, 0]
    R_num = data[:, 3]
    t_anal = t_sim + t0
    R_anal = scriven_radius(t_anal)

    # Relative error at each timestep
    rel_err = np.abs(R_num - R_anal) / R_anal

    # Error norms
    L2 = np.sqrt(np.mean(rel_err**2))
    Linf = np.max(rel_err)
    L1 = np.mean(rel_err)

    # Final time error
    final_err = rel_err[-1]

    if label:
        print(f"  {label}:")
        print(f"    Time range: [{t_sim[0]:.2e}, {t_sim[-1]:.2e}] s "
              f"({len(t_sim)} steps)")
        print(f"    L1  (mean |e|):  {L1:.6f}  ({L1*100:.4f}%)")
        print(f"    L2  (RMS |e|):   {L2:.6f}  ({L2*100:.4f}%)")
        print(f"    Linf (max |e|):  {Linf:.6f}  ({Linf*100:.4f}%)")
        print(f"    Final error:     {final_err:.6f}  ({final_err*100:.4f}%)")

    return {
        't_sim': t_sim, 'R_num': R_num, 'R_anal': R_anal,
        'rel_err': rel_err,
        'L1': L1, 'L2': L2, 'Linf': Linf, 'final': final_err,
    }


def richardson_extrapolation(f1, f2, f3, r21, r32):
    """
    Richardson extrapolation for non-uniform grid refinement.
    Following Celik et al. (2008) "Procedure for Estimation and Reporting
    of Uncertainty Due to Discretization in CFD Applications", JFE 130.

    f1 = finest grid solution (h1 smallest)
    f2 = medium grid solution
    f3 = coarsest grid solution
    r21 = h2/h1 (>1)
    r32 = h3/h2 (>1)

    Returns: (p, f_exact, GCI21, GCI32)
    """
    eps32 = f3 - f2
    eps21 = f2 - f1

    if abs(eps21) < 1e-15 or abs(eps32) < 1e-15:
        print("  WARNING: errors too small for reliable Richardson extrap.")
        return np.nan, f1, np.nan, np.nan

    s = np.sign(eps32 / eps21)

    # Iterative solution for observed order p
    # Initial guess (assuming uniform refinement)
    r_eff = np.sqrt(r21 * r32)
    p = abs(np.log(abs(eps32 / eps21))) / np.log(r_eff)

    # Iterate with correction for non-uniform refinement
    for _ in range(50):
        q_p = np.log((r21**p - s) / (r32**p - s))
        p_new = (1.0 / np.log(r21)) * abs(np.log(abs(eps32 / eps21)) + q_p)
        if abs(p_new - p) < 1e-6:
            p = p_new
            break
        p = p_new

    # Extrapolated value (Richardson)
    f_exact = (r21**p * f1 - f2) / (r21**p - 1.0)

    # GCI with safety factor 1.25 (3 grids available)
    e_a21 = abs((f1 - f2) / f1)  # approximate relative error
    e_a32 = abs((f2 - f3) / f2)
    GCI21 = 1.25 * e_a21 / (r21**p - 1.0)
    GCI32 = 1.25 * e_a32 / (r32**p - 1.0)

    return p, f_exact, GCI21, GCI32


# ===========================================================================
#  Main
# ===========================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  SCRIVEN PROBLEM - Grid Convergence Analysis")
    print("=" * 65)

    # Grid definitions (finest to coarsest for Richardson convention)
    grids = [
        {'label': r'$150^3$', 'N': 150, 'h': 2.0e-6,
         'file': os.path.join(SCRIVEN_DIR, "bench-data150.dat")},
        {'label': r'$125^3$', 'N': 125, 'h': 2.4e-6,
         'file': os.path.join(SCRIVEN_DIR, "Scriven-Struct-125",
                              "bench-data.dat")},
        {'label': r'$100^3$', 'N': 100, 'h': 3.0e-6,
         'file': os.path.join(SCRIVEN_DIR, "bench-data100.dat")},
        {'label': r'$75^3$',  'N': 75,  'h': 4.0e-6,
         'file': os.path.join(SCRIVEN_DIR, "bench-data.dat")},
    ]

    # Load all data first to find common time range
    raw_data = {}
    for g in grids:
        data = load_bench_data(g['file'])
        raw_data[g['N']] = data
        print(f"  Loaded {g['label']}: {data.shape[0]} steps, "
              f"t=[{data[0,0]:.2e}, {data[-1,0]:.2e}] s")

    # Find common time range (limited by shortest simulation)
    t_max = min(d[-1, 0] for d in raw_data.values())
    print(f"\n  Common time range: [1e-5, {t_max:.4e}] s")

    results = []
    for g in grids:
        print(f"\n--- Grid {g['label']} (h = {g['h']*1e6:.1f} um) ---")
        data = raw_data[g['N']]
        # Restrict to common time range
        mask = data[:, 0] <= t_max + 1e-10
        data = data[mask]
        errs = compute_errors(data, g['label'])
        errs['h'] = g['h']
        errs['N'] = g['N']
        errs['label'] = g['label']
        results.append(errs)

    # ------------------------------------------------------------------
    # Richardson extrapolation
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  RICHARDSON EXTRAPOLATION")
    print("=" * 65)

    r21 = results[1]['h'] / results[0]['h']  # h2/h1 = 3/2
    r32 = results[2]['h'] / results[1]['h']  # h3/h2 = 4/3
    print(f"  Refinement ratios: r21 = {r21:.4f}, r32 = {r32:.4f}")

    for metric_name, metric_key in [('L2 error', 'L2'),
                                     ('Linf error', 'Linf'),
                                     ('Final error', 'final')]:
        f1 = results[0][metric_key]
        f2 = results[1][metric_key]
        f3 = results[2][metric_key]

        p, f_ext, GCI21, GCI32 = richardson_extrapolation(
            f1, f2, f3, r21, r32)

        print(f"\n  {metric_name}:")
        print(f"    f1 (150³) = {f1:.6e}")
        print(f"    f2 (100³) = {f2:.6e}")
        print(f"    f3 (75³)  = {f3:.6e}")
        print(f"    Observed order p = {p:.3f}")
        print(f"    Richardson extrapolated = {f_ext:.6e}")
        print(f"    GCI_fine   (21) = {GCI21*100:.4f}%")
        print(f"    GCI_coarse (32) = {GCI32*100:.4f}%")
        # Check asymptotic range
        if not np.isnan(GCI21) and not np.isnan(GCI32):
            asymp = GCI32 / (r21**p * GCI21)
            print(f"    Asymptotic range indicator = {asymp:.4f} "
                  f"(should be ~1.0)")

    # ------------------------------------------------------------------
    # Simple log-log convergence rate (direct slope)
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  LOG-LOG CONVERGENCE RATES (simple slope)")
    print("=" * 65)

    hs = np.array([r['h'] for r in results])
    for metric_key, metric_name in [('L2', 'L2'), ('Linf', 'Linf')]:
        errs = np.array([r[metric_key] for r in results])
        # Fit log-log line
        coeffs = np.polyfit(np.log(hs), np.log(errs), 1)
        slope = coeffs[0]
        print(f"  {metric_name}: slope = {slope:.3f} "
              f"(convergence order ~ {slope:.2f})")

    # ------------------------------------------------------------------
    # Figure: Convergence plot (log-log, single panel)
    # ------------------------------------------------------------------
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4.5))

    hs_um = np.array([r['h'] * 1e6 for r in results])
    L2s = np.array([r['L2'] * 100 for r in results])
    Linfs = np.array([r['Linf'] * 100 for r in results])

    ax1.loglog(hs_um, L2s, 'bo-', ms=8, label=r'$L_2$ (RMS)')
    ax1.loglog(hs_um, Linfs, 'rs-', ms=8, label=r'$L_\infty$ (max)')

    # Reference slopes
    h_ref = np.linspace(1.5, 5.0, 50)
    scale1 = L2s[1] / (hs_um[1]**1)
    ax1.loglog(h_ref, scale1 * h_ref**1, 'k--', alpha=0.4,
               label='1st order')
    scale2 = L2s[1] / (hs_um[1]**2)
    ax1.loglog(h_ref, scale2 * h_ref**2, 'k:', alpha=0.4,
               label='2nd order')

    ax1.set_xlabel(r'Cell size $h$ [$\mu$m]')
    ax1.set_ylabel('Relative error [%]')
    ax1.legend(fontsize=10)
    ax1.set_xlim([1.5, 5.0])

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "scriven_convergence.png")
    plt.savefig(outpath, bbox_inches='tight')
    plt.savefig(outpath.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"\n  Saved: {outpath}")
    plt.close()

    # ------------------------------------------------------------------
    # Summary table (LaTeX format)
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  LATEX TABLE")
    print("=" * 65)
    print(r"""
\begin{table}[htbp]
    \centering
    \caption{Grid convergence study for the Scriven problem:
    relative error norms in bubble radius and observed convergence order.}
    \label{tab:scriven_convergence}
    \begin{tabular}{lccccc}
        \hline
        \textbf{Mesh} & $h$ [$\mu$m] & $N_\mathrm{cells}$
        & $L_2$ error [\%] & $L_\infty$ error [\%] \\
        \hline""")
    for r in reversed(results):  # coarsest to finest
        print(f"        {r['label']} & {r['h']*1e6:.1f} & "
              f"{r['N']**3:,} & {r['L2']*100:.2f} & "
              f"{r['Linf']*100:.2f} \\\\")
    print(r"""        \hline
    \end{tabular}
\end{table}
""")

    print("\nDone.")
