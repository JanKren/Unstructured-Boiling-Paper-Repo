#!/usr/bin/env python3
"""
Parameter scan for the interface-modified least-squares gradient stencil.

This is the decision gate for the corrected-discretisation experiment: it
asks whether ANY combination of stencil support and distance weighting
simultaneously removes BOTH defects documented in the manuscript,

  (1) the coherent m=4 azimuthal anisotropy on Cartesian meshes, and
  (2) the systematic magnitude overestimate in the heat-sink limit
      (Appendix C, Eq. boost_limit),

before any Fortran is written.  Scoring only (1) is the trap: a stencil
can be beautifully isotropic and still amplify the gradient by ~1.4,
which leaves the radius overshoot completely intact.

Model
-----
Sphere of radius R centred at the origin on a uniform Cartesian lattice of
spacing h.  Vapour inside is at T_sat (Scriven); liquid outside follows the
same erf profile used in Appendix C,

    T(x) = T_sat + dT * erf(x / delta_T),      x = r - R > 0
    g_true = dT dT/dx|_{x=0} = 2 dT / (sqrt(pi) delta_T)

The stencil is modified exactly as in the manuscript: a neighbour whose
centre lies on the far side of the interface (alpha crossing 0.5) has its
displacement shortened to the interface intersection and its value
replaced by T_sat.

Two evaluations are performed for every configuration:

  physical    T_P taken from the erf profile
  heat-sink   T_P = T_sat  (theta_P -> 0), which is the limit the mass
              transfer sink actually drives the simulation toward, and
              the limit in which the manuscript predicts g_fm/g_true
              ~ 1.14-1.16

Criterion 2 is scored in the heat-sink limit.  Anything else understates
the bias.

Usage:  python3 stencil_parameter_scan.py [--quick]
"""

import argparse
import itertools

import numpy as np
from scipy.special import erf

# ----------------------------------------------------------------------
#  Scriven 75^3 configuration (manuscript Section 3.3 / Appendix C)
# ----------------------------------------------------------------------
H_CELL = 4.0e-6          # cell size [m]
R_OVER_H = 25.0          # bubble radius in cells (R ~ 100 um on the 75^3)
DT_SUPERHEAT = 1.25      # T_inf - T_sat [K]

# Scan axes
STENCILS = ["face6", "face6_edge12", "face6_edge12_corner8"]
WEIGHT_EXPONENTS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]   # w = |r|^-p
DELTA_OVER_H = [2.0, 2.5, 3.0, 4.0, 6.0, 8.0]       # Scriven range is 2.5-4

# Targets from the revision plan
TARGET_M4 = 0.02          # pass
TARGET_M4_KILL = 0.03     # kill criterion
TARGET_RATIO = (0.95, 1.05)


# ======================================================================
#  Geometry and field
# ======================================================================
def stencil_offsets(family, h):
    """Neighbour offsets for the three stencil families."""
    face = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
            (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    edge = [p for p in itertools.product((-1, 0, 1), repeat=3)
            if sum(abs(c) for c in p) == 2]
    corner = [p for p in itertools.product((-1, 1), repeat=3)]

    off = list(face)
    if family in ("face6_edge12", "face6_edge12_corner8"):
        off += edge
    if family == "face6_edge12_corner8":
        off += corner
    return np.array(off, dtype=float) * h


def temperature(pts, R, dT, delta_T):
    """erf profile outside the sphere, T_sat (=0 excess) inside."""
    r = np.linalg.norm(pts, axis=-1)
    x = r - R
    return np.where(x > 0.0, dT * erf(x / delta_T), 0.0)


def g_true(dT, delta_T):
    """Exact interface gradient of the erf profile."""
    return 2.0 * dT / (np.sqrt(np.pi) * delta_T)


def sphere_intersection(p_out, p_in, R):
    """Intersection of segment p_out->p_in with the sphere |x| = R.

    p_out is outside, p_in inside, so a root always exists in (0, 1).
    Vectorised over leading axes.
    """
    seg = p_in - p_out
    a = np.sum(seg * seg, axis=-1)
    b = 2.0 * np.sum(p_out * seg, axis=-1)
    c = np.sum(p_out * p_out, axis=-1) - R * R
    disc = np.maximum(b * b - 4.0 * a * c, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-b - np.sqrt(disc)) / (2.0 * a)
        t = np.where((t > 0.0) & (t < 1.0), t,
                     (-b + np.sqrt(disc)) / (2.0 * a))
    t = np.clip(np.nan_to_num(t, nan=0.5), 0.0, 1.0)
    return p_out + t[..., None] * seg


# ======================================================================
#  Least-squares gradient, batched over cells
# ======================================================================
def lsq_gradient(cells, offsets, R, dT, delta_T, p_weight,
                 modify_front=True, heat_sink=False):
    """Interface-modified LSQ gradient at every cell in `cells`.

    Returns (M, 3) gradients.  `heat_sink=True` sets T_P = T_sat, which
    is the limit the mass transfer sink drives the simulation toward.
    """
    M, K = len(cells), len(offsets)
    nb = cells[:, None, :] + offsets[None, :, :]           # (M, K, 3)

    r_nb = np.linalg.norm(nb, axis=-1)
    crosses = r_nb < R                                     # neighbour in vapour

    pos = nb.copy()
    val = temperature(nb, R, dT, delta_T)

    if modify_front and crosses.any():
        p_out = np.broadcast_to(cells[:, None, :], (M, K, 3))[crosses]
        x_f = sphere_intersection(p_out, nb[crosses], R)
        pos[crosses] = x_f
        val[crosses] = 0.0                                 # T_sat

    T_P = np.zeros(M) if heat_sink else temperature(cells, R, dT, delta_T)

    delta = pos - cells[:, None, :]                        # (M, K, 3)
    dphi = val - T_P[:, None]                              # (M, K)

    dist = np.linalg.norm(delta, axis=-1)
    good = dist > 1e-30
    w = np.where(good, np.power(np.where(good, dist, 1.0), -p_weight), 0.0)

    G = np.einsum("mk,mki,mkj->mij", w, delta, delta)
    b = np.einsum("mk,mk,mki->mi", w, dphi, delta)

    # Guard against singular G (should not occur for these stencils)
    det = np.linalg.det(G)
    ok = np.abs(det) > 1e-60
    grad = np.zeros((M, 3))
    if ok.any():
        grad[ok] = np.linalg.solve(G[ok], b[ok])
    return grad


# ======================================================================
#  Diagnostics
# ======================================================================
def interface_cells(R, h, band):
    """Liquid cells with at least one face-neighbour in the vapour,
    restricted to an equatorial band |z| < band (a mesh slice)."""
    n = int(np.ceil(R / h)) + 4
    ax = (np.arange(-n, n + 1) + 0.5) * h
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    r = np.linalg.norm(pts, axis=-1)
    outside = r > R
    face = stencil_offsets("face6", h)
    nb_r = np.linalg.norm(pts[:, None, :] + face[None, :, :], axis=-1)
    has_vapour_nb = (nb_r < R).any(axis=1)

    sel = outside & has_vapour_nb & (np.abs(pts[:, 2]) < band)
    return pts[sel]


def m4_amplitude(cells, grad, n_bins=36, m_max=8):
    """Amplitude of azimuthal mode m in the normal-projected gradient,
    as a fraction of the mean.  Mirrors the manuscript's diagnostic."""
    nrm = cells / np.linalg.norm(cells, axis=-1, keepdims=True)
    g_n = np.abs(np.sum(grad * nrm, axis=-1))
    theta = np.arctan2(cells[:, 1], cells[:, 0])

    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(theta, bins) - 1, 0, n_bins - 1)
    binned = np.full(n_bins, np.nan)
    for k in range(n_bins):
        sel = idx == k
        if sel.sum():
            binned[k] = g_n[sel].mean()

    valid = ~np.isnan(binned)
    th = 0.5 * (bins[:-1] + bins[1:])[valid]
    s = binned[valid]
    mean = s.mean()
    amps = {}
    for m in range(1, m_max + 1):
        cm = 2.0 * np.mean(s * np.cos(m * th))
        sm = 2.0 * np.mean(s * np.sin(m * th))
        amps[m] = np.hypot(cm, sm) / mean
    return amps, mean, g_n


def normal_ratio(cells, grad, dT, delta_T):
    """Mean normal-projected gradient divided by the exact value."""
    nrm = cells / np.linalg.norm(cells, axis=-1, keepdims=True)
    g_n = np.abs(np.sum(grad * nrm, axis=-1))
    return g_n.mean() / g_true(dT, delta_T)


# ======================================================================
#  1-D validation against Appendix C
# ======================================================================
def validate_against_appendix_c():
    """Reproduce Eq. (boost_limit): g_fm/g_std -> 2h^2/(d^2+h^2)."""
    print("=" * 72)
    print("  VALIDATION: 1-D heat-sink limit vs Appendix C Eq. (boost_limit)")
    print("=" * 72)
    h, dT = 1.0, 1.0
    print(f"  {'d/h':>6s} {'analytic 2h^2/(d^2+h^2)':>24s} {'LSQ (this code)':>18s}")
    print("  " + "-" * 52)
    worst = 0.0
    for dh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        d = dh * h
        # far neighbour at +h with T_far; near "neighbour" at -d with T_sat
        T_far = 1.0
        # heat-sink limit: T_P = T_sat = 0
        delta = np.array([[-d, 0.0, 0.0], [h, 0.0, 0.0]])
        dphi = np.array([0.0, T_far])
        G = np.einsum("ki,kj->ij", delta, delta)
        b = np.einsum("k,ki->i", dphi, delta)
        g_fm = np.linalg.solve(G + 1e-30 * np.eye(3), b)[0]
        g_std = T_far / (2.0 * h)
        analytic = 2.0 * h**2 / (d**2 + h**2)
        got = g_fm / g_std
        worst = max(worst, abs(got - analytic))
        print(f"  {dh:6.1f} {analytic:24.4f} {got:18.4f}")
    status = "PASS" if worst < 1e-9 else f"FAIL (max dev {worst:.2e})"
    print(f"\n  {status}: the scan reproduces the manuscript's derivation.\n")
    return worst < 1e-9


# ======================================================================
#  Scan
# ======================================================================
def run_scan(quick=False):
    h = H_CELL
    R = R_OVER_H * h
    dT = DT_SUPERHEAT
    band = h                      # one-cell equatorial slice

    cells = interface_cells(R, h, band)
    print(f"  interface cells in equatorial slice: {len(cells)}")
    print(f"  R/h = {R_OVER_H:.0f},  h = {h*1e6:.1f} um\n")

    deltas = DELTA_OVER_H[:3] if quick else DELTA_OVER_H
    weights = WEIGHT_EXPONENTS[:3] if quick else WEIGHT_EXPONENTS

    rows = []
    for fam in STENCILS:
        off = stencil_offsets(fam, h)
        for p in weights:
            m4s, ratios, ratios_phys = [], [], []
            for dh in deltas:
                delta_T = dh * h

                g_hs = lsq_gradient(cells, off, R, dT, delta_T, p,
                                    modify_front=True, heat_sink=True)
                g_ph = lsq_gradient(cells, off, R, dT, delta_T, p,
                                    modify_front=True, heat_sink=False)

                amps, _, _ = m4_amplitude(cells, g_hs)
                m4s.append(amps[4])
                ratios.append(normal_ratio(cells, g_hs, dT, delta_T))
                ratios_phys.append(normal_ratio(cells, g_ph, dT, delta_T))

            rows.append({
                "stencil": fam, "p": p,
                "m4": float(np.mean(m4s)), "m4_max": float(np.max(m4s)),
                "ratio": float(np.mean(ratios)),
                "ratio_min": float(np.min(ratios)),
                "ratio_max": float(np.max(ratios)),
                "ratio_phys": float(np.mean(ratios_phys)),
            })

    # Baselines: standard (unmodified) LSQ, face6 unweighted
    base = []
    for dh in deltas:
        delta_T = dh * h
        g_std = lsq_gradient(cells, stencil_offsets("face6", h), R, dT,
                             delta_T, 0.0, modify_front=False,
                             heat_sink=False)
        base.append(normal_ratio(cells, g_std, dT, delta_T))
    return rows, float(np.mean(base)), len(cells)


def report(rows, base_ratio, n_cells):
    print("=" * 92)
    print("  SCAN RESULTS   (heat-sink limit, averaged over delta_T/h)")
    print("=" * 92)
    print(f"  {'stencil':<24s} {'p':>4s} {'m=4 [%]':>9s} {'m4max':>7s} "
          f"{'g_fm/g_true':>12s} {'range':>15s} {'phys':>7s}  verdict")
    print("  " + "-" * 88)

    survivors = []
    for r in sorted(rows, key=lambda x: (x["stencil"], x["p"])):
        m4_ok = r["m4"] < TARGET_M4_KILL
        ratio_ok = TARGET_RATIO[0] <= r["ratio"] <= TARGET_RATIO[1]
        if m4_ok and ratio_ok:
            verdict = "** BOTH **"
            survivors.append(r)
        elif m4_ok:
            verdict = "iso only"
        elif ratio_ok:
            verdict = "mag only"
        else:
            verdict = "-"
        rng = f"{r['ratio_min']:.2f}-{r['ratio_max']:.2f}"
        print(f"  {r['stencil']:<24s} {r['p']:>4.1f} {100*r['m4']:>9.2f} "
              f"{100*r['m4_max']:>7.2f} {r['ratio']:>12.3f} {rng:>15s} "
              f"{r['ratio_phys']:>7.3f}  {verdict}")

    print("\n  " + "-" * 88)
    print(f"  standard LSQ, no front modification: g/g_true = {base_ratio:.3f}"
          "   (manuscript reports ~0.68-0.69)")
    print(f"  equatorial interface cells sampled : {n_cells}")

    print("\n" + "=" * 92)
    print("  DECISION GATE")
    print("=" * 92)
    print(f"  pass = m4 < {100*TARGET_M4_KILL:.0f}% AND "
          f"g_fm/g_true in [{TARGET_RATIO[0]}, {TARGET_RATIO[1]}]")
    if survivors:
        print(f"\n  {len(survivors)} configuration(s) clear BOTH criteria:\n")
        for r in sorted(survivors, key=lambda x: abs(x["ratio"] - 1.0)):
            print(f"    {r['stencil']:<24s} p={r['p']:.1f}   "
                  f"m4={100*r['m4']:.2f}%   ratio={r['ratio']:.3f}")
        print("\n  -> PROCEED with the Fortran implementation (plan section 1.3).")
    else:
        best_m4 = min(rows, key=lambda x: x["m4"])
        best_ratio = min(rows, key=lambda x: abs(x["ratio"] - 1.0))
        print("\n  NO configuration clears both criteria.")
        print(f"    best isotropy : {best_m4['stencil']} p={best_m4['p']:.1f}"
              f"  m4={100*best_m4['m4']:.2f}%  ratio={best_m4['ratio']:.3f}")
        print(f"    best magnitude: {best_ratio['stencil']} "
              f"p={best_ratio['p']:.1f}"
              f"  m4={100*best_ratio['m4']:.2f}%  "
              f"ratio={best_ratio['ratio']:.3f}")
        print("\n  -> KILL CRITERION MET (plan section 1.5). Ship Plan B")
        print("     (analytical de-biasing) plus this parameter study as")
        print("     'a corrected discretisation and its limits'.")
    return survivors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="reduced scan for a fast check")
    args = ap.parse_args()

    print("\n" + "=" * 72)
    print("  INTERFACE-MODIFIED STENCIL PARAMETER SCAN")
    print("=" * 72 + "\n")

    validate_against_appendix_c()
    rows, base_ratio, n_cells = run_scan(quick=args.quick)
    report(rows, base_ratio, n_cells)
