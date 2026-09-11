#!/usr/bin/env python3
"""
Attribution experiment for the m=4 gradient anisotropy on structured meshes.

The manuscript attributes the four-fold anisotropy to the interface-modified
least-squares gradient stencil interacting with Cartesian topology.  Two
independent synthetic models of that stencil (stencil_parameter_scan.py, and
the paper's own analyze_stencil_anisotropy.py) produce only 1-3% anisotropy,
while the simulation shows 10-16%.  This script localises the difference.

The saved "Temperature Gradients [K/m]" is the RAW interface-modified LSQ
gradient: Save_Vtu_Fields.f90:596-599 calls Grad_Variable_With_Front, which
does NOT invoke Extrapolate_Normal_To_Front.  So the field in the VTU is
exactly the quantity the synthetic models compute, and the comparison is
like-for-like.

Method
------
The T-Flows gradient depends on exactly two data: the temperature field, and
the interface geometry (which faces cross, and where x_f lands).  Each is
replaced independently by an idealised counterpart:

               interface = ACTUAL (alpha)      interface = IDEAL (sphere)
  T = ACTUAL      (1) reproduces the VTU          (3) isolates T field
  T = IDEAL erf   (2) isolates interface          (4) reproduces the scan

Case 1 validates the reimplementation against the saved field.  Case 4 should
land near the synthetic scan's 1-3%.  Whichever of (2) or (3) retains the
anisotropy names the mechanism.

Replicates Calculate_Grad_Matrix_With_Front.f90 and
Grad_Component_No_Refresh_With_Front.f90 exactly, including the fallback to
the standard gradient matrix when the front-modified Jacobian is <= 0.

Usage:
  python3 analyze_anisotropy_attribution.py [case.pvtu]
"""

import sys

import numpy as np
import pyvista as pv
from scipy.optimize import curve_fit
from scipy.special import erf

DEFAULT_CASE = ("/home/jan/archive/Scriven-Merlin/Scriven-Struct-125-smallDt/"
                "bubble-ts001000.pvtu")
SHELL_CELLS = 8          # analyse a shell of +-8h about the interface
M_MAX = 8


# ======================================================================
#  Lattice
# ======================================================================
def load_lattice(fname):
    """Load a uniform-Cartesian VTU onto a regular (nx,ny,nz) index grid."""
    mesh = pv.read(fname)
    cc = mesh.cell_centers().points
    vol = mesh.cell_data["Grid Cell Volume [m^3]"]
    h = float(np.cbrt(np.median(vol)))

    origin = cc.min(axis=0)
    idx = np.rint((cc - origin) / h).astype(np.int64)
    shape = tuple(idx.max(axis=0) + 1)
    if int(np.prod(shape)) != len(cc):
        sys.exit(f"mesh is not a full uniform lattice: "
                 f"{np.prod(shape)} lattice sites vs {len(cc)} cells")

    lin = np.ravel_multi_index(idx.T, shape)

    def to_grid(arr):
        out = np.empty(np.prod(shape) if arr.ndim == 1
                       else (np.prod(shape), arr.shape[1]), dtype=np.float64)
        out[lin] = arr
        return out.reshape(shape if arr.ndim == 1 else shape + (arr.shape[1],))

    data = {
        "h": h, "shape": shape, "origin": origin,
        "T": to_grid(mesh.cell_data["Temperature [K]"].astype(np.float64)),
        "vof": to_grid(mesh.cell_data["Vof Sharp [1]"].astype(np.float64)),
        "grad_saved": to_grid(
            mesh.cell_data["Temperature Gradients [K/m]"].astype(np.float64)),
        "vol": to_grid(vol.astype(np.float64)),
    }
    ax = [origin[k] + h * np.arange(shape[k]) for k in range(3)]
    data["X"], data["Y"], data["Z"] = np.meshgrid(*ax, indexing="ij")
    return data


# ======================================================================
#  Bubble and thermal-layer fit
# ======================================================================
def fit_bubble(d):
    """Vapour centroid, equivalent radius, T_sat, superheat, delta_T."""
    vof, vol = d["vof"], d["vol"]
    P = np.stack([d["X"], d["Y"], d["Z"]], axis=-1)

    vap = vof < 0.5
    w = vol[vap]
    ctr = np.average(P[vap], axis=0, weights=w)

    V_vap = float((vol * (1.0 - vof)).sum())
    R = (3.0 * V_vap / (4.0 * np.pi)) ** (1.0 / 3.0)

    T_sat = float(np.median(d["T"][vof < 0.05]))
    T_inf = float(np.percentile(d["T"], 99.9))
    dT = T_inf - T_sat

    # Fit delta_T on the liquid side, out to 6 cells from the interface
    r = np.linalg.norm(P - ctr, axis=-1)
    liq = (vof > 0.95) & (r > R) & (r < R + 6.0 * d["h"])
    x, y = (r[liq] - R).ravel(), (d["T"][liq] - T_sat).ravel()

    def model(xx, amp, dl):
        return amp * erf(xx / dl)

    try:
        popt, _ = curve_fit(model, x, y, p0=[dT, 3.0 * d["h"]], maxfev=20000)
        amp, delta_T = float(popt[0]), float(abs(popt[1]))
    except Exception:
        amp, delta_T = dT, 3.0 * d["h"]

    return dict(ctr=ctr, R=R, T_sat=T_sat, dT=amp, delta_T=delta_T,
                r=r, P=P)


# ======================================================================
#  Interface-modified LSQ gradient  (mirrors T-Flows)
# ======================================================================
def modified_gradient(d, fit, T, use_ideal_interface):
    """Raw interface-modified LSQ gradient on the whole lattice.

    Replicates Calculate_Grad_Matrix_With_Front +
    Grad_Component_No_Refresh_With_Front, unweighted, with the
    jac <= 0 fallback to the standard gradient matrix.
    """
    h, shape = d["h"], d["shape"]
    vof, r, R, T_sat = d["vof"], fit["r"], fit["R"], fit["T_sat"]

    # Symmetric 3x3 stored as T-Flows does: 1=xx 2=yy 3=zz 4=xy 5=xz 6=yz
    G = [np.zeros(shape) for _ in range(6)]
    G0 = [np.zeros(shape) for _ in range(6)]     # standard, for fallback
    b = [np.zeros(shape) for _ in range(3)]

    IJ = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 0, 1), (4, 0, 2), (5, 1, 2)]

    for axis in range(3):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis], hi[axis] = slice(0, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)

        # full connection vector c1 -> c2 (axis-aligned, length h)
        conn = np.zeros(3)
        conn[axis] = h

        a1, a2 = vof[lo], vof[hi]
        T1, T2 = T[lo], T[hi]

        if use_ideal_interface:
            s1, s2 = r[lo] - R, r[hi] - R
            crosses = (s1 * s2) < 0.0
            # exact intersection of the axis-aligned segment with the sphere
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(crosses, s1 / (s1 - s2), 0.0)
        else:
            crosses = ((a1 - 0.5) * (a2 - 0.5)) < 0.0
            den = np.abs(a1 - a2)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(crosses & (den > 1e-15),
                                np.abs(a1 - 0.5) / np.where(den > 1e-15,
                                                            den, 1.0), 0.0)
        frac = np.clip(np.nan_to_num(frac), 0.0, 1.0)

        # displacement vectors and value differences on each side
        f = frac[..., None]
        r1 = np.where(crosses[..., None], f * conn, conn)
        r2 = np.where(crosses[..., None], (1.0 - f) * conn, conn)
        d1 = np.where(crosses, T_sat - T1, T2 - T1)
        d2 = np.where(crosses, T2 - T_sat, T2 - T1)

        for k, i, j in IJ:
            G[k][lo] += r1[..., i] * r1[..., j]
            G[k][hi] += r2[..., i] * r2[..., j]
            G0[k][lo] += conn[i] * conn[j]
            G0[k][hi] += conn[i] * conn[j]
        for i in range(3):
            b[i][lo] += d1 * r1[..., i]
            b[i][hi] += d2 * r2[..., i]

    # T-Flows' explicit determinant, and its fallback
    def jac_of(g):
        return (g[0] * g[1] * g[2] - g[0] * g[5] ** 2 - g[3] ** 2 * g[2]
                + 2.0 * g[3] * g[4] * g[5] - g[4] ** 2 * g[1])

    jac = jac_of(G)
    bad = jac <= 0.0
    n_fallback = int(bad.sum())
    for k in range(6):
        G[k] = np.where(bad, G0[k], G[k])
    jac = jac_of(G)

    inv = [
        (G[1] * G[2] - G[5] ** 2) / jac,
        (G[0] * G[2] - G[4] ** 2) / jac,
        (G[0] * G[1] - G[3] ** 2) / jac,
        -(G[3] * G[2] - G[4] * G[5]) / jac,
        (G[3] * G[5] - G[4] * G[1]) / jac,
        -(G[0] * G[5] - G[3] * G[4]) / jac,
    ]
    MAP = [[0, 3, 4], [3, 1, 5], [4, 5, 2]]
    grad = np.stack(
        [sum(inv[MAP[i][j]] * b[j] for j in range(3)) for i in range(3)],
        axis=-1)
    return grad, n_fallback


# ======================================================================
#  Metrics
# ======================================================================
def fourier_lsq(theta, g, m_max=M_MAX):
    cols = [np.ones_like(theta)]
    for m in range(1, m_max + 1):
        cols += [np.cos(m * theta), np.sin(m * theta)]
    coef, *_ = np.linalg.lstsq(np.stack(cols, axis=1), g, rcond=None)
    a0 = coef[0]
    return ({m: float(np.hypot(coef[2 * m - 1], coef[2 * m]) / abs(a0))
             for m in range(1, m_max + 1)}, float(a0))


def fourier_lsq_phase(theta, g, m_max=M_MAX):
    """As fourier_lsq, but also returns the phase of each mode.

    Fits  g(theta) = a0 * [ 1 + sum_m A_m cos( m (theta - phi_m) ) ]  and
    returns ({m: A_m}, {m: phi_m in degrees}, a0).  The phase of mode m is
    defined modulo 360/m degrees; for m = 4 that is modulo 90, so

        phi_4 ~ 0   -> maxima on the coordinate axes
        phi_4 ~ 45  -> maxima on the face diagonals

    Added alongside fourier_lsq rather than replacing it, so that the
    bit-exact validation of the existing attribution path is untouched.
    """
    cols = [np.ones_like(theta)]
    for m in range(1, m_max + 1):
        cols += [np.cos(m * theta), np.sin(m * theta)]
    coef, *_ = np.linalg.lstsq(np.stack(cols, axis=1), g, rcond=None)
    a0 = coef[0]
    amp, pha = {}, {}
    for m in range(1, m_max + 1):
        c, s_ = coef[2 * m - 1], coef[2 * m]
        amp[m] = float(np.hypot(c, s_) / abs(a0))
        ph = np.degrees(np.arctan2(s_, c)) / m
        per = 360.0 / m
        ph = ph % per
        if per - ph < 1e-9:      # snap the wrap-around to 0
            ph = 0.0
        pha[m] = float(ph)
    return amp, pha, float(a0)


def metrics_phase(d, fit, grad, mask):
    """As metrics(), but returns Fourier phases too (T0.2)."""
    q = (fit["P"] - fit["ctr"])[mask]
    rr = np.linalg.norm(q, axis=-1)
    nrm = q / rr[:, None]
    g_n = np.abs(np.sum(grad[mask] * nrm, axis=-1))
    theta = np.arctan2(q[:, 1], q[:, 0])
    amps, phas, a0 = fourier_lsq_phase(theta, g_n)
    return amps, phas, a0, g_n


def metrics(d, fit, grad, mask):
    q = (fit["P"] - fit["ctr"])[mask]
    rr = np.linalg.norm(q, axis=-1)
    nrm = q / rr[:, None]
    g_n = np.abs(np.sum(grad[mask] * nrm, axis=-1))
    theta = np.arctan2(q[:, 1], q[:, 0])
    amps, a0 = fourier_lsq(theta, g_n)
    return amps, a0, g_n


# ======================================================================
#  Main
# ======================================================================
def main(fname):
    print("=" * 78)
    print("  ANISOTROPY ATTRIBUTION")
    print("=" * 78)
    print(f"  case: {fname}\n")

    d = load_lattice(fname)
    fit = fit_bubble(d)
    h = d["h"]
    print(f"  lattice {d['shape']}   h = {h*1e6:.4f} um")
    print(f"  bubble  R = {fit['R']*1e6:.2f} um   centre "
          f"{np.round(fit['ctr']*1e6, 3)} um")
    print(f"  T_sat = {fit['T_sat']:.4f} K   superheat = {fit['dT']:.4f} K")
    print(f"  fitted delta_T = {fit['delta_T']*1e6:.3f} um "
          f"(= {fit['delta_T']/h:.2f} h)")
    g_ex = 2.0 * fit["dT"] / (np.sqrt(np.pi) * fit["delta_T"])
    print(f"  exact interface gradient = {g_ex:,.0f} K/m\n")

    # Ideal erf temperature field about the fitted sphere
    x = fit["r"] - fit["R"]
    T_ideal = np.where(x > 0.0,
                       fit["T_sat"] + fit["dT"] * erf(x / fit["delta_T"]),
                       fit["T_sat"])

    # ---- analysis sets -------------------------------------------------
    # The manuscript selects 0.3 < alpha < 0.7, which straddles BOTH
    # phases.  In Scriven the vapour is isothermal at T_sat, so its cells
    # carry a near-zero gradient; averaging them with liquid-side cells
    # halves the mean and injects spurious angular structure, because the
    # vapour/liquid census per angular bin follows the lattice.  The
    # physically meaningful set is the liquid-side interface band, which
    # is where the mass transfer gradient is actually taken.
    vof = d["vof"]

    def liquid_band(use_ideal):
        """Liquid-side interface band, consistent with the interface
        definition the gradient uses.  Mixing the two (actual-alpha mask
        with an ideal-sphere gradient) selects mismatched cells and
        manufactures artefacts, so each case gets its own band."""
        is_vap = (fit["r"] < fit["R"]) if use_ideal else (vof < 0.5)
        nb = np.zeros(d["shape"], dtype=bool)
        for axis in range(3):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis], hi[axis] = slice(0, -1), slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            nb[lo] |= is_vap[hi]
            nb[hi] |= is_vap[lo]
        return (~is_vap) & nb

    mask_paper = (vof > 0.3) & (vof < 0.7)
    mask = liquid_band(False)

    n_vap_in_paper = int((mask_paper & (vof < 0.5)).sum())
    print(f"  manuscript mask 0.3<alpha<0.7 : {int(mask_paper.sum())} cells"
          f"  ({n_vap_in_paper} of them VAPOUR-side, "
          f"{100*n_vap_in_paper/mask_paper.sum():.0f}%)")
    print(f"  liquid-side band, actual alpha: {int(mask.sum())} cells")
    print(f"  liquid-side band, ideal sphere: "
          f"{int(liquid_band(True).sum())} cells\n")

    cases = [
        ("1  T actual  / interface actual", d["T"], False),
        ("2  T IDEAL   / interface actual", T_ideal, False),
        ("3  T actual  / interface IDEAL ", d["T"], True),
        ("4  T IDEAL   / interface IDEAL ", T_ideal, True),
    ]

    print("=" * 78)
    print("  LIQUID-SIDE INTERFACE BAND")
    print("=" * 78)
    print(f"  {'case':<34s} {'m2':>6s} {'m4':>7s} {'m6':>6s} {'m8':>6s} "
          f"{'<|gn|>':>9s} {'/exact':>7s}")
    print("  " + "-" * 74)

    results = {}
    for label, Tf, ideal in cases:
        grad, nfb = modified_gradient(d, fit, Tf, ideal)
        amps, a0, g_n = metrics(d, fit, grad, liquid_band(ideal))
        results[label] = (amps, a0, grad, nfb)
        print(f"  {label:<34s} {100*amps[2]:6.2f} {100*amps[4]:7.2f} "
              f"{100*amps[6]:6.2f} {100*amps[8]:6.2f} {a0:9.0f} "
              f"{a0/g_ex:7.3f}")

    amps_s, a0_s, g_s = metrics(d, fit, d["grad_saved"], mask)
    print("  " + "-" * 74)
    print(f"  {'VTU saved field (reference)':<34s} {100*amps_s[2]:6.2f} "
          f"{100*amps_s[4]:7.2f} {100*amps_s[6]:6.2f} {100*amps_s[8]:6.2f} "
          f"{a0_s:9.0f} {a0_s/g_ex:7.3f}")

    # Same quantities under the manuscript's two-phase mask, to show what
    # the published numbers are actually measuring.
    print("\n" + "=" * 78)
    print("  SAME FIELDS UNDER THE MANUSCRIPT MASK (0.3 < alpha < 0.7)")
    print("=" * 78)
    print(f"  {'case':<34s} {'m2':>6s} {'m4':>7s} {'m6':>6s} {'m8':>6s} "
          f"{'<|gn|>':>9s} {'/exact':>7s}")
    print("  " + "-" * 74)
    for label, _, _ in cases:
        grad = results[label][2]
        amps_p, a0_p, _ = metrics(d, fit, grad, mask_paper)
        print(f"  {label:<34s} {100*amps_p[2]:6.2f} {100*amps_p[4]:7.2f} "
              f"{100*amps_p[6]:6.2f} {100*amps_p[8]:6.2f} {a0_p:9.0f} "
              f"{a0_p/g_ex:7.3f}")
    amps_ps, a0_ps, _ = metrics(d, fit, d["grad_saved"], mask_paper)
    print("  " + "-" * 74)
    print(f"  {'VTU saved field (reference)':<34s} {100*amps_ps[2]:6.2f} "
          f"{100*amps_ps[4]:7.2f} {100*amps_ps[6]:6.2f} {100*amps_ps[8]:6.2f} "
          f"{a0_ps:9.0f} {a0_ps/g_ex:7.3f}")

    # --- validation -----------------------------------------------------
    print("\n" + "=" * 78)
    print("  VALIDATION of the reimplementation (case 1 vs the saved field)")
    print("=" * 78)
    g1 = results[cases[0][0]][2]
    a = np.linalg.norm(g1[mask], axis=-1)
    bmag = np.linalg.norm(d["grad_saved"][mask], axis=-1)
    ok = bmag > 0
    rel = np.abs(a[ok] - bmag[ok]) / bmag[ok]
    corr = np.corrcoef(a[ok], bmag[ok])[0, 1]
    print(f"  correlation           : {corr:.6f}")
    print(f"  median |rel. error|   : {100*np.median(rel):.3f}%")
    print(f"  90th pct |rel. error| : {100*np.percentile(rel, 90):.3f}%")
    print(f"  singular-Jacobian fallbacks (case 1): "
          f"{results[cases[0][0]][3]}")
    good = corr > 0.99 and np.median(rel) < 0.05
    print(f"\n  {'PASS' if good else 'SUSPECT'}: reimplementation "
          f"{'matches' if good else 'does NOT match'} T-Flows.")

    # --- verdict --------------------------------------------------------
    m4 = {k: v[0][4] for k, v in results.items()}
    keys = [c[0] for c in cases]
    print("\n" + "=" * 78)
    print("  ATTRIBUTION")
    print("=" * 78)
    if not good:
        print("  Reimplementation did not validate; verdict withheld.")
        return
    base, mixT, mixI, ideal = (m4[keys[0]], m4[keys[1]],
                               m4[keys[2]], m4[keys[3]])
    r_base = results[keys[0]][1] / g_ex
    r_ideal = results[keys[3]][1] / g_ex

    print("  Self-consistent configurations (both inputs from the same state):")
    print(f"    case 4  ideal  T + ideal  interface   m4 = {100*ideal:6.2f}%"
          f"   <|gn|>/exact = {r_ideal:.3f}")
    print(f"    case 1  actual T + actual interface   m4 = {100*base:6.2f}%"
          f"   <|gn|>/exact = {r_base:.3f}")
    print("\n  Mixed configurations (one input idealised, one not -- these")
    print("  also carry the bubble's departure from a sphere, so read them")
    print("  as an upper bound on each component, not a clean split):")
    print(f"    case 2  ideal  T + actual interface   m4 = {100*mixT:6.2f}%")
    print(f"    case 3  actual T + ideal  interface   m4 = {100*mixI:6.2f}%")
    print()

    if ideal < 0.02 and base > 2.0 * ideal:
        print("  -> The STENCIL IS NOT THE SOURCE.  Applied to consistent")
        print("     idealised inputs it is isotropic to "
              f"{100*ideal:.2f}% and accurate to")
        print(f"     {abs(1-r_ideal)*100:.1f}%.  The anisotropy enters through the "
              "SIMULATION STATE")
        print("     -- the alpha-derived interface position and the "
              "co-adapted")
        print("     temperature field -- which the stencil then differentiates")
        print("     faithfully.")
        if mixT > 2.0 * base and mixI > 2.0 * base:
            print()
            print("     Note that each mixed case exceeds the baseline: the two")
            print("     state errors are ANTI-correlated and largely cancel.")
            print("     The measured anisotropy is a residual after that")
            print("     cancellation, not the primary effect.")
    else:
        print("  -> Stencil contribution is not negligible; see the table.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CASE)
