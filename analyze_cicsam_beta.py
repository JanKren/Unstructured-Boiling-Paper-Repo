#!/usr/bin/env python3
"""
TEST B: does CICSAM compress a PERFECT sphere anisotropically?

The advection attribution rests on a step that has not been tested: that the
mesh-aligned error enters through CICSAM's flux limiter.  This script tests
that mechanism directly, in the only way that has proved reliable in this
investigation -- by substitution rather than by inference.

It builds the EXACT volume fraction of a sphere (no advection history, no
phase change, no gradient stencil in the loop), prescribes a purely radial
expansion velocity, and evaluates the CICSAM limiter exactly as
Vof_Mod/Core/Predict_Beta.f90 does:

    dotprod    = signo * ( grad(alpha)_donor . d_s )
    alfa_u     = clamp( alfa_a - 2*dotprod , 0, 1 )        ! virtual upwind
    alfa_d_til = (alfa_d - alfa_u) / (alfa_a - alfa_u)
    alfa_cbc   = min(1, alfa_d_til / cod)                  ! compressive
    alfa_uq    = min(cod*alfa_d_til + (1-cod)*(6*alfa_d_til+3)/8, alfa_cbc)
    gamma_f    = min( (dotprod / (|grad alpha| * d_s))**2 , 1 )
    alfa_f_til = gamma_f*alfa_cbc + (1 - gamma_f)*alfa_uq
    beta_f     = clamp( (alfa_f_til - alfa_d_til)/(1 - alfa_d_til) , 0, 1 )

beta_f = 0 is upwind (diffusive), beta_f = 1 is downwind (compressive), so
beta_f IS the amount of interface sharpening each face receives.  If beta_f
carries a four-fold azimuthal mode on a perfect sphere, CICSAM sharpens
grid-aligned directions more than diagonal ones from the very first step,
and the interface must drift towards the diagonals.  If it does not, the
advection attribution needs rethinking.

The limiter is evaluated with three normals for gamma_f:
    raw alpha        -- what T-Flows does now
    smoothed alpha   -- Vof % nx, already computed each step for curvature
    exact radial     -- the ideal, as a floor on what any normal could give

Usage:  python3 analyze_cicsam_beta.py [case.pvtu] [--co 0.2] [--sub 12]
"""

import argparse
import os

import numpy as np

from analyze_anisotropy_attribution import (
    DEFAULT_CASE, load_lattice, fit_bubble, fourier_lsq, fourier_lsq_phase,
    M_MAX,
)
from analyze_extraction_error import exact_sphere_vof


def grad_central(f, h):
    g = np.empty(f.shape + (3,))
    for ax in range(3):
        g[..., ax] = np.gradient(f, h, axis=ax)
    return g


def smooth_vof(f, n_iter):
    s = f.copy()
    for _ in range(n_iter):
        acc = np.zeros_like(s)
        for ax in range(3):
            acc += np.roll(s, 1, axis=ax) + np.roll(s, -1, axis=ax)
        s = acc / 6.0
    return s


def beta_field(vof, grad_a, h, co, axis, ctr, X, Y, Z, gamma_fix=None):
    """CICSAM beta_f on faces normal to `axis`, transcribed from
    Predict_Beta.f90.  The donor is chosen by the sign of the prescribed
    radial expansion flux, exactly as the solver chooses it from v_flux."""
    lo = [slice(None)] * 3
    hi = [slice(None)] * 3
    lo[axis] = slice(0, -1)
    hi[axis] = slice(1, None)
    lo, hi = tuple(lo), tuple(hi)

    # face-centre radial direction -> sign of the outward expansion flux
    fc = [0.5 * (c[lo] + c[hi]) for c in (X, Y, Z)]
    fr = np.sqrt(sum((fc[k] - ctr[k]) ** 2 for k in range(3)))
    v_flux = (fc[axis] - ctr[axis]) / np.where(fr > 0, fr, 1.0)   # \propto u.n

    pos = v_flux > 0.0                      # donor is the lower cell
    def pick(arr_lo, arr_hi):
        return np.where(pos, arr_lo, arr_hi)

    alfa_d = pick(vof[lo], vof[hi])
    alfa_a = pick(vof[hi], vof[lo])
    signo = np.where(pos, 1.0, -1.0)

    g_d = np.stack([pick(grad_a[lo][..., k], grad_a[hi][..., k])
                    for k in range(3)], axis=-1)
    gmag = np.linalg.norm(g_d, axis=-1)

    dotprod = signo * g_d[..., axis] * h
    prodmag = gmag * h

    alfa_u = np.clip(alfa_a - 2.0 * dotprod, 0.0, 1.0)
    den = alfa_a - alfa_u
    good = np.abs(den) > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        alfa_d_til = np.where(good, (alfa_d - alfa_u)
                              / np.where(good, den, 1.0), 0.0)
        cod = min(1.0, co)
        inrange = (alfa_d_til >= 0.0) & (alfa_d_til <= 1.0)
        alfa_cbc = np.where(inrange,
                            np.minimum(1.0, alfa_d_til / max(cod, 1e-30)),
                            alfa_d_til)
        alfa_uq = np.where(inrange,
                           np.minimum(cod * alfa_d_til + (1.0 - cod)
                                      * (6.0 * alfa_d_til + 3.0) / 8.0,
                                      alfa_cbc),
                           alfa_d_til)
        ang = np.where(prodmag > 1e-30,
                       dotprod / np.where(prodmag > 1e-30, prodmag, 1.0), 0.0)
        gamma_f = (np.minimum(ang ** 2, 1.0) if gamma_fix is None
                   else np.full_like(ang, gamma_fix))
        alfa_f_til = gamma_f * alfa_cbc + (1.0 - gamma_f) * alfa_uq
        db = 1.0 - alfa_d_til
        beta = np.where(np.abs(db) > 1e-12,
                        (alfa_f_til - alfa_d_til)
                        / np.where(np.abs(db) > 1e-12, db, 1.0), 0.0)
    beta = np.clip(np.nan_to_num(beta), 0.0, 1.0)
    valid = good & inrange & (gmag > 1e-30)
    return beta, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default=DEFAULT_CASE)
    ap.add_argument("--co", type=float, default=0.2, help="Courant number")
    ap.add_argument("--sub", type=int, default=12)
    ap.add_argument("--fig", default=None,
                    help="write beta_f vs azimuth to this path")
    a = ap.parse_args()

    d = load_lattice(a.case)
    fit = fit_bubble(d)
    ctr = np.asarray(fit["ctr"], float)
    R, h = fit["R"], d["h"]
    X, Y, Z = d["X"], d["Y"], d["Z"]
    dx, dy, dz = X - ctr[0], Y - ctr[1], Z - ctr[2]
    rr = np.sqrt(dx**2 + dy**2 + dz**2)

    print(f"\n{'='*78}\n  TEST B: does CICSAM compress a PERFECT sphere "
          f"anisotropically?\n{'='*78}")
    print(f"  R = {R*1e6:.2f} um, h = {h*1e6:.3f} um, R/h = {R/h:.1f}, "
          f"Co = {a.co}")
    print("  alpha field: EXACT sphere volume fraction (no advection history)")

    vof = exact_sphere_vof(d, ctr, R, a.sub, band=3.0)

    n_exact = np.stack([dx / rr, dy / rr, dz / rr], -1)
    gmag_ref = np.linalg.norm(grad_central(vof, h), axis=-1)

    graw = grad_central(vof, h)
    variants = {
        "raw alpha gradient (T-Flows now)":      (graw, None),
        "smoothed alpha, 2 cycles":              (grad_central(smooth_vof(vof, 2), h), None),
        "exact radial normal (ideal floor)":     (n_exact * gmag_ref[..., None], None),
        "gamma_f forced = 1  (pure CBC)":        (graw, 1.0),
        "gamma_f forced = 0  (pure UQ)":         (graw, 0.0),
        "gamma_f forced = 0.5 (no orientation)": (graw, 0.5),
    }

    # Aggregate the compression each interface CELL receives over all six
    # faces.  beta_f on a single axis is not cubically symmetric by
    # construction, so m2 and m6 would be non-zero for a trivial reason; the
    # per-cell aggregate is the physically meaningful quantity and restores
    # the symmetry that acts as the sanity check.
    band = np.abs(rr - R) < 1.5 * h
    theta_all = np.arctan2(dy, dx)[band]
    print(f"  interface cells in band: {int(band.sum()):,}\n")
    print(f"  {'gamma_f formed from':<40s} {'<beta>':>8s} {'a4':>9s} "
          f"{'phi4[deg]':>8s}  {'max at':<10s} {'m2':>6s} {'m6':>7s} "
          f"{'m8':>7s}  {'cub':>4s}")
    print("  " + "-" * 108)
    res = {}
    curves = {}
    for label, (g, gfix) in variants.items():
        acc = np.zeros(d["shape"])
        cnt = np.zeros(d["shape"])
        for axis in range(3):
            b, valid = beta_field(vof, g, h, a.co, axis, ctr, X, Y, Z,
                                  gamma_fix=gfix)
            lo = [slice(None)] * 3; hi = [slice(None)] * 3
            lo[axis] = slice(0, -1); hi[axis] = slice(1, None)
            lo, hi = tuple(lo), tuple(hi)
            for tgt in (lo, hi):          # both cells share the face
                acc[tgt] += np.where(valid, b, 0.0)
                cnt[tgt] += valid.astype(float)
        mean_b = np.where(cnt > 0, acc / np.where(cnt > 0, cnt, 1.0), np.nan)
        ok = band & (cnt > 0)
        amps, phas, a0 = fourier_lsq_phase(np.arctan2(dy, dx)[ok],
                                           mean_b[ok], M_MAX)
        res[label] = (100 * amps[4], phas[4])
        th = np.arctan2(dy, dx)[ok]
        eq = np.abs(np.arcsin(np.clip((Z[ok] - ctr[2])
                                      / np.maximum(rr[ok], 1e-30), -1, 1))) < np.radians(20)
        edges = np.linspace(-np.pi, np.pi, 73)
        idx = np.digitize(th[eq], edges) - 1
        prof = np.array([mean_b[ok][eq][idx == b].mean() if (idx == b).any()
                         else np.nan for b in range(72)])
        curves[label] = (0.5 * (edges[1:] + edges[:-1]), prof / a0)
        cubic = "yes" if (100*amps[2] < 0.5 and 100*amps[6] < 0.5) else "NO"
        where = ("diagonals" if 30.0 <= phas[4] <= 60.0 else
                 "axes" if (phas[4] < 15.0 or phas[4] > 75.0) else "mixed")
        print(f"  {label:<40s} {a0:8.4f} {100*amps[4]:8.4f}% "
              f"{phas[4]:8.2f}  {where:<10s} {100*amps[2]:6.3f}% "
              f"{100*amps[6]:6.3f}% {100*amps[8]:6.3f}%  {cubic:>4s}")

    k = list(res)
    print()
    print("  beta_f = 0 upwind (diffusive), 1 downwind (compressive).")
    print("  phi4 ~ 45 deg : compression MAXIMAL on the face diagonals")
    print("  phi4 ~  0 deg : compression MAXIMAL on the coordinate axes")
    print()
    for lab in k:
        a4, p4 = res[lab]
        print(f"    {lab:<40s} a4 = {a4:7.4f}%   phi4 = {p4:6.2f} deg")

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 11, 'font.family': 'serif',
                             'savefig.dpi': 300, 'axes.grid': False,
                             'lines.linewidth': 1.5})
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        style = {k[0]: ('C0', '-'), k[1]: ('C1', '--'), k[2]: ('C2', ':'),
                 k[3]: ('C3', '-'), k[4]: ('C4', '--'), k[5]: ('C5', ':')}
        for lab, (thc, pr) in curves.items():
            c, ls = style.get(lab, ('k', '-'))
            ax.plot(np.degrees(thc), pr, ls, color=c,
                    label=f"{lab}  ($a_4$={res[lab][0]:.2f}%)")
        for x in (-135, -45, 45, 135):
            ax.axvline(x, color='0.75', lw=0.8, zorder=0)
        for x in (-180, -90, 0, 90, 180):
            ax.axvline(x, color='0.9', lw=0.8, ls='-', zorder=0)
        ax.set_xlim(-180, 180); ax.set_xticks(range(-180, 181, 45))
        ax.set_xlabel(r"azimuth $\theta$ [deg]")
        ax.set_ylabel(r"$\beta_f / \langle\beta_f\rangle$")
        ax.set_title("CICSAM compression factor on an exact sphere "
                     r"(grey: axes, dark: $\langle110\rangle$ diagonals)")
        ax.legend(fontsize=7.5, ncol=2, loc='upper center',
                  bbox_to_anchor=(0.5, -0.18))
        fig.tight_layout()
        os.makedirs(os.path.dirname(a.fig), exist_ok=True)
        fig.savefig(a.fig, bbox_inches='tight')
        print(f"\n  figure -> {a.fig}")


if __name__ == "__main__":
    main()
