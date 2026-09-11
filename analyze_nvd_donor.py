#!/usr/bin/env python3
"""
TEST B2: is the normalised donor value ITSELF four-fold anisotropic?

Test B (analyze_cicsam_beta.py) shows that beta_f carries a four-fold mode on
a perfect sphere, and that freezing gamma_f does not remove it.  That locates
the defect upstream of the blend but does not measure the quantity it is
attributed to.  This script measures it.

It reuses Test B's lattice, exact-sphere volume fraction and CICSAM
transcription unchanged, and reports the azimuthal spectrum of

    alfa_d_til = (alfa_d - alfa_u) / (alfa_a - alfa_u)

with NO gamma_f anywhere in the loop -- alfa_d_til is formed before the
blend is reached, so nothing in it depends on theta_f.  If it carries m = 4
at phi_4 = 45 deg on an exact sphere, the attribution in Section 4.5 is a
measurement rather than an inference.

It also evaluates, on the same cells, the analytical prediction of that
section: the width of the volume-fraction ramp of a plane interface,

    W(n) / h = |n_x| + |n_y| + |n_z| = ||n||_1                     Eq. (ramp)

which is pi/2-periodic and even about 45 deg, hence pure m = 4, 8, ... with
phi_4 = 45 deg exactly and 13.3 % in m = 4 about its mean in the equatorial
plane.  Agreement of the two phases, and of the sign, is the check.

Finally it repeats the beta_f measurement with gamma_f frozen at 1, 0.5 and
0, so the "seven-eighths of the mode survives with no orientation
correction" claim has its 3-D numbers in one place.

Usage:  python3 analyze_nvd_donor.py [case.pvtu] [--co 0.2] [--sub 12]
"""

import argparse

import numpy as np

from analyze_anisotropy_attribution import (
    DEFAULT_CASE, load_lattice, fit_bubble, fourier_lsq, M_MAX,
)
from analyze_extraction_error import exact_sphere_vof
from analyze_cicsam_beta import grad_central


def donor_field(vof, grad_a, h, co, axis, ctr, X, Y, Z, gamma_mode="cos2"):
    """alfa_d_til and beta_f on faces normal to `axis`.

    Transcribed from analyze_cicsam_beta.beta_field without alteration except
    that alfa_d_til is also returned, and gamma_f may be overridden:
        gamma_mode = "cos2"  -> min((dotprod/prodmag)**2, 1), as implemented
        gamma_mode = float   -> that constant, removing theta_f entirely
    """
    lo = [slice(None)] * 3
    hi = [slice(None)] * 3
    lo[axis] = slice(0, -1)
    hi[axis] = slice(1, None)
    lo, hi = tuple(lo), tuple(hi)

    fc = [0.5 * (c[lo] + c[hi]) for c in (X, Y, Z)]
    fr = np.sqrt(sum((fc[k] - ctr[k]) ** 2 for k in range(3)))
    v_flux = (fc[axis] - ctr[axis]) / np.where(fr > 0, fr, 1.0)

    pos = v_flux > 0.0

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
        if gamma_mode == "cos2":
            ang = np.where(prodmag > 1e-30,
                           dotprod / np.where(prodmag > 1e-30, prodmag, 1.0),
                           0.0)
            gamma_f = np.minimum(ang ** 2, 1.0)
        else:
            gamma_f = np.full_like(alfa_d_til, float(gamma_mode))
        alfa_f_til = gamma_f * alfa_cbc + (1.0 - gamma_f) * alfa_uq
        db = 1.0 - alfa_d_til
        beta = np.where(np.abs(db) > 1e-12,
                        (alfa_f_til - alfa_d_til)
                        / np.where(np.abs(db) > 1e-12, db, 1.0), 0.0)
    beta = np.clip(np.nan_to_num(beta), 0.0, 1.0)
    valid = good & inrange & (gmag > 1e-30)
    return np.nan_to_num(alfa_d_til), beta, valid


def cell_aggregate(shape, per_face):
    """Average a face quantity onto cells over all six faces, exactly as
    analyze_cicsam_beta does: beta_f (and alfa_d_til) on a single axis is not
    cubically symmetric by construction."""
    acc = np.zeros(shape)
    cnt = np.zeros(shape)
    for axis, (val, valid) in per_face.items():
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[axis] = slice(0, -1)
        hi[axis] = slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        for tgt in (lo, hi):
            acc[tgt] += np.where(valid, val, 0.0)
            cnt[tgt] += valid.astype(float)
    return np.where(cnt > 0, acc / np.where(cnt > 0, cnt, 1.0), np.nan), cnt


def report(label, theta, val, out):
    amps, a0 = fourier_lsq(theta, val, M_MAX)
    # phase of the m = 4 component, in [0, 90)
    c = np.cos(4 * theta)
    sn = np.sin(4 * theta)
    Md = np.column_stack([np.ones_like(theta), c, sn])
    coef, *_ = np.linalg.lstsq(Md, val / np.mean(val), rcond=None)
    phi = np.degrees(np.arctan2(coef[2], coef[1])) / 4.0 % 90.0
    cubic = "yes" if (100 * amps[2] < 0.5 and 100 * amps[6] < 0.5) else "NO"
    print(f"  {label:<42s} {a0:8.4f} {100*amps[2]:6.2f}% "
          f"{100*amps[4]:7.2f}% {100*amps[6]:6.2f}% {phi:7.2f}  {cubic:>6s}")
    out[label] = (100 * amps[4], phi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", nargs="?", default=DEFAULT_CASE)
    ap.add_argument("--co", type=float, default=0.2)
    ap.add_argument("--sub", type=int, default=12)
    a = ap.parse_args()

    d = load_lattice(a.case)
    fit = fit_bubble(d)
    ctr = np.asarray(fit["ctr"], float)
    R, h = fit["R"], d["h"]
    X, Y, Z = d["X"], d["Y"], d["Z"]
    dx, dy, dz = X - ctr[0], Y - ctr[1], Z - ctr[2]
    rr = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    print(f"\n{'='*84}\n  TEST B2: is the normalised donor value itself "
          f"four-fold anisotropic?\n{'='*84}")
    print(f"  R = {R*1e6:.2f} um, h = {h*1e6:.3f} um, R/h = {R/h:.1f}, "
          f"Co = {a.co}")
    print("  alpha field: EXACT sphere volume fraction (no advection history)")

    vof = exact_sphere_vof(d, ctr, R, a.sub, band=3.0)
    grad_a = grad_central(vof, h)

    band = np.abs(rr - R) < 1.5 * h
    theta = np.arctan2(dy, dx)
    print(f"  interface cells in band: {int(band.sum()):,}\n")

    print(f"  {'quantity':<42s} {'mean':>8s} {'m2':>7s} {'m4':>8s} "
          f"{'m6':>7s} {'phi4':>7s}  {'cubic':>6s}")
    print("  " + "-" * 92)

    out = {}

    # --- 1. alfa_d_til itself: no gamma_f anywhere in its formation --------
    per_face = {}
    for axis in range(3):
        adt, _, valid = donor_field(vof, grad_a, h, a.co, axis,
                                    ctr, X, Y, Z)
        per_face[axis] = (adt, valid)
    adt_cell, cnt = cell_aggregate(d["shape"], per_face)
    ok = band & (cnt > 0)
    report("alfa_d_til  (no gamma_f in it at all)", theta[ok], adt_cell[ok], out)

    # --- 2. the analytical prediction, Eq. (ramp) -------------------------
    n_exact = np.stack([dx / rr, dy / rr, dz / rr], -1)
    l1 = np.abs(n_exact).sum(axis=-1)          # W(n)/h
    report("||n||_1 = W(n)/h   [Eq. ramp_width]", theta[ok], l1[ok], out)

    # --- 3. beta_f with gamma_f as implemented and frozen -----------------
    print()
    for label, mode in (("beta_f, gamma_f = cos^2 theta_f (as implemented)",
                         "cos2"),
                        ("beta_f, gamma_f = 1   (purely compressive)", 1.0),
                        ("beta_f, gamma_f = 0.5 (no theta dependence)", 0.5),
                        ("beta_f, gamma_f = 0   (purely diffusive)", 0.0)):
        pf = {}
        for axis in range(3):
            _, b, valid = donor_field(vof, grad_a, h, a.co, axis,
                                      ctr, X, Y, Z, gamma_mode=mode)
            pf[axis] = (b, valid)
        b_cell, cnt_b = cell_aggregate(d["shape"], pf)
        okb = band & (cnt_b > 0)
        report(label, theta[okb], b_cell[okb], out)

    # --- verdict -----------------------------------------------------------
    a4_adt, phi_adt = out["alfa_d_til  (no gamma_f in it at all)"]
    a4_l1, phi_l1 = out["||n||_1 = W(n)/h   [Eq. ramp_width]"]
    a4_impl, _ = out["beta_f, gamma_f = cos^2 theta_f (as implemented)"]
    a4_frozen, _ = out["beta_f, gamma_f = 0.5 (no theta dependence)"]

    print(f"\n  alfa_d_til carries m4 = {a4_adt:.2f}% at phi4 = {phi_adt:.2f} deg,"
          f"\n  formed before any blending, with no theta_f in it.")
    print(f"  The ramp width ||n||_1 predicts phi4 = {phi_l1:.2f} deg "
          f"(m4 = {a4_l1:.2f}%),\n  with no free parameter.")
    print(f"  Phase agreement: {abs(phi_adt - phi_l1):.2f} deg.")
    print(f"\n  beta_f: {a4_impl:.2f}% as implemented, {a4_frozen:.2f}% with "
          f"gamma_f frozen\n  -> {100*a4_frozen/max(a4_impl,1e-9):.0f}% of the "
          f"mode survives with no orientation correction.")
    if abs(phi_adt - 45.0) < 1.0 and a4_adt > 1.0:
        print("\n  => The attribution of Section 4.5 is measured, not inferred.")
    else:
        print("\n  => Does NOT reproduce the expected signature; check before "
              "citing.")


if __name__ == "__main__":
    main()
