#!/usr/bin/env python3
"""
T3.1 -- is the four-fold anisotropy a property of the normalised-variable FAMILY?

The paper claims the defect belongs to the NVD family (CICSAM, HRIC, STACS) but has
measured CICSAM only, and generalising from one scheme to a family is what referees
challenge.  This applies the identical exact-sphere harness to three schemes.

Two of the three are transcribed from T-Flows itself, so the comparison is exact:

  CICSAM  Predict_Beta.f90:58-116   gamma_f = min(cos^2 theta, 1)
                                    blends CBC against ULTIMATE-QUICKEST
  STACS   Predict_Beta.f90:148-180  gamma_f = min(cos^4 theta, 1)
                                    blends SUPERBEE against STOIC

HRIC is not in T-Flows and is taken from Muzaferija & Peric's formulation:

    alfa_f = alfa_d_til                     (alfa_d_til < 0 or > 1)
           = 2 alfa_d_til                   (0 <= alfa_d_til < 0.5)
           = 1                              (0.5 <= alfa_d_til <= 1)
    alfa_f <- alfa_f sqrt(cos theta) + alfa_d_til (1 - sqrt(cos theta))

Note the orientation correction enters differently in each: as a blend weight in
CICSAM and STACS (cos^2, cos^4) and as a direct interpolation in HRIC (sqrt(cos)).
If the anisotropy were produced BY the orientation correction, three such different
corrections could not give the same phase and comparable amplitude.

All three are evaluated on the exact volume fraction of a sphere with a prescribed
radial expansion -- no advection history, no phase change, no gradient stencil --
and aggregated over the six faces of each interface cell.

Usage:  python3 analyze_nvd_family.py [--co 0.2] [--sub 12]
"""

import argparse

import numpy as np

from analyze_anisotropy_attribution import (
    DEFAULT_CASE, load_lattice, fit_bubble, fourier_lsq_phase, M_MAX,
)
from analyze_extraction_error import exact_sphere_vof
from analyze_cicsam_beta import grad_central


def limiter(scheme, alfa_d, alfa_a, alfa_u, cosang, co):
    """Face value in normalised variables, then beta_f."""
    den = alfa_a - alfa_u
    good = np.abs(den) > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        adt = np.where(good, (alfa_d - alfa_u) / np.where(good, den, 1.0), 0.0)
        inr = (adt >= 0.0) & (adt <= 1.0)
        cod = min(1.0, co)

        if scheme == "CICSAM":
            cbc = np.where(inr, np.minimum(1.0, adt / max(cod, 1e-30)), adt)
            uq = np.where(inr, np.minimum(cod * adt + (1.0 - cod)
                                          * (6.0 * adt + 3.0) / 8.0, cbc), adt)
            g = np.minimum(cosang ** 2, 1.0)
            aft = g * cbc + (1.0 - g) * uq

        elif scheme == "STACS":
            sup = np.where((adt > 0.0) & (adt < 1.0), 1.0, adt)
            sto = np.where((adt > 0.0) & (adt <= 0.5), 0.5 + 0.5 * adt,
                  np.where((adt > 0.5) & (adt <= 5.0 / 6.0),
                           3.0 / 8.0 + 0.75 * adt,
                  np.where((adt > 5.0 / 6.0) & (adt <= 1.0), 1.0, adt)))
            g = np.minimum(cosang ** 4, 1.0)
            aft = g * sup + (1.0 - g) * sto

        elif scheme == "HRIC":
            base = np.where(inr, np.where(adt < 0.5, 2.0 * adt, 1.0), adt)
            s = np.sqrt(np.clip(np.abs(cosang), 0.0, 1.0))
            aft = base * s + adt * (1.0 - s)

        else:
            raise ValueError(scheme)

        db = 1.0 - adt
        beta = np.where(np.abs(db) > 1e-12,
                        (aft - adt) / np.where(np.abs(db) > 1e-12, db, 1.0), 0.0)
    return np.clip(np.nan_to_num(beta), 0.0, 1.0), (good & inr)


def beta_axis(vof, grad_a, h, co, axis, ctr, X, Y, Z, scheme):
    lo = [slice(None)] * 3; hi = [slice(None)] * 3
    lo[axis] = slice(0, -1); hi[axis] = slice(1, None)
    lo, hi = tuple(lo), tuple(hi)
    fc = [0.5 * (c[lo] + c[hi]) for c in (X, Y, Z)]
    fr = np.sqrt(sum((fc[k] - ctr[k]) ** 2 for k in range(3)))
    pos = ((fc[axis] - ctr[axis]) / np.where(fr > 0, fr, 1.0)) > 0.0
    pick = lambda a, b: np.where(pos, a, b)

    alfa_d = pick(vof[lo], vof[hi])
    alfa_a = pick(vof[hi], vof[lo])
    signo = np.where(pos, 1.0, -1.0)
    g_d = np.stack([pick(grad_a[lo][..., k], grad_a[hi][..., k])
                    for k in range(3)], axis=-1)
    gmag = np.linalg.norm(g_d, axis=-1)
    dotprod = signo * g_d[..., axis] * h
    prodmag = gmag * h
    cosang = np.where(prodmag > 1e-30,
                      dotprod / np.where(prodmag > 1e-30, prodmag, 1.0), 0.0)
    alfa_u = np.clip(alfa_a - 2.0 * dotprod, 0.0, 1.0)
    beta, ok = limiter(scheme, alfa_d, alfa_a, alfa_u, cosang, co)
    return beta, ok & (gmag > 1e-30)


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

    print(f"\n{'='*92}\n  T3.1  the same defect across the normalised-variable "
          f"family\n{'='*92}")
    print(f"  exact sphere volume fraction, R = {R*1e6:.2f} um, "
          f"h = {h*1e6:.3f} um, Co = {a.co}")
    vof = exact_sphere_vof(d, ctr, R, a.sub, band=3.0)
    grad = grad_central(vof, h)
    band = np.abs(rr - R) < 1.5 * h
    print(f"  interface cells: {int(band.sum()):,}\n")
    print("  %-9s %-28s %8s %9s %9s %-11s %7s %7s" %
          ("scheme", "orientation correction", "<beta>", "a4 [%]",
           "phi4[deg]", "max at", "m2 [%]", "m6 [%]"))
    print("  " + "-" * 92)
    desc = {"CICSAM": "blend weight cos^2(theta)",
            "STACS": "blend weight cos^4(theta)",
            "HRIC": "interpolation sqrt(cos theta)"}
    for scheme in ("CICSAM", "STACS", "HRIC"):
        acc = np.zeros(d["shape"]); cnt = np.zeros(d["shape"])
        for axis in range(3):
            b, ok = beta_axis(vof, grad, h, a.co, axis, ctr, X, Y, Z, scheme)
            lo = [slice(None)] * 3; hi = [slice(None)] * 3
            lo[axis] = slice(0, -1); hi[axis] = slice(1, None)
            for tgt in (tuple(lo), tuple(hi)):
                acc[tgt] += np.where(ok, b, 0.0)
                cnt[tgt] += ok.astype(float)
        mb = np.where(cnt > 0, acc / np.where(cnt > 0, cnt, 1.0), np.nan)
        sel = band & (cnt > 0)
        amps, phas, a0 = fourier_lsq_phase(np.arctan2(dy, dx)[sel], mb[sel],
                                           M_MAX)
        where = ("diagonals" if 30 <= phas[4] <= 60 else
                 "axes" if (phas[4] < 15 or phas[4] > 75) else "mixed")
        print("  %-9s %-28s %8.4f %9.4f %9.2f %-11s %7.3f %7.3f" %
              (scheme, desc[scheme], a0, 100 * amps[4], phas[4], where,
               100 * amps[2], 100 * amps[6]))

    print("\n  All three apply the orientation correction differently, at")
    print("  different powers, and in different places in the formula.")


if __name__ == "__main__":
    main()
