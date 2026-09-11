#!/usr/bin/env python3
"""Four-fold azimuthal mode of the BUBBLE SHAPE for the pure Hyper-C ladder.

Every anisotropy figure quoted for the gamma_f = 1 runs so far has been the
body-minus-axis difference from the cubic-invariant regression.  That is a
projection onto one basis function, not a spectrum, and it cannot show whether
the deformation still has the cubic symmetry and the 45 deg phase that the blend
imprints.  This measures the azimuthal spectrum of the front radius directly.

Convention, chosen to match the gradient m = 4 the paper already reports
(`Azimuthal m = 4 amplitude ... with the Fourier modes fitted to the scattered
data`), so that the two are comparable:

  * every front element, not an equatorial slice.  A one-cell slice holds
    200-300 cells and aliases badly -- m = 8 came out at 151% and m = 4 swung
    50% -> 1.2% purely with slice thickness (CLAUDE.md).  These fronts carry
    12k-55k elements.
  * least squares on the scattered (theta, r) points, area-weighted, never
    binned.
  * cubic symmetry is the sanity check: m = 2 and m = 6 must sit near zero
    while m = 4 and m = 8 carry the signal.  If they do not, distrust the run.

The liquid-side mask that matters for the gradient does not apply here: front
elements are the reconstructed interface itself, not cells straddling it, so
there is no vapour-side population to average in.

Run from the paper directory.
"""

import os
import sys

import numpy as np
import pyvista as pv

ROOT = os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling")
M_MAX = 8

# label -> (directory, mesh tag)
RUNS = [
    ("Hyper-C 075", "Data/pure075",       "Scriven-Struct-075"),
    ("Hyper-C 100", "Data/pure100",       "Scriven-Struct-100"),
    ("Hyper-C 125", "Data/pure125",       "Scriven-Struct-125"),
    ("Hyper-C 150", "Data/pure150",       "Scriven-Struct-150"),
    ("blend   075", "Data/gamma-dt1000",  "Scriven-Struct-075"),
    ("blend   100", "Data/gamma-st100",   "Scriven-Struct-100"),
    ("blend   125", "Data/gamma-st125",   "Scriven-Struct-125"),
    ("blend   150", "Data/gamma-st150",   "Scriven-Struct-150"),
]
STEPS = (200, 400, 600, 800, 1000)


def spectrum(path, m_max=M_MAX):
    """Area-weighted Fourier spectrum of the front radius against azimuth.

    Returns (mean radius [um], {m: (amplitude % of mean, phase [deg])}, n)."""
    mesh = pv.read(path)
    if mesh.n_cells == 0:
        raise ValueError(f"{path}: empty front")
    if "ElementCoordinates" in mesh.cell_data:
        c = np.asarray(mesh.cell_data["ElementCoordinates"])
    else:
        c = mesh.cell_centers().points
    if "ElementArea" in mesh.cell_data:
        a = np.asarray(mesh.cell_data["ElementArea"])
    else:
        a = mesh.compute_cell_sizes(length=False, area=True,
                                    volume=False).cell_data["Area"]
    # the front is centred on the origin; verified on every file read here
    c = c - np.average(c, axis=0, weights=a)
    r = np.linalg.norm(c, axis=1)
    th = np.arctan2(c[:, 1], c[:, 0])

    cols = [np.ones_like(th)]
    for k in range(1, m_max + 1):
        cols += [np.cos(k * th), np.sin(k * th)]
    design = np.stack(cols, axis=1)
    w = np.sqrt(a)
    coef, *_ = np.linalg.lstsq(design * w[:, None], r * w, rcond=None)

    a0 = coef[0]
    out = {}
    for k in range(1, m_max + 1):
        ck, sk = coef[2 * k - 1], coef[2 * k]
        amp = np.hypot(ck, sk) / a0 * 100.0
        pha = np.degrees(np.arctan2(sk, ck)) / k
        out[k] = (amp, pha % (360.0 / k))
    return a0 * 1e6, out, len(r)


def main():
    print("=" * 86)
    print("BUBBLE-SHAPE FOUR-FOLD MODE, area-weighted LSQ on all front elements")
    print("=" * 86)
    print(f"{'run':<13}{'ts':>6}{'n_elem':>9}{'<R> um':>9}"
          f"{'m=4 [%]':>10}{'phi_4':>8}{'m=8':>8}{'m=2':>8}{'m=6':>8}  symmetry")
    store = {}
    for lab, d, tag in RUNS:
        for ts in STEPS:
            p = os.path.join(ROOT, d, f"{tag}-front-ts{ts:06d}.pvtu")
            if not os.path.exists(p):
                continue
            try:
                r0, s, n = spectrum(p)
            except Exception as exc:                      # noqa: BLE001
                print(f"{lab:<13}{ts:>6}  FAILED: {exc}")
                continue
            store[(lab, ts)] = (r0, s)
            ok = "ok" if (s[2][0] < 1.0 and s[6][0] < 1.0) else "** m2/m6 HIGH"
            print(f"{lab:<13}{ts:>6}{n:>9}{r0:>9.2f}"
                  f"{s[4][0]:>10.3f}{s[4][1]:>8.2f}{s[8][0]:>8.3f}"
                  f"{s[2][0]:>8.3f}{s[6][0]:>8.3f}  {ok}")
        print()

    print("=" * 86)
    print("AT t = 1 ms -- the ladder")
    print("=" * 86)
    print(f"{'level':<7}{'Hyper-C m=4':>13}{'phi_4':>8}   |{'blend m=4':>11}{'phi_4':>8}"
          f"   | ratio")
    for lv in ("075", "100", "125", "150"):
        h = store.get((f"Hyper-C {lv}", 1000))
        b = store.get((f"blend   {lv}", 1000))
        if not (h and b):
            continue
        hm, bm = h[1][4][0], b[1][4][0]
        print(f"{lv:<7}{hm:>13.3f}{h[1][4][1]:>8.2f}   |{bm:>11.3f}"
              f"{b[1][4][1]:>8.2f}   | {bm/hm:.2f}x")

    print()
    print("Matched-radius comparison (blend interpolated in ts to the Hyper-C radius):")
    for lv in ("075", "100", "125", "150"):
        h = store.get((f"Hyper-C {lv}", 1000))
        if not h:
            continue
        bs = [(store[(f"blend   {lv}", t)][0], store[(f"blend   {lv}", t)][1][4][0])
              for t in STEPS if (f"blend   {lv}", t) in store]
        if len(bs) < 2:
            continue
        rr = np.array([x[0] for x in bs])
        mm = np.array([x[1] for x in bs])
        if not (rr.min() <= h[0] <= rr.max()):
            print(f"  {lv}: Hyper-C R = {h[0]:.2f} um outside the blend's range "
                  f"{rr.min():.1f}-{rr.max():.1f} -- not interpolated")
            continue
        bm = float(np.interp(h[0], rr, mm))
        print(f"  {lv}: at R = {h[0]:.2f} um -- Hyper-C {h[1][4][0]:.3f}% "
              f"against blend {bm:.3f}%, ratio {bm/h[1][4][0]:.2f}x")


if __name__ == "__main__":
    sys.exit(main())
