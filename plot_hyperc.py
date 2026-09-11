#!/usr/bin/env python3
"""Figure for Section 4.2.1: what declining the blend weight buys, and what it does not.

Two panels, both at t = 1 ms and dt = 1 us, the ladder's operating point:

  (a) the refinement trajectory.  Table 10 lists the radii; the panel shows the
      thing the table cannot, that the two sequences are convergent in opposite
      directions.  The fitted limits R = R_inf - C n^-p are drawn as dashed
      asymptotes so the divergence of the blend is visible as a destination
      rather than inferred from four numbers.

  (b) the bubble itself at that instant: the z = 0 cross-section of the
      reconstructed front at 150^3, at true scale, against the analytical
      Scriven circle.  The slice is taken as a thin equatorial slab of front
      elements sorted by azimuth, the same construction as the existing
      bubble_slice figure.

      The slice plane cuts FACE diagonals, not body diagonals: with n_z = 0 the
      cubic invariant is s = nx^4 + ny^4, which is 1 on an axis and 1/2 at 45
      degrees, so the in-plane contrast is about three quarters of the body-axis
      figure the table quotes.  It is still large enough to see -- 13% for the
      blend against 5% for pure Hyper-C -- which is the point of drawing it at
      true scale rather than magnified.

      One caveat, inherited from plot_blend_slices.py: Scriven predicts a
      volume-equivalent radius while these outlines are front radii, and the two
      differ by the effective interface thickness, which is exactly what the
      blend weight controls.  The paper compares R_front against R_Scriven
      throughout (Table 10 does), so the circle is drawn on that convention and
      the caption says which quantity it is.

  blend    Data/gamma-dt1000, gamma-st{100,125,150}
  Hyper-C  Data/pure{075,100,125,150}
All eight reproduce Table 10's radii exactly.
"""

import os
import glob
import sys

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

ROOT = os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling")
import scriven_reference
R_SCR = scriven_reference.r_scriven_um(1.0e-3)   # analytical radius at 1 ms [um]
N = np.array([75, 100, 125, 150])   # cells per direction
XMAX = 175                          # panel (a) x-limit; the data end at 150
                                    # and extending further is empty space

BLEND = {75: "gamma-dt1000", 100: "gamma-st100", 125: "gamma-st125", 150: "gamma-st150"}
HYPER = {75: "pure075", 100: "pure100", 125: "pure125", 150: "pure150"}


def front(level_dir):
    g = sorted(glob.glob(f"{ROOT}/Data/{level_dir}/*front-ts001000.pvtu"))
    if not g:
        raise FileNotFoundError(level_dir)
    m = pv.read(g[0])
    c = np.asarray(m.cell_data["ElementCoordinates"])
    a = np.asarray(m.cell_data["ElementArea"])
    c = c - np.average(c, axis=0, weights=a)
    r = np.linalg.norm(c, axis=1)
    n = c / r[:, None]
    return r * 1e6, (n ** 4).sum(axis=1), a


def radius(level_dir):
    r, _, a = front(level_dir)
    return float(np.average(r, weights=a))


def outline(level_dir, slab=0.06, nbin=240):
    """Equatorial (z = 0) outline of the reconstructed front, (azimuth, r [um]).

    A slab |z| < slab*R of front elements is binned in azimuth and averaged,
    the construction the existing bubble_slice figure uses.  At 150^3 the slab
    is about four cells deep, so this is not the one-cell band whose aliasing
    Appendix A warns about; it is also only ever read as a curve here, never as
    a mode amplitude.
    """
    m = pv.read(sorted(glob.glob(f"{ROOT}/Data/{level_dir}/*front-ts001000.pvtu"))[0])
    c = np.asarray(m.cell_data["ElementCoordinates"])
    a = np.asarray(m.cell_data["ElementArea"])
    c = c - np.average(c, axis=0, weights=a)
    R = np.average(np.linalg.norm(c, axis=1), weights=a)
    q = c[np.abs(c[:, 2]) < slab * R]
    th = np.arctan2(q[:, 1], q[:, 0])
    rr = np.hypot(q[:, 0], q[:, 1]) * 1e6
    edges = np.linspace(-np.pi, np.pi, nbin + 1)
    idx = np.digitize(th, edges) - 1
    tb, rb = [], []
    for k in range(nbin):
        sel = idx == k
        if sel.sum() == 0:
            continue
        tb.append(0.5 * (edges[k] + edges[k + 1]))
        rb.append(float(rr[sel].mean()))
    tb, rb = np.array(tb), np.array(rb)
    o = np.argsort(tb)
    tb, rb = tb[o], rb[o]
    return np.append(tb, tb[0] + 2 * np.pi), np.append(rb, rb[0])   # close the loop


def fit_limit(n, R):
    """Least squares on R = R_inf - C n^-p, scanning p (three parameters, four points)."""
    best = None
    for p in np.arange(0.4, 4.0, 0.002):
        A = np.column_stack([np.ones_like(n, dtype=float), -n.astype(float) ** -p])
        coef, res, *_ = np.linalg.lstsq(A, R, rcond=None)
        resid = float(np.sum((A @ coef - R) ** 2))
        if best is None or resid < best[0]:
            best = (resid, p, coef[0], coef[1])
    _, p, R_inf, C = best
    return p, R_inf, C


def main():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))

    # ---- (a) refinement trajectory ---------------------------------------
    for lab, table, colour, mark in (("$\\gamma_f = \\cos^2\\theta_f$ (blend)", BLEND, "C3", "s"),
                                     ("$\\gamma_f = 1$ (pure Hyper-C)", HYPER, "C0", "o")):
        R = np.array([radius(table[k]) for k in N])
        e = 100 * (R - R_SCR) / R_SCR
        ax[0].plot(N, e, mark + "-", color=colour, lw=1.6, ms=6, label=lab)
        # No extrapolated limit is drawn or annotated.  The paper (Sec. 4.8,
        # Table 10) reports the four measured levels only; the R_inf / p fit was
        # removed on 4 Sep 2026.  fit_limit() is left defined but is not called.
        print(f"  {lab:34}  " + "  ".join(f"{n}:{v:+6.2f}%" for n, v in zip(N, e)))
    ax[0].axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax[0].set_xscale("log"); ax[0].set_xticks(N); ax[0].set_xticklabels(N)
    ax[0].minorticks_off()
    ax[0].set_xlim(70, XMAX)
    ax[0].set_xlabel("cells per direction $n$")
    ax[0].set_ylabel("$R_\\mathrm{front}/R_\\mathrm{Scriven} - 1$  [%]")
    ax[0].set_title("(a) refinement at $\\Delta t = 1\\,\\mu$s", fontsize=10, loc="left")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8, loc="center right")

    # ---- (b) the bubble at t = 1 ms: z = 0 cross-section, true scale -------
    th = np.linspace(0, 2 * np.pi, 400)
    ax[1].plot(R_SCR * np.cos(th), R_SCR * np.sin(th), "--", color="k", lw=1.3,
               label=f"Scriven, $R = {R_SCR:.1f}\\,\\mu$m")
    for lab, table, colour in (("blend", BLEND, "C3"), ("pure Hyper-C", HYPER, "C0")):
        t, r = outline(table[150])
        ax[1].plot(r * np.cos(t), r * np.sin(t), color=colour, lw=1.8, label=lab)
        inplane = 100 * (r.max() - r.min()) / r.min()
        print(f"  150^3 {lab:14} slice radius {r.min():6.2f}-{r.max():6.2f} um "
              f"({inplane:+.1f}% in plane)")
    lim = 1.28 * R_SCR
    for ang, txt, at in ((0.0, "axis", 0.60), (np.pi / 4, "face diagonal", 0.62)):
        ax[1].plot([0, lim * np.cos(ang)], [0, lim * np.sin(ang)], ":", color="0.6", lw=0.8)
        ax[1].annotate(txt, xy=(at * lim * np.cos(ang), at * lim * np.sin(ang)),
                       fontsize=7.5, color="0.35", ha="center", va="center",
                       rotation=np.degrees(ang),
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax[1].set_aspect("equal")
    ax[1].set_xlim(-lim, lim); ax[1].set_ylim(-lim, lim)
    ax[1].set_xlabel("$x$ [$\\mu$m]"); ax[1].set_ylabel("$y$ [$\\mu$m]")
    ax[1].set_title("(b) bubble at $t = 1$ ms, $z = 0$, structured $150^3$",
                    fontsize=10, loc="left")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7.5, loc="lower left", framealpha=1.0)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{ROOT}/pics/hyperc_ladder.{ext}", dpi=200, bbox_inches="tight")
    print("\nwritten: pics/hyperc_ladder.{pdf,png}")


if __name__ == "__main__":
    sys.exit(main())
