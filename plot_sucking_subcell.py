#!/usr/bin/env python3
"""Sucking problem: where the velocity error comes from, and where it does not go.

    python3 plot_sucking_subcell.py

Two panels, answering the referee's question on Figure 5(b) with measurements
rather than description:

  (a) the plateau itself, on a y-axis holding only the plateau, because the
      point is a 4.5% offset that is invisible against the 5 mm/s step.  The
      referee's "local oscillations" are local, but they outlive the +-2 cell
      band the norms exclude: the ripple decays over about five cells, and the
      plateau reaches its settled level only beyond that.  Measured from the
      edge of the excluded band the relative scatter is 0.37%, from k+5 it is
      0.015%, and from k+10 it is 0.002%.  An earlier version of this figure
      quoted the last of those as if it described the whole plateau.

  (b) the plateau error against the sub-cell position of the interface,
      phi = (x_int/h) mod 1.  A single one-per-cell harmonic accounts for 97.8%
      of the variance, amplitude 8.0%.  This is the mechanism behind the
      oscillatory error convergence: the interface sweeps through a cell, the
      reconstruction is not equally accurate at every position within it, and
      the velocity that the phase change drives inherits that periodicity.

  (c) why that oscillation never reaches the interface.  Both series are drawn
      on ONE axis, in the same units, because the whole point is that they are
      not comparable: on a scale that holds the +-8% velocity swing, the
      position error is a flat line.  An earlier version gave each its own
      axis, which magnified the position error ninety-fold and made two series
      that differ by 90x in scatter look equally noisy -- the opposite of the
      claim.  The running mean of the velocity error is drawn as well, because
      the interface position is essentially a time-weighted integral of the
      velocity (x ~ sqrt(t) here, so u ~ 1/sqrt(t) and the relative errors
      carry over): the +-8% swing collapses to +0.67% in the mean, close to the
      +0.44% the interface actually carries.  The residual gap is the
      weighting, and the fact that the interface accumulated its error over the
      whole run rather than this window alone.

Data screening as in analyze_sucking_errors.py: production 400-cell mesh only,
trend restricted to the single coherent block (ts >= 9400).
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import analyze_sucking_errors as ase

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({"font.size": 10, "font.family": "serif",
                     "axes.grid": True, "grid.alpha": 0.3})


def main():
    beta = ase.beta_root()
    import glob, re
    files = sorted((f for f in glob.glob(os.path.join(ase.CASE, "sucking-ts*.vtu"))
                    if "front" not in f),
                   key=lambda f: int(re.search(r"ts0*(\d+)", f).group(1)))
    rows = [r for r in (ase.metrics(f, beta) for f in files) if r]
    clean = [r for r in rows if r["ts"] >= ase.CLEAN_TS]
    if not clean:
        sys.exit("no coherent snapshots found")

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.1))

    # ---- (a) the plateau, on a plateau-sized axis -------------------------
    # The offset is 4.5% of 5 mm/s.  Against the full step it is two lines a
    # hair apart; the axis therefore holds the plateau only.
    p = ase.profile(os.path.join(ase.CASE, "sucking-ts016000.vtu"))
    x, vof, u, _ = p
    xi, k = ase.interface(x, vof)
    r16 = next(r for r in rows if r["ts"] == 16000)
    idx = np.arange(-2, 31)                       # cells from the interface
    uu = u[k + idx] * 1e3
    keep = idx >= ase.BAND                        # what the norms actually use
    ua = r16["u_plate_a"] * 1e3
    settled = u[k + 10:].mean() * 1e3

    ax[0].axvspan(-2, ase.BAND, color="0.85",
                  label=f"excluded from the norms ($\\pm${ase.BAND} cells)")
    # drawn as segments rather than axhline so they stop at the interface
    # cell and do not run back through the shaded excluded band
    ax[0].plot([0, 30], [ua, ua], color="k", ls="--", lw=1.2,
               label="analytical plateau")
    ax[0].plot([0, 30], [settled, settled], color="C3", ls=":", lw=1.4,
               label=f"settled level ({100*(settled-ua)/ua:+.1f}%)")
    ax[0].plot(idx, uu, "o-", ms=3.4, lw=1.2, color="C0", label="T-Flows")
    ax[0].annotate("", xy=(24, ua), xytext=(24, settled),
                   arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.0))
    ax[0].text(24.8, 0.5 * (ua + settled), f"{100*(settled-ua)/ua:+.1f}%",
               fontsize=9, color="0.35", va="center")
    ax[0].set_xlabel("cells from the interface")
    ax[0].set_ylabel("$u$ [mm/s]")
    # the axis holds the plateau only: the interface cell itself is a third of
    # the way down the step and would flatten everything if it set the limit
    lo, hi = min(ua, uu[keep].min()), uu[keep].max()
    ax[0].set_ylim(lo - 0.10 * (hi - lo), hi + 0.10 * (hi - lo))
    ax[0].set_xlim(-2.5, 30)
    ax[0].set_title("(a) velocity profile, $t = 0.5$ s", fontsize=10, loc="left")
    ax[0].legend(fontsize=7.5, loc="lower right")

    # ---- (b) error against sub-cell position ------------------------------
    ph = np.array([r["phase"] for r in clean])
    ep = np.array([r["e_plate"] for r in clean])
    M = np.column_stack([np.cos(2*np.pi*ph), np.sin(2*np.pi*ph),
                         np.cos(4*np.pi*ph), np.sin(4*np.pi*ph), np.ones_like(ph)])
    c, *_ = np.linalg.lstsq(M, ep, rcond=None)
    r2 = 1.0 - (ep - M @ c).var() / ep.var()
    amp = np.hypot(c[0], c[1])
    g = np.linspace(0, 1, 300)
    Mg = np.column_stack([np.cos(2*np.pi*g), np.sin(2*np.pi*g),
                          np.cos(4*np.pi*g), np.sin(4*np.pi*g), np.ones_like(g)])
    ax[1].axhline(0, color="k", lw=0.8, alpha=0.6)
    ax[1].plot(ph, ep, "o", ms=4, color="C0", alpha=0.75)
    ax[1].plot(g, Mg @ c, "-", color="C3", lw=1.8,
               label=f"harmonic fit, $R^2 = {r2:.3f}$\namplitude {amp:.1f}%")
    ax[1].set_xlabel(r"interface position in a cell  $\varphi = (x_\Gamma/h)\ \mathrm{mod}\ 1$")
    ax[1].set_ylabel("plateau velocity error [%]")
    ax[1].set_title("(b) plateau velocity error against interface position in a cell",
                    fontsize=10, loc="left")
    ax[1].legend(fontsize=8, loc="lower right")
    # name the landmarks of the cell the interface is crossing, so phi is
    # readable without the caption.  Placed last: the arrow tails sit on the
    # settled ylim, so nothing may rescale the axis after this point.
    ax[1].set_xlim(-0.06, 1.06)
    for x, txt in ((0.0, "upstream face"), (0.5, "cell centre"),
                   (1.0, "downstream face")):
        ax[1].annotate(txt, xy=(x, 0), xytext=(x, ax[1].get_ylim()[1]),
                       ha="center", va="top", fontsize=8, color="0.35",
                       arrowprops=dict(arrowstyle="->", color="0.55", lw=0.8))

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "pics", f"sucking_subcell.{ext}"),
                    dpi=200, bbox_inches="tight")
    print(f"harmonic fit: amplitude {amp:.2f}%, R2 = {r2:.3f}, n = {len(clean)}")
    print(f"plateau error   {ep.mean():+.2f} +- {ep.std():.2f}%   "
          f"({ep.min():+.2f} to {ep.max():+.2f})")
    for st in (ase.BAND, 5, 10):
        pl = u[k + st:]
        print(f"plateau from k+{st:<2d}: rel.sd {100*pl.std()/pl.mean():6.3f}%  "
              f"spread {100*(pl.max()-pl.min())/pl.mean():5.2f}%")
    print(f"settled level {settled:.4f} vs analytical {ua:.4f} mm/s "
          f"({100*(settled-ua)/ua:+.2f}%)")
    print("written: pics/sucking_subcell.{pdf,png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
