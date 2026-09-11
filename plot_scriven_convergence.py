#!/usr/bin/env python3
"""Figure 7(b) redrawn: R_front convergence at t = 1 ms, both mesh families.

The published panel plotted the relative error in the volume-equivalent radius
at t = 1.5 ms, and was the only figure in the paper on R_vol -- which forced a
standing caveat about the 3.3%/4.9% offset against the R_front used everywhere
else.  It is redrawn here on R_front at t = 1 ms, the instant of Table 3, so
figure and table report one quantity at one instant.

t = 1 ms rather than 1.5 ms because the d_f-corrected polyhedral campaign
(`ScrivenPolyDfix`) runs to 1.2 ms.

Both families are read the same way: the area-weighted mean distance of the
front element centroids from the bubble centre.

  structured  Data/Aniso/aniso-{075,100,125,150}-off, dt = 2 us, ts 500
  polyhedral  Data/polydfix/fronts/run-{75,100,125,150}, dt = 2 us, ts 500

Only the polyhedral family is plotted.  The structured CICSAM error grows under
refinement -- the paper says so repeatedly -- and a log-log panel with reference
slopes invites the reader to read an order off a sequence that has none.  The
structured numbers are in Table 3, where a divergent sequence can be read as
what it is.  The one structured sequence that does converge, pure Hyper-C, is
introduced only in Section 4.2.1 and is plotted there.
"""

import os
import glob
import sys

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

import scriven_reference

ROOT = os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling")
# derived in scriven_reference from the run properties, not restated here
T0 = scriven_reference.T0
R_SCR_1MS = scriven_reference.r_scriven_um(1.0e-3)
H = np.array([4.05, 3.03, 2.42, 2.01])              # nominal cell size [um]


def r_front(path):
    m = pv.read(path)
    c = np.asarray(m.cell_data["ElementCoordinates"])
    a = np.asarray(m.cell_data["ElementArea"])
    c = c - np.average(c, axis=0, weights=a)
    return float(np.average(np.linalg.norm(c, axis=1), weights=a)) * 1e6


def collect(pattern_for):
    out = []
    for lv in ("075", "100", "125", "150"):
        g = sorted(glob.glob(pattern_for(lv)))
        if not g:
            raise FileNotFoundError(pattern_for(lv))
        out.append(r_front(g[0]))
    return np.array(out)


def main():
    poly = collect(lambda lv: f"{ROOT}/Data/polydfix/fronts/run-{int(lv)}/*front-ts000500.pvtu")
    ep = 100 * (poly - R_SCR_1MS) / R_SCR_1MS

    print("polyhedral R_front at t = 1 ms  [um]  and relative error")
    for i, lv in enumerate(("075", "100", "125", "150")):
        print(f"  {lv}   {poly[i]:7.2f} ({ep[i]:+6.2f}%)")
    op = np.polyfit(np.log(H), np.log(np.abs(ep)), 1)[0]
    print(f"\n  polyhedral observed order  {op:+.2f}")

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    # No order is quoted on the plot.  The polyhedral errors are of one sign and
    # approaching zero, so a slope fitted to log|e| steepens without bound near
    # the crossing.  The value it returns here is above three, which is an
    # artefact of the limit rather than a property of the scheme, and the text
    # says so.  The reference slopes are drawn for orientation only.
    ax.loglog(H, np.abs(ep), "o-", lw=1.6, ms=6, color="C0",
              label="polyhedral, $\\Delta t = 2\\,\\mu$s")

    # reference slopes anchored on the coarsest polyhedral point
    for order, ls in ((1, ":"), (2, "-.")):
        ref = np.abs(ep[0]) * (H / H[0]) ** order
        ax.loglog(H, ref, ls, color="0.55", lw=1.1,
                  label=f"{'first' if order==1 else 'second'} order")

    ax.set_xticks(H)
    ax.set_xticklabels([f"{v:.2f}" for v in H])
    ax.minorticks_off()
    ax.set_xlabel("$h$ [$\\mu$m]")
    ax.set_ylabel("$|R_\\mathrm{front}/R_\\mathrm{Scriven} - 1|$  [%]")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{ROOT}/pics/scriven_convergence_new.{ext}", dpi=200,
                    bbox_inches="tight")
    print("\nwritten: pics/scriven_convergence_new.{pdf,png}")


if __name__ == "__main__":
    sys.exit(main())
