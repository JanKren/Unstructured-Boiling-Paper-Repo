#!/usr/bin/env python3
"""Time history of the mass, after Sato & Nicheno (JCP 2013, Fig. 8).

    python3 plot_mass_history.py

Two definitions of mass, both non-dimensionalised by M_0:

    M_d = sum_cells rho V_cell                      mass inside the domain
    M_s = M_d + sum_t sum_outlet rho u.S dt         mass allowing for egress

Sato & Nicheno's Fig. 8 draws the CONSERVATION ERROR, M_s/M_0 - 1, on the left
axis, with M_d/M_0 on a right-hand axis for context -- theirs spans +-2e-12 on
the left and 0.80 to 1.00 on the right.  This figure follows that construction.
An earlier version put both masses on one 0-to-1 axis, on which our 1e-4 error
is a flat line at unity indistinguishable from perfect conservation: it showed
everything except the quantity the figure exists for.

BOTH HALVES ARE MEASURED, WHICH IS THE WHOLE POINT.  The egress term is
accumulated inside the solver by the hook in
`instrumentation/End_Of_Time_Step_mass_balance.f90` -- it is recorded nowhere
otherwise.  Inferring it as M_0 - M_d would make M_s identically M_0, a flat
line produced by algebra rather than by conservation and indistinguishable by
eye from the real thing.

WHAT THE FIGURE SHOWS.  The three cases span two orders in cell count, three
SIMPLE tolerances and egress from 22% to 90%, and their conservation errors all
sit at a few times 1e-4.  That the floor does not move with the tolerance is the
substance: the formulation solves a VOLUME continuity equation, drives its
residual to ~1e-11, and conserves mass only to the discretisation level.  Sato
and Nicheno's 1e-12 is what a mass-conservative construction achieves, which is
a different property rather than a better value of the same one.

Data: `/home/jan/runs/MassBalance/<case>/mass-balance.dat`, columns
t, M_d, M_s, M_d/M_0, M_s/M_0-1 [, Q_out, sum_mdot].
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR",
                            os.path.join(os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling"), "pics"))
ROOT = os.environ.get("MASSBALANCE_DIR", "/home/jan/runs/MassBalance")

# label, directory, SIMPLE tolerance as shipped, time unit and its scale
CASES = [
    ("(a) Stefan",     "stefan",    r"$10^{-3}$", "ms", 1e3),
    ("(b) Sucking",    "sucking",   r"$10^{-3}$", "ms", 1e3),
    # the polyhedral 125^3 case, so this panel is the same run as the Scriven
    # row of the budgets table rather than a coarser one two paragraphs away
    ("(c) Scriven, polyhedral $125^3$", "scriven125", r"$10^{-4}$", "ms", 1e3),
]

# Sizes are set for the PAGE, not for the file.  Three panels side by side make
# this figure 14.1 in wide and it is included at \linewidth, so it lands at 0.46
# scale and everything in it is more than halved: the 10 pt this used to carry
# reached the page at 4.6 pt, against the 7.5 pt of the extrapolation figure,
# which is drawn 7.1 in wide and barely scaled.  16 pt here and 17 pt on the key
# land at 7.4 and 7.8 pt, so the two figures read alike side by side.
plt.rcParams.update({"font.size": 16, "font.family": "serif",
                     "axes.grid": True, "grid.alpha": 0.3})


def main():
    have = [(l, d, tol, u, s) for l, d, tol, u, s in CASES
            if os.path.exists(os.path.join(ROOT, d, "mass-balance.dat"))]
    if not have:
        sys.exit("no mass-balance.dat found")

    fig, axes = plt.subplots(1, len(have), figsize=(4.7 * len(have), 3.8))
    axes = np.atleast_1d(axes)

    print(f"  {'case':<26}{'egress':>9}{'median |M_s/M_0-1|':>21}{'final':>11}")
    for ax, (lab, d, tol, unit, scale) in zip(axes, have):
        a = np.loadtxt(os.path.join(ROOT, d, "mass-balance.dat"))
        t, md, err = a[:, 0] * scale, a[:, 3], a[:, 4]

        # After Sato & Nicheno Fig. 8: the error on the left axis, where it is
        # legible, and the domain mass on the right for context.
        ax.axhline(0.0, color="k", lw=0.9)
        ax.plot(t, err, color="C0", lw=1.9, label=r"$M_s/M_0 - 1$")
        lim = 1.15 * np.abs(err).max()
        ax.set_ylim(-lim, lim)
        ax.set_xlim(0, t.max())
        ax.set_xlabel(f"time [{unit}]")
        ax.set_ylabel(r"$M_s/M_0 - 1$", color="C0")
        ax.tick_params(axis="y", labelcolor="C0")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        tw = ax.twinx()
        tw.plot(t, md, color="C3", lw=1.5, ls="--", label=r"$M_d/M_0$")
        tw.set_ylim(min(md) - 0.02, 1.02)
        tw.set_ylabel(r"$M_d/M_0$", color="C3")
        tw.tick_params(axis="y", labelcolor="C3")
        tw.grid(False)

        ax.set_title(lab, fontsize=16, loc="left")

        print(f"  {lab:<26}{100*(md[-1]-1):>8.1f}%{np.median(np.abs(err)):>21.2e}"
              f"{np.abs(err[-1]):>11.2e}")

    # One key for the whole figure, as the extrapolation figure carries.  The
    # two curves and the zero line mean the same thing in all three panels, so
    # they are stated once rather than boxed three times; both axes are already
    # colour-coded to match, and each panel keeps its own scale.
    key = [Line2D([], [], color="C0", lw=1.9, label=r"$M_s/M_0 - 1$"),
           Line2D([], [], color="C3", lw=1.5, ls="--", label=r"$M_d/M_0$"),
           Line2D([], [], color="k", lw=0.9, label="exact conservation")]
    # See the note in plot_scriven_extrapolation.  The reserved fraction is
    # larger here because the key is set at 15 pt against a 3.8 in height.
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    fig.legend(handles=key, loc="lower center", ncol=3, frameon=False,
               fontsize=17, handlelength=1.6, columnspacing=1.6,
               bbox_to_anchor=(0.5, 0.0))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"mass_history.{ext}"),
                    dpi=200, bbox_inches="tight")
    print("written: pics/mass_history.{pdf,png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
