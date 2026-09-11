#!/usr/bin/env python3
"""Mass and energy conservation of the phase-change model, Scriven problem.

    python3 plot_conservation.py

A reviewer asked for a conservation analysis of both benchmarks.  What was
produced under the name `*_mass_conservation` is a pair of ACCURACY plots ---
radius growth and radius error --- so the request has not in fact been answered.
This is the conservation analysis.

Two balances are checked, both of which the model must satisfy identically if it
is implemented consistently, and neither of which is a restatement of the other:

  MASS.  Every kilogram that crosses the interface becomes vapour, so the vapour
  volume must grow at exactly the rate the mass-transfer model prescribes,

      rho_v dV_v/dt  =  sum_c m_dot(c) .

  ENERGY.  The net heat delivered to the interface is consumed as latent heat,

      q_liq - q_vap  =  h_fg sum_c m_dot(c) .

The two use different outputs -- the first the volume integral of the volume
fraction, the second the interfacial heat fluxes -- so agreement in both is a
genuine check on the coupling rather than on one routine.

DATA.  `Scriven_Algebraic_Poly`, the polyhedral 75-equivalent Scriven case at
the paper's 1.25 K superheat (verified: max T = 101.25 in the profile file),
CICSAM, dt = 1 us.  The mass balance comes from the six-column `bench-data.dat`
(t, sum_m_dot, V_v, R_vol, R_Scriven, area); the heat fluxes are scraped from the
`# q_vap= ... q_liq=` lines of the solver log, which is the only place they are
recorded.

WHY THIS RUN.  The production Scriven and Sucking runs used a four-column
`End_Of_Time_Step.f90` that writes neither m_dot nor the heat fluxes, and their
logs do not carry them either, so no conservation figure can be made from them
without re-running.  This case does record both, at the correct physics.  It is
not identical to a tabulated production run -- it uses dt = 1 us where those use
2 us -- and the caption must say so.
"""

import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.join(os.environ.get("GOLD_DIR", "/home/jan/runs/RisingBubbleCases_gold"), "Scriven_Algebraic_Poly")
LOG = "RB-Algebraic_Poly-nanfix-6075443.out"
RHO_V, H_FG = 0.597, 2.26e6

plt.rcParams.update({"font.size": 11, "font.family": "serif",
                     "axes.grid": True, "grid.alpha": 0.3})


def load():
    d = np.loadtxt(os.path.join(CASE, "bench-data.dat"))
    t, mdot, vol = d[:, 0], d[:, 1], d[:, 2]
    txt = open(os.path.join(CASE, LOG), errors="ignore").read()
    q = np.array([[float(a), float(b)] for a, b in
                  re.findall(r"# q_vap=\s*([-\d.E+]+)\s+q_liq=\s*([-\d.E+]+)", txt)])
    n = min(len(t), len(q))
    return t[:n], mdot[:n], vol[:n], q[:n, 0], q[:n, 1]


TMAX = 1.0e-3          # the paper reports Scriven at t = 1 ms; see below


def main():
    t, mdot, vol, q_vap, q_liq = load()

    # The run is only usable over the reported window.  Beyond ~2 ms sum_m_dot
    # changes sign and the volume stops tracking it, and the log carries at
    # least one corrupted heat-flux value (order 1e78 -- the file is named
    # "nanfix").  Both are guarded rather than plotted: a conservation figure
    # must not be drawn through numbers the run itself did not survive.
    finite = np.isfinite(q_vap) & np.isfinite(q_liq) & (np.abs(q_liq) < 1e10)
    keep = (t <= TMAX) & finite
    dropped = int((t <= TMAX).sum() - keep.sum())
    if dropped:
        print(f"  note: {dropped} corrupted log rows dropped inside the window")
    t, mdot, vol = t[keep], mdot[keep], vol[keep]
    q_vap, q_liq = q_vap[keep], q_liq[keep]

    lhs_m = RHO_V * np.gradient(vol, t)      # rho_v dV_v/dt
    lhs_e = q_liq - q_vap                    # net heat to the interface
    rhs_e = mdot * H_FG                      # latent heat consumed

    ok = np.abs(mdot) > 1e-30
    err_m = np.full_like(t, np.nan); err_e = np.full_like(t, np.nan)
    err_m[ok] = np.abs(lhs_m[ok] - mdot[ok]) / np.abs(mdot[ok])
    err_e[ok] = np.abs(lhs_e[ok] - rhs_e[ok]) / np.abs(rhs_e[ok])

    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.9))
    tm = t * 1e3

    for a, (lhs, rhs, err, lab_l, lab_r, ttl, unit) in enumerate((
            (lhs_m, mdot, err_m, r"$\rho_v\,\mathrm{d}V_v/\mathrm{d}t$",
             r"$\sum_c \dot{m}_c$", "(a) mass", "kg s$^{-1}$"),
            (lhs_e, rhs_e, err_e, r"$q_\mathrm{liq}-q_\mathrm{vap}$",
             r"$h_{fg}\sum_c \dot{m}_c$", "(b) energy", "W"))):
        ax[a].plot(tm, rhs, color="C0", lw=3.0, alpha=0.40, label=lab_r)
        ax[a].plot(tm, lhs, color="C3", lw=1.2, ls="--", label=lab_l)
        ax[a].set_xlabel("time [ms]")
        ax[a].set_ylabel(f"[{unit}]")
        ax[a].set_title(f"{ttl}: median imbalance "
                        f"{np.nanmedian(err):.1e}", fontsize=10.5, loc="left")
        ax[a].legend(fontsize=9.5, loc="lower right")
        ax[a].set_xlim(0, tm.max())

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, "pics", f"conservation.{ext}"),
                    dpi=200, bbox_inches="tight")

    print("Scriven polyhedral 75-equiv, 1.25 K, CICSAM, dt = 1 us")
    print(f"  {'window':<18}{'mass imbalance':>18}{'energy imbalance':>19}")
    for lab, sel in (("first half", t <= 0.5 * TMAX),
                     ("second half", t > 0.5 * TMAX),
                     ("whole window", np.ones_like(t, bool))):
        s = sel & ok
        print(f"  {lab:<18}{np.nanmedian(err_m[s]):>18.2e}"
              f"{np.nanmedian(err_e[s]):>19.2e}")
    print(f"\n  window: 0 to {TMAX*1e3:.0f} ms, {t.size} steps")
    print("written: pics/conservation.{pdf,png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
