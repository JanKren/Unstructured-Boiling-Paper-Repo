#!/usr/bin/env python3
"""Figure 8: Roache extrapolation of R_front, in the mesh and in the time step.

Column (a), mesh refinement.  The published panel plotted |e| against h on
log-log for the polyhedral family only, with the structured family withheld
because its error grows under refinement and a log-log panel invites an order
to be read off it.  That framing follows from fitting log|e| against the
*analytical* solution, which presumes the limit is zero error.  Fitted instead
in the three-parameter form of Roache (1997, 1998),

    R/R_Scriven = b + a (h/h_150)^p,

with the limit b free, both families are well behaved and the structured one is
seen to converge to a value that is not the analytical solution.  Both are
therefore plotted, resolved by direction, with the extrapolated limits marked
on the axis h = 0.

Column (b), time-step refinement.  Here the exponent is not fitted.  Operator
splitting makes the interface lag by O(dt), so p = 1 follows from theory, and
the observed orders of 0.90 and 0.78 corroborate it.  The limit is therefore
Roache's two-point extrapolation on the finest available pair at fixed p = 1,

    f_0 = f_fine + (f_fine - f_coarse) / (r^p - 1),

which is what Table 4 tabulates.  The coarsest step of each family is plotted
but does not enter its fit: on the polyhedral family the 10 us point sits well
off the asymptotic line, which is the reason the fit is anchored on the fine
pair and the reason a figure says more here than the table does.

Data are the tabulated values of Tables 3 and 4 (t = 1.0 ms), so this script
needs no run output and reproduces both tables exactly.
"""
import os
import numpy as np, matplotlib

OUTPUT_DIR = os.environ.get("OUTPUT_DIR",
                            os.path.join(os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling"), "pics"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
from itertools import combinations

# Geometry.  Two columns now, so the float takes the full text width instead of
# the 0.62 it took when this was a single stacked pair.
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.8,
                     "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 8.5, "figure.dpi": 160})

# ---------------------------------------------------------------- mesh column
H = np.array([4.05, 3.03, 2.42, 2.01])          # nominal cell size [um]
HT = H / H[-1]                                   # normalised, 1.000 at the finest
SERIES = {              # per cent error in R_front, Table 3, at t = 1.0 ms
    "structured": {"body diagonal": [ 5.72, 11.85, 15.08, 16.81],
                   "mean":          [ 1.96,  7.11,  9.57, 10.75],
                   "axis":          [-3.99, -0.49,  0.66,  0.87]},
    "polyhedral": {"body diagonal": [-6.81, -2.31, -0.24,  0.26],
                   "mean":          [-7.63, -3.38, -1.42, -0.92],
                   "axis":          [-8.88, -5.01, -3.22, -2.75]},
}
# The one direction-to-colour mapping the paper uses.  It is stated once, in
# the figure-level key built in main(), and nowhere else.
COL = {"body diagonal": "C3", "mean": "C0", "axis": "C2"}

# ------------------------------------------------------------ temporal column
R_SCRIVEN = 116.64      # um at t = 1.0 ms, the normalisation of Table 3
# R_front [um] against dt [us] on the 75^3 mesh of each family, resolved by
# direction exactly as the mesh column is: the area-weighted mean, and the
# regression on the cubic invariant s evaluated at s = 1 (axis) and s = 1/3
# (body diagonal).  Measured by analyze_temporal_convergence, which reads the
# same fronts the mesh table is built from; steps run coarse to fine.
TEMPORAL = {
    "structured": {"dt": [2.0, 1.0, 0.1],
                   "body diagonal": [123.3226, 125.4656, 127.6105],
                   "mean":          [118.9347, 120.8670, 122.8104],
                   "axis":          [111.9910, 113.5771, 115.1914]},
    "polyhedral": {"dt": [10.0, 2.0, 1.0],
                   "body diagonal": [100.2453, 108.6980, 110.1228],
                   "mean":          [ 99.6391, 107.7440, 109.0928],
                   "axis":          [ 98.7216, 106.2935, 107.5273]},
}
P_TIME = 1.0            # order of the splitting in the time step, fixed
FS = 1.25               # Roache's factor of safety


P_MESH = 2.0            # formal order of the scheme, fixed per Roache.  A freely
                        # fitted exponent returns 1.65 to 3.84 here, with
                        # leave-one-out ranges two to three times as wide, which
                        # is the signature of sequences that are not asymptotic.


def model(x, a, b):
    """Richardson form at the fixed formal order, f = b + a h^P_MESH."""
    return a * x**P_MESH + b


def fit(y):
    """Two parameters on four levels, so the fit is over-determined and its
    residual is meaningful.  Ranges are the four leave-one-out refits."""
    q, _ = curve_fit(model, HT, y, p0=[-0.05, 1.05], maxfev=400000)
    rms = float(np.sqrt(np.mean((model(HT, *q) - y) ** 2)))
    jb = []
    for i in combinations(range(4), 3):
        try:
            jb.append(curve_fit(model, HT[list(i)], y[list(i)], p0=q, maxfev=400000)[0][1])
        except Exception:
            pass
    return q, rms, (min(jb), max(jb))


def richardson(f_fine, f_coarse, r, p=P_TIME):
    """Roache's two-point extrapolation to zero step at an assumed order q."""
    return f_fine + (f_fine - f_coarse) / (r**p - 1.0)


def gci(f_fine, f_coarse, r, p=P_TIME, fs=FS):
    return fs * abs((f_coarse - f_fine) / f_fine) / (r**p - 1.0)


def mesh_panel(ax, fam, D, xs):
    for k, v in D.items():
        y = 1 + np.array(v) / 100.0
        q, rms, (b0, b1) = fit(y)
        ax.plot(HT, y, "o", ms=5, color=COL[k], zorder=3)
        ax.plot(xs, model(xs, *q), "-", lw=1.4, color=COL[k])
        print(f"{fam + ' ' + k:<24}{q[1]:>9.4f}{b0:>9.3f} to {b1:<7.3f}"
              f"{rms * 100:>8.3f}%{(q[1] - 1) * 100:>10.1f}")
    ax.axhline(1.0, color="0.35", ls="--", lw=0.9)
    ax.grid(alpha=0.3); ax.set_xlim(-0.09, HT.max() * 1.04)
    ax.set_ylabel(r"$R_\mathrm{front}/R_\mathrm{Scriven}$")


def temporal_panel(ax, fam):
    dt = np.array(TEMPORAL[fam]["dt"], float)        # coarse to fine
    xs = np.linspace(0.0, dt.max() * 1.06, 200)
    # Finest pair carries every fit.  r is their ratio, p is fixed at unity.
    r = dt[-2] / dt[-1]
    for k in ("body diagonal", "mean", "axis"):
        R = np.array(TEMPORAL[fam][k], float)
        y = R / R_SCRIVEN
        lim = richardson(y[-1], y[-2], r)
        # Roache's two-point form is exactly a straight line in dt through the
        # pair, so the fit is drawn as that line and continued to dt = 0.
        slope = (y[-1] - lim) / dt[-1]
        # Every step is drawn the same way.  Which of them carry the fit is
        # said in the caption instead of encoded in the marker: the coarsest
        # is excluded, and on the polyhedral family it sits far enough off the
        # line to be read as a miss rather than an exclusion, so the caption
        # has to be explicit whatever the symbols do.
        ax.plot(dt, y, "o", ms=5, color=COL[k], zorder=3)
        ax.plot(xs, lim + slope * xs, "-", lw=1.4, color=COL[k])
        print(f"{fam + ' ' + k + ' (dt)':<24}{'p=1':>7}{'':>18}{lim:>9.4f}"
              f"{'':>18}{(lim - 1) * 100:>10.1f}   R0 = {lim * R_SCRIVEN:7.2f} um")
    ax.axhline(1.0, color="0.35", ls="--", lw=0.9)
    ax.grid(alpha=0.3); ax.set_xlim(-0.03 * dt.max(), dt.max() * 1.06)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0] if dt.max() < 5 else [0, 2, 4, 6, 8, 10])
    ax.set_ylabel(r"$R_\mathrm{front}/R_\mathrm{Scriven}$")
    Rm = np.array(TEMPORAL[fam]["mean"], float)
    print(f"{fam + ' GCI on the mean':<24}{'':>52}"
          f"GCI(2,1us) = {gci(Rm[np.argmin(abs(dt - 1.0))], Rm[np.argmin(abs(dt - 2.0))], 2.0) * 100:5.2f}%")


def main():
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 4.0))
    xs = np.linspace(0.0, HT.max() * 1.04, 400)
    print(f"{'series':<24}{'b':>9}{'b range':>18}{'rms':>9}{'limit [%]':>11}")
    for row, (fam, D) in enumerate(SERIES.items()):
        mesh_panel(axs[row, 0], fam, D, xs)
        temporal_panel(axs[row, 1], fam)
        for col in (0, 1):
            axs[row, col].text(0.5, 1.02, fam, transform=axs[row, col].transAxes,
                               ha="center", va="bottom", fontsize=8, style="italic")
    axs[0, 0].set_title("(a) mesh refinement, $p = 2$", loc="left", pad=14)
    axs[0, 1].set_title("(b) time-step refinement, $p = 1$", loc="left", pad=14)
    axs[1, 0].set_xlabel("$h/h_{150^3}$")
    axs[1, 1].set_xlabel(r"$\Delta t$ [$\mu$s]")
    axs[0, 0].set_xticklabels([]); axs[0, 1].set_xlabel(r"$\Delta t$ [$\mu$s]")

    # One legend for the whole figure.  It is a key to the ENCODING rather than
    # a list of series: colour is the direction, a line is the fit and a marker
    # is a computed level, and those three conventions hold in all four panels.
    # Repeating them per panel cost twelve entries for three colours and put a
    # frame over a curve in two of them.  The fitted limits the entries used to
    # carry are tabulated -- the mesh ones in the extrapolation table, the
    # temporal ones as the dt -> 0 rows of the temporal table.
    key = [Line2D([], [], color=COL[k], lw=0, marker="o", ms=5, label=k)
           for k in ("body diagonal", "mean", "axis")]
    key += [Line2D([], [], color="0.25", lw=1.4, label="convergence fit"),
            Line2D([], [], color="0.35", lw=0.9, ls="--",
                   label="analytical solution")]
    # Lay the axes out into the band above the key first.  tight_layout does
    # not account for a figure-level legend, so placing the key before it let
    # the axes -- and their x labels -- move down onto the key.
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.legend(handles=key, loc="lower center", ncol=5, frameon=False,
               handlelength=1.6, columnspacing=1.4,
               bbox_to_anchor=(0.5, 0.0))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"scriven_extrapolation.{ext}"), dpi=200, bbox_inches="tight")
    print("\nwritten: pics/scriven_extrapolation.{pdf,png}")


if __name__ == "__main__":
    main()
