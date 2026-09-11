#!/usr/bin/env python3
"""Figure 8: Scriven sensitivity analysis, rebuilt on the d_f-corrected runs.

The published figure (pics/scriven_paper_figure.pdf, 6 March) predates two
things: the frame-invariant d_f correction, and the July re-run campaign.  Its
polyhedral panels are therefore inconsistent with Table 3, which is built from
`ScrivenPolyDfix`.  The structured panels are unaffected -- on a Cartesian mesh
the cell-centre connection is axis-aligned, so the normal projection and the
old Hadamard norm coincide and the correction is a no-op there -- and are
redrawn from the same runs as before.

Everything is plotted as R_front, the area-weighted mean distance of the front
element centroids from the bubble centre, because that is the radius the
paper's tables use.  R_front/R_vol is NOT constant (it runs 1.00 -> 1.05 over
the record), so the continuous bench trace cannot simply be scaled.

  * structured runs carry 181 front snapshots, so R_front is read directly.
  * the polyhedral runs carry 6, so R_front is measured there and the ratio
    R_front/R_vol interpolated onto the bench time base.  Validated against
    run-125, which carries 61 snapshots: reconstructing its curve from only the
    6 instants the other runs have reproduces the true R_front to 0.11% worst
    case beyond t = 0.2 ms, 0.025% mean.

Three figures, each written to its own file.

  scriven_sensitivity_dt    (a) time step, polyhedral 75^3, dt = 1, 2, 10 us
                            (b) time step, structured 75^3, same three steps
  scriven_sensitivity_mesh  (a) polyhedral mesh convergence, dt = 2 us
                            (b) structured mesh convergence, dt = 2 us
  scriven_gradient          interface-modified against standard LSQ, 75^3, 1 us
"""

import glob
import os
import re
import sys

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

import scriven_reference

ROOT = os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling")
STRUCT = os.environ.get("SCRIVEN_DIR",
                          "/home/jan/runs/tflows-vof/Validation/Scriven")
POLY = os.path.join(ROOT, "Data/polydfix")

# The reference solution is derived once, in scriven_reference, from the
# properties the runs were configured with.  It used to be calibrated here to
# the two radii the manuscript printed, on the reasoning that matching the
# stated numbers kept the figure and Table 3 consistent -- but those radii were
# themselves built on a stale root (beta = 4.0602 against the 4.064685 Eq. (16)
# gives) and on a conductivity no run used, so the calibration propagated the
# error instead of avoiding it.  Nothing is restated here now.
scriven = scriven_reference.r_scriven_um


def r_front(path):
    """Area-weighted mean front radius [um], centred on the front itself."""
    m = pv.read(path)
    if m.n_cells == 0:
        return None
    c = (np.asarray(m.cell_data["ElementCoordinates"])
         if "ElementCoordinates" in m.cell_data else m.cell_centers().points)
    a = (np.asarray(m.cell_data["ElementArea"])
         if "ElementArea" in m.cell_data
         else m.compute_cell_sizes(length=False, area=True,
                                   volume=False).cell_data["Area"])
    c = c - np.average(c, axis=0, weights=a)
    return float(np.average(np.linalg.norm(c, axis=1), weights=a)) * 1e6


T_MAX = 1.2e-3          # the polyhedral campaign ends here; all panels share it
R_INIT = 50.0           # initialised bubble radius [um], ellipsoid_parameters.ini


def series_from_fronts(pattern, dt, every=1, anchor=False):
    """R_front(t) read directly, for runs with dense front output.

    `anchor` prepends the initial condition.  The front object does not
    exist until the first step completes, so the ts000000 file holds zero
    cells and is skipped, which leaves the 10 us curve starting at 0.1 ms.
    At t = 0 the interface is the initialised R0 sphere and R_front = R_vol
    exactly, so the point is the initial condition rather than an
    extrapolation.  series_from_ratio anchors the same way, which is why
    every polyhedral curve already begins at the axis.
    """
    t, r = ([0.0], [R_INIT]) if anchor else ([], [])
    for f in sorted(glob.glob(pattern)):
        ts = int(re.search(r"ts(\d+)", f).group(1))
        if ts == 0 or (ts // 10) % every or ts * dt > T_MAX + 1e-12:
            continue
        v = r_front(f)
        if v:
            t.append(ts * dt)
            r.append(v)
    return np.array(t), np.array(r)


def series_from_ratio(run, dt, bench=None, fronts=None):
    """R_front(t) for runs with sparse fronts: measure the ratio, interpolate it
    onto the bench time base, and apply it to the continuous R_vol trace.

    `bench` and `fronts` override the polyhedral layout, for runs whose bench
    record and front output live in different trees.  The structured 75^3 at
    2 us is one: 600 bench rows against 24 front snapshots, so reading the
    fronts alone gives 24 points where the trace holds every step.
    """
    b = np.loadtxt(bench or os.path.join(POLY, f"{run}.dat"))
    tb, rv = b[:, 0], b[:, 3] * 1e6
    ts_, rt = [], []
    for f in sorted(glob.glob(
            fronts or os.path.join(POLY, "fronts", run, "*front*.pvtu"))):
        ts = int(re.search(r"ts(\d+)", f).group(1))
        if ts == 0:
            continue
        v = r_front(f)
        if v:
            ts_.append(ts * dt)
            rt.append(v / np.interp(ts * dt, tb, rv))
    # anchor at t = 0, where the interface is the initialised sphere and
    # R_front = R_vol exactly; run-125 measures the ratio at 1.0003 by
    # t = 0.02 ms, so this is a measurement rather than an assumption, and it
    # makes every curve in a panel begin at the same instant.
    ts_ = np.concatenate(([0.0], np.array(ts_)))
    rt = np.concatenate(([1.0], np.array(rt)))
    return tb, rv * np.interp(tb, ts_, rt)


def main():
    C = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Two figures rather than one four-panel figure.  The time-step pair
    # belongs to the time-step subsection and the mesh pair to the two that
    # follow it, so each float can sit in the section that discusses it and
    # the reader is no longer sent forward past the mesh-convergence figure.
    fig_dt, ax_dt = plt.subplots(1, 2, figsize=(11, 4.0))
    fig_me, ax_me = plt.subplots(1, 2, figsize=(11, 4.0))
    # The gradient comparison is a different question from mesh convergence --
    # one mesh, one time step, two operators -- and it was sharing a float with
    # the polyhedral convergence panel only because there were two panels to
    # place.  With both families now drawn it stands alone, which also keeps
    # every panel at the width it needs: three across the text block would
    # render their labels at about 3.5 pt.
    fig_gr, ax_gr = plt.subplots(1, 1, figsize=(5.6, 4.0))

    # ---- dt (a) time step, polyhedral 75^3 -------------------------------
    for i, (run, dt, lab) in enumerate((("run-75-dt1", 1e-6, "$1\\,\\mu$s"),
                                        ("run-75", 2e-6, "$2\\,\\mu$s"),
                                        ("run-75-dt10", 1e-5, "$10\\,\\mu$s"))):
        t, r = series_from_ratio(run, dt)
        ax_dt[0].plot(t * 1e3, r / scriven(t), color=C[i], lw=1.6, label=lab)
    ax_dt[0].set_title("(a) time step, polyhedral $75^3$",
                       fontsize=10, loc="left")

    # ---- reference: polyhedral 75^3 at 1 us, for both structured panels ---
    tp, rp = series_from_ratio("run-75-dt1", 1e-6)

    # ---- dt (b) time step, structured 75^3 -------------------------------
    # The 2 us run is the one the convergence study is carried out at.  It
    # was originally recorded to 1.0 ms only, and the last 0.2 ms of this
    # curve used to be interpolated between the 1 and 10 us runs; it is now
    # the cold 600-step record, which reaches 1.2 ms like the other two and
    # reproduces the earlier 500-step trace bit for bit.  Its bench record
    # holds every step and its fronts are written every 25, so it is
    # reconstructed the same way panel (a) is rather than drawn as markers.
    # The reconstruction is exact at the 24 front instants and the ratio it
    # interpolates between them runs 1.001 to 1.034, monotone and slowly
    # varying.  The 1 and 10 us runs have no bench record, so they are read
    # from fronts as before.
    for i, (d, dt, lab) in enumerate(
            ((f"{STRUCT}/Scriven-Struct-75-smallDt", 1e-6, "$1\\,\\mu$s"),
             (None, 2e-6, "$2\\,\\mu$s"),
             (f"{STRUCT}/Scriven-Struct-75-largeDt", 1e-5, "$10\\,\\mu$s"))):
        if d is None:
            t, r = series_from_ratio(
                None, dt,
                bench=f"{ROOT}/Data/blend-dt2/aniso-075-off-600.dat",
                fronts=f"{ROOT}/Data/Aniso/aniso-075-off-600/*front*.pvtu")
        else:
            t, r = series_from_fronts(d + "/*front*.pvtu", dt, anchor=True)
            # The anchor is the exact initial condition and the segment from
            # it to the run's first saved front is a straight join across a
            # gap in the OUTPUT rather than a measured path.  It is drawn
            # solid like the rest of the curve and the caption says what it
            # is: at 10 us the gap is 0.1 ms and a dashed run-in was the only
            # broken line in the figure, at 1 us it is 0.01 ms and invisible,
            # so the distinction cost more in the reading than it carried.
        ax_dt[1].plot(t * 1e3, r / scriven(t), color=C[i], lw=1.6, label=lab)
        if dt == 2e-6:
            print(f"  structured 2 us, measured: R/R_Scriven "
                  f"{np.interp(1.0e-3, t, r) / scriven(1.0e-3):.4f} at 1.0 ms "
                  f"-> {np.interp(T_MAX, t, r) / scriven(T_MAX):.4f} "
                  f"at {T_MAX*1e3:.1f} ms")

    ax_dt[1].plot(tp * 1e3, rp / scriven(tp), ":", color="0.45", lw=1.3,
                  label="polyhedral $75^3$, $1\\,\\mu$s")
    ax_dt[1].set_title("(b) time step, structured $75^3$",
                       fontsize=10, loc="left")

    # ---- mesh (a) polyhedral mesh convergence, dt = 2 us -----------------
    for i, lv in enumerate(("75", "100", "125", "150")):
        t, r = series_from_ratio(f"run-{lv}", 2e-6)
        ax_me[0].plot(t * 1e3, r / scriven(t), color=C[i], lw=1.6,
                      label=f"polyhedral ${lv}^3$")
    ax_me[0].set_title("(a) polyhedral mesh convergence, $\\Delta t = 2\\,\\mu$s",
                       fontsize=10, loc="left")
    # widen the axis a little at the bottom so the four-entry legend has
    # somewhere to sit that is not on top of the 100^3 and 125^3 curves
    lo, hi = ax_me[0].get_ylim()
    ax_me[0].set_ylim(lo - 0.28 * (hi - lo), hi)
    ax_me[0].legend(fontsize=8, framealpha=0.92, loc="lower center", ncol=2)

    # ---- mesh (b) structured mesh convergence, dt = 2 us -----------------
    # The same quantity, time step and reconstruction as (a), on the other
    # topology, so the two panels can be read against each other: the
    # polyhedral curves approach unity from below and order themselves by
    # refinement, the structured ones cross it and fan the other way.  What
    # the pair shows and the tabulated errors do not is that refinement moves
    # the crossing earlier rather than reducing the error -- the 75^3 crosses
    # near 0.78 ms, the 150^3 is already above unity by 0.2 ms.
    for i, lv in enumerate(("075", "100", "125", "150")):
        bench = f"{ROOT}/Data/blend-dt2/aniso-{lv}-off-600.dat"
        fronts = f"{ROOT}/Data/Aniso/aniso-{lv}-off-600/*front*.pvtu"
        # A missing input must stop the script, never draw an empty panel:
        # analyze_mass_conservation once regenerated its figure with all six
        # of its records absent and produced a plausible plot of the
        # analytical curve alone.
        assert os.path.exists(bench), f"missing bench record: {bench}"
        assert glob.glob(fronts), f"no front snapshots matching {fronts}"
        t, r = series_from_ratio(None, 2e-6, bench=bench, fronts=fronts)
        ax_me[1].plot(t * 1e3, r / scriven(t), color=C[i], lw=1.6,
                      label=f"structured ${int(lv)}^3$")
    ax_me[1].set_title("(b) structured mesh convergence, "
                       "$\\Delta t = 2\\,\\mu$s", fontsize=10, loc="left")

    # ---- gradient construction, its own float ----------------------------
    for i, (d, lab) in enumerate(
            ((f"{STRUCT}/Scriven-Struct-75-smallDt", "interface-modified"),
             (f"{STRUCT}/Scriven-Struct-75-noFrontGrad", "standard LSQ"))):
        t, r = series_from_fronts(d + "/*front*.pvtu", 1e-6)
        ax_gr.plot(t * 1e3, r / scriven(t), color=C[i], lw=1.6, label=lab)
    ax_gr.plot(tp * 1e3, rp / scriven(tp), ":", color="0.45", lw=1.3,
               label="polyhedral $75^3$, $1\\,\\mu$s")
    ax_gr.set_title("gradient construction, structured $75^3$, "
                    "$\\Delta t = 1\\,\\mu$s", fontsize=10, loc="left")

    for a in (*ax_dt, *ax_me, ax_gr):
        a.set_xlim(left=0.0)
        a.margins(x=0.02)
        a.set_xlabel("$t$ [ms]")
        a.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
        a.set_ylabel("$R_\\mathrm{front}/R_\\mathrm{Scriven}$")
        a.grid(alpha=0.3)
        # Both convergence panels carry four curves that fan across the whole
        # axis, so each opens headroom below and takes a two-column legend
        # there; the others have room for "best" to find.
        if a is ax_me[1]:
            lo, hi = a.get_ylim()
            a.set_ylim(lo - 0.28 * (hi - lo), hi)
            a.legend(fontsize=8, framealpha=0.92, loc="lower center", ncol=2)
        elif a is not ax_me[0]:
            a.legend(fontsize=8, framealpha=0.92, loc="best")

    for fig, stem in ((fig_dt, "scriven_sensitivity_dt"),
                      (fig_me, "scriven_sensitivity_mesh"),
                      (fig_gr, "scriven_gradient")):
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(f"{ROOT}/pics/{stem}.{ext}", dpi=200,
                        bbox_inches="tight")
        print(f"written: pics/{stem}.{{pdf,png}}")

    # ---- report the values the text quotes -------------------------------
    print("\npolyhedral R_front error at t = 1.0 ms (Table 3 says "
          "-7.63, -3.38, -1.42, -0.92):")
    for lv in ("75", "100", "125", "150"):
        t, r = series_from_ratio(f"run-{lv}", 2e-6)
        v = np.interp(1e-3, t, r)
        print(f"   poly-{lv:<4} {v:8.3f} um   {100*(v-scriven(1e-3))/scriven(1e-3):+7.2f}%")


if __name__ == "__main__":
    sys.exit(main())
