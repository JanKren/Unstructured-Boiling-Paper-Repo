#!/usr/bin/env python3
"""The deformation with the volume-fraction transport removed.

Panel (a) shows the interface radius against the cubic invariant
s = nx^4+ny^4+nz^4 at matched radius, which is the same regression the other
shape figures use: s is 1 along a coordinate axis and 1/3 along a body
diagonal, so a negative slope is a bubble bulging towards the corners.  With
the transport removed the line is nearly flat; with it active the same bubble,
at the same radius and the same R/h, carries an order of magnitude more slope.

Panel (b) is the part that separates a bias from an accumulation.  Plotting the
axis-to-diagonal difference against radius, the no-transport curve is constant
across a factor 2.2 in radius while the unmodified one climbs monotonically.  A
constant offset is what a discretisation bias looks like -- the reconstruction
is slightly non-spherical at every radius and stays so.  Something that grows
with interface displacement is what a transport defect looks like.

Matched radius rather than matched time throughout, because the four-fold mode
collapses on R/h.

Usage:  python3 plot_noadvection.py [--out pics/noadvection.pdf]
"""

import argparse
import glob
import os

import numpy as np
import pyvista as pv

# The comparison that isolates the transport must hold the flux fixed.  Both
# UNIF_ON and NOADV use UNIFORM_MASS_TRANSFER at dt = 1 us on the same mesh and
# differ only in whether the volume fraction is transported; COMPUTED is the
# unmodified production run, shown for context, and differs in the flux as well.
NOADV = "Data/inflate-unif"        # uniform mdot, transport removed
UNIF_ON = "Data/unif-transport"    # uniform mdot, transport active  <- the pair
COMPUTED = "Data/Aniso/aniso-075-off"   # computed mdot, transport active
H = 4.054e-6            # cell size of the structured 75^3 mesh


def front(path):
    """Element areas, radii and cubic invariant of one saved front."""
    m = pv.read(path)
    if m.n_cells == 0:
        return None
    sized = m.compute_cell_sizes(length=False, area=True, volume=False)
    a = np.asarray(sized.cell_data["Area"], float)
    c = np.asarray(sized.cell_centers().points, float)
    ctr = (a[:, None] * c).sum(0) / a.sum()
    q = c - ctr
    r = np.linalg.norm(q, axis=1)
    ok = r > 0
    a, q, r = a[ok], q[ok], r[ok]
    n = q / r[:, None]
    return a, r, (n ** 4).sum(-1)


def fit(a, r, s):
    """Area-weighted regression of r on s; reduces to <r> when the slope is 0."""
    w = np.sqrt(a)
    c0, c1 = np.linalg.lstsq(np.vstack([w, w * s]).T, w * r, rcond=None)[0]
    return c0, c1, float((a * r).sum() / a.sum())


def series(directory):
    out = []
    for f in sorted(glob.glob(os.path.join(directory, "*front-ts*.pvtu"))):
        got = front(f)
        if got is None:
            continue
        a, r, s = got
        c0, c1, mean = fit(a, r, s)
        axis, body = c0 + c1, c0 + c1 / 3.0
        out.append(dict(R=mean, dev=100 * (body / axis - 1), a=a, r=r, s=s,
                        c0=c0, c1=c1))
    return sorted(out, key=lambda d: d["R"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pics/noadvection.pdf")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'font.family': 'serif',
                         'savefig.dpi': 300, 'axes.grid': True,
                         'grid.alpha': 0.3})

    noadv, unif, comp = series(NOADV), series(UNIF_ON), series(COMPUTED)
    if not noadv or not unif:
        raise SystemExit("missing front output; expected pvtu under "
                         f"{NOADV} and {UNIF_ON}")
    full = unif        # panel (a) compares the single-variable pair

    fig = plt.figure(figsize=(7.4, 3.3))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # ---- (a) radius against the cubic invariant, at matched radius --------
    # pick the pair closest in radius so the comparison is at matched R/h
    pairs = [(n, f) for n in noadv for f in full]
    n_sel, f_sel = min(pairs, key=lambda p: abs(p[0]["R"] - p[1]["R"]))

    edges = np.linspace(1 / 3, 1, 13)
    mid = 0.5 * (edges[:-1] + edges[1:])
    for d, col, mark, lab in ((f_sel, 'tab:grey', 'o',
                               rf"transport, $R={f_sel['R']*1e6:.0f}\,\mu$m"),
                              (n_sel, 'tab:red', 's',
                               rf"no transport, $R={n_sel['R']*1e6:.0f}\,\mu$m")):
        # area-weighted mean radius per bin, normalised by that run's own mean
        idx = np.clip(np.digitize(d["s"], edges) - 1, 0, len(mid) - 1)
        num = np.bincount(idx, weights=d["a"] * d["r"], minlength=len(mid))
        den = np.bincount(idx, weights=d["a"], minlength=len(mid))
        with np.errstate(invalid="ignore"):
            binned = num / den
        m = den > 0
        norm = (d["a"] * d["r"]).sum() / d["a"].sum()
        ax1.plot(mid[m], binned[m] / norm, mark, color=col, ms=3.5, lw=0,
                 label=lab)
        ss = np.linspace(1 / 3, 1, 50)
        ax1.plot(ss, (d["c0"] + d["c1"] * ss) / norm, '-', color=col, lw=1.2)

    ax1.axhline(1.0, color='k', lw=0.6, ls=':')
    ax1.set_xlabel(r"cubic invariant $s = n_x^4+n_y^4+n_z^4$")
    ax1.set_ylabel(r"$R(s)\,/\,\langle R\rangle$")
    ax1.legend(fontsize=7, loc='upper right', framealpha=0.9)
    # headroom at the bottom for the direction labels, which would otherwise
    # sit outside the axes and collide with the panel title
    lo, hi = ax1.get_ylim()
    ax1.set_ylim(lo - 0.11 * (hi - lo), hi)
    ax1.text(0.02, 0.03, "body diagonal", transform=ax1.transAxes,
             fontsize=6.5, ha='left', va='bottom', color='0.35')
    ax1.text(0.98, 0.03, "axis", transform=ax1.transAxes,
             fontsize=6.5, ha='right', va='bottom', color='0.35')
    ax1.set_title(r"(a) uniform $\dot m$, matched radius", fontsize=9, pad=6)

    # ---- (b) deformation against radius -----------------------------------
    ax2.plot([d["R"] * 1e6 for d in comp], [d["dev"] for d in comp],
             '^--', color='0.65', ms=4, lw=1.1,
             label=r"computed $\dot m$, transport")
    ax2.plot([d["R"] * 1e6 for d in unif], [d["dev"] for d in unif],
             'o-', color='tab:grey', ms=4, lw=1.3,
             label=r"uniform $\dot m$, transport")
    ax2.plot([d["R"] * 1e6 for d in noadv], [d["dev"] for d in noadv],
             's-', color='tab:red', ms=4, lw=1.3,
             label=r"uniform $\dot m$, no transport")
    ax2.axhline(0.0, color='k', lw=0.6, ls=':')
    ax2.set_xlabel(r"interface radius $R$ [$\mu$m]")
    ax2.set_ylabel(r"body diagonal $-$ axis [%]")
    ax2.legend(fontsize=6.6, loc='upper left', bbox_to_anchor=(0.015, 0.88),
               framealpha=0.92, handlelength=1.8, borderpad=0.4)
    lo2, hi2 = ax2.get_ylim()
    ax2.set_ylim(lo2, hi2 + 0.16 * (hi2 - lo2))
    ax2.text(0.02, 0.97, "(b) does it accumulate?", transform=ax2.transAxes,
             fontsize=9, ha='left', va='top')

    sec = ax2.secondary_xaxis('top', functions=(lambda x: x * 1e-6 / H,
                                                lambda x: x * H * 1e6))
    sec.set_xlabel(r"$R/h$", fontsize=8)
    sec.tick_params(labelsize=7)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches='tight')
    fig.savefig(a.out.replace('.pdf', '.png'), bbox_inches='tight')

    print(f"\n  wrote {a.out} and {a.out.replace('.pdf', '.png')}")
    print(f"  panel (a) pair: transport removed R = {n_sel['R']*1e6:.2f} um, "
          f"active R = {f_sel['R']*1e6:.2f} um "
          f"(mismatch {abs(n_sel['R']-f_sel['R'])*1e6:.2f} um)")
    print(f"  {'R [um]':>9}{'R/h':>7}{'dev [%]':>10}   configuration")
    for lab, ser in (("uniform mdot, NO transport", noadv),
                     ("uniform mdot, transport", unif),
                     ("computed mdot, transport", comp)):
        for d in ser:
            print(f"  {d['R']*1e6:9.2f}{d['R']/H:7.1f}{d['dev']:+10.3f}   {lab}")


if __name__ == "__main__":
    main()
