#!/usr/bin/env python3
"""
Figure: the bubble stays square when the mass transfer rate is made uniform.

The most direct test of the attribution available.  The computed interfacial
mass flux is rescaled so that every interface element receives the SAME mass
per unit area, with the domain total left unchanged (`UNIFORM_MASS_TRANSFER`
in the solver).  The bubble therefore grows at the same rate to the same
radius, and the only thing removed is the angular variation of the source.

If the diagonal bulge were driven by the anisotropic mass transfer rate --
which is what the anisotropic interfacial gradient produces -- it would vanish.
It does not: two thirds of it survives, so the deformation is inherited from
the mesh-aligned transport of the volume fraction and cannot be removed by
fixing the source.

Panel (a) regresses the interface radius on the cubic invariant
s = nx^4 + ny^4 + nz^4, which is 1 along a coordinate axis, 1/2 on an edge
diagonal and 1/3 on a body diagonal.  A negative slope is a bubble bulging
towards the cube diagonals.  Panel (b) shows the same data as the normalised
radius against azimuth, where the four-fold mode is directly visible.

Usage:
  python3 plot_uniform_mdot.py CONTROL.pvtu UNIFORM.pvtu [--out PATH] [--tag 075]
"""

import argparse
import os

import numpy as np


def interface_shape(fname):
    """Radius, cubic invariant and azimuth over the liquid-side interface band."""
    from analyze_anisotropy_attribution import load_lattice, fit_bubble
    d = load_lattice(fname)
    fit = fit_bubble(d)
    vof = d["vof"]

    # liquid cells with at least one vapour face-neighbour: the band on which
    # the mass transfer gradient is actually taken
    is_vap = vof < 0.5
    nb = np.zeros_like(is_vap)
    for ax in range(3):
        nb |= np.roll(is_vap, 1, ax) | np.roll(is_vap, -1, ax)
    mask = (~is_vap) & nb

    q = (fit["P"] - fit["ctr"])[mask]
    r = np.linalg.norm(q, axis=-1)
    n = q / r[:, None]
    s = (n ** 4).sum(-1)
    theta = np.arctan2(q[:, 1], q[:, 0])
    return r, s, theta


def radial_label_angle(theta_bins, profiles, ticks, keep_off_deg=15.0,
                       window=1):
    """Angle (degrees) at which the radial tick labels sit farthest from the
    plotted profiles: maximise, over the bin centres, the distance between the
    profiles (taken over `window` neighbouring bins, since a label has
    angular extent) and the nearest labelled ring, staying `keep_off_deg`
    away from the angular labels at multiples of 90 degrees."""
    ticks = np.asarray(ticks, dtype=float)
    profs = [np.asarray(p, dtype=float) for p in profiles]
    n = len(theta_bins)
    best_ang, best_gap = 22.5, -1.0
    for i, th in enumerate(theta_bins):
        deg = np.degrees(th) % 360.0
        off = min(abs(((deg - k) + 180.0) % 360.0 - 180.0)
                  for k in range(0, 360, 90))
        if off < keep_off_deg:
            continue
        idx = [(i + j) % n for j in range(-window, window + 1)]
        vals = np.concatenate([p[idx] for p in profs])
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        gap = min(np.min(np.abs(ticks - v)) for v in vals)
        if gap > best_gap:
            best_gap, best_ang = gap, deg
    return best_ang


def draw(data, out):
    """The two-panel figure from `data` (label -> dict(r, s, th, c0, c1,
    col, mk)), separated from the loading so that it can be checked without
    the field data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'font.family': 'serif',
                         'savefig.dpi': 300, 'axes.grid': True,
                         'grid.alpha': 0.3, 'lines.linewidth': 1.4})

    fig = plt.figure(figsize=(7.4, 3.3))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='polar')

    # ---- (a) radius against the cubic invariant --------------------------
    edges = np.linspace(1.0 / 3.0, 1.0, 19)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    for lbl, D in data.items():
        idx = np.clip(np.digitize(D["s"], edges) - 1, 0, len(ctr) - 1)
        prof = np.array([D["r"][idx == b].mean() if (idx == b).any() else np.nan
                         for b in range(len(ctr))])
        ax1.plot(ctr, prof * 1e6, D["mk"], color=D["col"], ms=3.2,
                 ls='none', label=lbl)
        ss = np.linspace(1 / 3, 1, 50)
        ax1.plot(ss, (D["c0"] + D["c1"] * ss) * 1e6, '-', color=D["col"], lw=1.3)
    ax1.set_xlabel(r"cubic invariant $s = n_x^4+n_y^4+n_z^4$")
    ax1.set_ylabel(r"interface radius [$\mu$m]")
    ax1.set_xticks([1 / 3, 0.5, 1.0])
    ax1.set_xticklabels(["1/3\nbody diag.", "1/2\nedge diag.", "1\naxis"], fontsize=8)
    ax1.set_xlim(0.30, 1.03)
    ax1.legend(frameon=False, loc="lower left", fontsize=8)
    ax1.set_title("(a) shape against mesh orientation", fontsize=9)

    # ---- (b) the same data as a polar profile ----------------------------
    nb = 72
    eth = np.linspace(-np.pi, np.pi, nb + 1)
    cth = 0.5 * (eth[1:] + eth[:-1])
    profiles = []
    for lbl, D in data.items():
        idx = np.clip(np.digitize(D["th"], eth) - 1, 0, nb - 1)
        prof = np.array([D["r"][idx == b].mean() if (idx == b).any() else np.nan
                         for b in range(nb)]) / D["r"].mean()
        profiles.append(prof)
        t = np.concatenate([cth, cth[:1]])
        ax2.plot(t, np.concatenate([prof, prof[:1]]), color=D["col"], label=lbl)
    ticks = [0.95, 1.0, 1.05]
    ax2.set_ylim(0.93, 1.07)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(["0.95", "1", "1.05"], fontsize=7)
    # The binned profiles span only 0.98 to 1.02, so the 0.95 and 1.05 rings
    # carry no data near them and their labels have nothing to sit against
    # unless the rings themselves are visible.  The angular spokes stay faint
    # so that they do not compete, and R = <R> is dashed: fixing the middle
    # ring is what tells the reader which of the other two is which.
    ax2.yaxis.grid(True, color='0.55', linewidth=0.6, alpha=1.0)
    ax2.xaxis.grid(True, color='0.85', linewidth=0.6, alpha=1.0)
    for line, t in zip(ax2.yaxis.get_gridlines(), ticks):
        if t == 1.0:
            line.set_linestyle((0, (4, 2)))
    # radial values where they sit farthest from both curves.  They are drawn
    # by hand rather than through set_rlabel_position, whose radial pad pushes
    # a label clear of the ring it names: on an axes this small that put 0.95
    # at a radius reading 0.967 and 1.05 on top of the outer spine.  Anchored
    # on the ring, each number sits in the gap it labels.
    ang = radial_label_angle(cth, profiles, ticks)
    ax2.set_rlabel_position(ang)
    ax2.set_yticklabels([])
    for t, txt in zip(ticks, ["0.95", "1", "1.05"]):
        ax2.text(np.radians(ang), t, txt, fontsize=7, ha='center', va='center',
                 zorder=5, bbox=dict(facecolor='white', edgecolor='none',
                                     alpha=0.9, pad=0.15))
    print(f"  (b) radial labels {ticks} at {ang:.1f} deg")
    ax2.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax2.set_xticklabels([r"$0^\circ$", "", r"$90^\circ$", "", r"$180^\circ$", "", r"$270^\circ$", ""], fontsize=7)
    ax2.set_title(r"(b) $R(\theta)/\langle R\rangle$ by azimuth",
                  fontsize=9, pad=14)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
    print(f"\n  figure -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("control")
    ap.add_argument("uniform")
    ap.add_argument("--out", default="pics/uniform_mdot.pdf")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    runs = [("Computed $\\dot{m}$", a.control, "0.25", "o"),
            ("Uniform $\\dot{m}$",  a.uniform, "C3", "s")]
    data = {}
    for lbl, fn, col, mk in runs:
        r, s, th = interface_shape(fn)
        A = np.vstack([np.ones_like(s), s]).T
        c0, c1 = np.linalg.lstsq(A, r, rcond=None)[0]
        data[lbl] = dict(r=r, s=s, th=th, c0=c0, c1=c1, col=col, mk=mk)
        print(f"  {lbl:<22s} n={len(r):6d}  <R>={r.mean()*1e6:8.3f} um  "
              f"dR/ds={c1*1e6:+9.4f} um  normalised={c1/r.mean():+.4f}  "
              f"axis-diagonal={100*(c1*(1-1/3))/r.mean():+.3f}%")

    draw(data, a.out)


if __name__ == "__main__":
    main()
