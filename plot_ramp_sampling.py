#!/usr/bin/env python3
"""Why the limiter is orientation-dependent before any blending is applied.

    python3 plot_ramp_sampling.py

Section 4 states the ramp width W(n) = h||n||_1 and its consequence, but the
step that makes it intuitive is not drawn anywhere: the limiter samples that
ramp at a FIXED spacing h, so the fraction of the transition it covers per
sample is

    Delta = |n_i| / ||n||_1

which is 1 on an axis, 1/2 on a face diagonal and 1/3 on the body diagonal. The
same physical profile is therefore read at three different effective
resolutions depending on which way the interface faces, and everything the
limiter computes from it inherits that. The blame usually falls on theta_f and
the blend weight, which are downstream of this and, as the frozen-gamma control
shows, secondary to it.

  (a) the ramp itself: alpha of a cubic cell against the plane's normal
      distance, for the three symmetry orientations, with the sample points the
      limiter actually uses marked at their true spacing.

  (b) why withdrawing compression makes it worse. Hyper-C takes the CBC bound
      min(1, alpha_D_tilde/Co), which SATURATES once alpha_D_tilde > Co and
      clips the orientation variation away; ULTIMATE-QUICKEST is linear in
      alpha_D_tilde and passes it through undiminished. That is the mechanism
      behind the monotone ordering in Table 8 -- 3.21% at gamma = 1 rising to
      9.71% at gamma = 0 -- which the paper reports as a correlation.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR",
                            os.path.join(os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling"), "pics"))
CO = 0.2
LADDER_CO = (0.014, 0.020, 0.026, 0.032)   # 075..150 at dt = 1 us,
                                           # from analyze_hyperc_dt.py
NSUB = 160          # sub-samples per direction for the cube-cut volume

plt.rcParams.update({"font.size": 10, "font.family": "serif",
                     "axes.grid": True, "grid.alpha": 0.3})

ORI = [("normal on an axis",  np.array([1.0, 0.0, 0.0]), "C0"),
       ("face diagonal",      np.array([1.0, 1.0, 0.0]), "C1"),
       ("body diagonal",      np.array([1.0, 1.0, 1.0]), "C2")]


def alpha_of_s(n, s_grid):
    """Volume fraction of a unit cube cut by a plane at signed distance s."""
    g = (np.arange(NSUB) + 0.5) / NSUB - 0.5
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    nn = n / np.linalg.norm(n)
    t = np.sort((X * nn[0] + Y * nn[1] + Z * nn[2]).ravel())
    return np.searchsorted(t, s_grid) / t.size


def saturated_fraction_report():
    """Fraction of interfacial faces on which Hyper-C is clipped.

    The paper quotes this figure, so it is computed here rather than by hand.
    alpha_tilde_D is rebuilt exactly as Predict_Beta does -- upwind value
    extrapolated as alfa_a - 2*dotprod with dotprod = dalpha/dx_i * h -- on the
    exact volume fraction of a sphere, and the faces where alpha_tilde_D
    exceeds Co are those where min(1, alpha_tilde_D/Co) has saturated and the
    orientation dependence stops propagating.
    """
    from analyze_anisotropy_attribution import load_lattice, fit_bubble, DEFAULT_CASE
    from analyze_extraction_error import exact_sphere_vof
    from analyze_cicsam_beta import grad_central

    d = load_lattice(DEFAULT_CASE)
    f = fit_bubble(d)
    ctr = np.asarray(f["ctr"], float); R, h = f["R"], d["h"]
    X, Y, Z = d["X"], d["Y"], d["Z"]
    vof = exact_sphere_vof(d, ctr, R, 12, band=3.0)
    g = grad_central(vof, h)

    vals = []
    for axis in range(3):
        lo = [slice(None)] * 3; hi = [slice(None)] * 3
        lo[axis] = slice(0, -1); hi[axis] = slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        fc = [0.5 * (c[lo] + c[hi]) for c in (X, Y, Z)]
        fr = np.sqrt(sum((fc[k] - ctr[k]) ** 2 for k in range(3)))
        pos = ((fc[axis] - ctr[axis]) / np.where(fr > 0, fr, 1.0)) > 0
        pick = lambda a, b: np.where(pos, a, b)
        ad = pick(vof[lo], vof[hi]); aa = pick(vof[hi], vof[lo])
        sg = np.where(pos, 1.0, -1.0)
        gd = np.stack([pick(g[lo][..., k], g[hi][..., k]) for k in range(3)], -1)
        dot = sg * gd[..., axis] * h
        au = np.clip(aa - 2.0 * dot, 0.0, 1.0)
        den = aa - au
        ok = np.abs(den) > 1e-12
        adt = np.where(ok, (ad - au) / np.where(ok, den, 1.0), 0.0)
        sel = ok & (adt >= 0) & (adt <= 1) & (np.linalg.norm(gd, axis=-1) > 1e-30)
        vals.append(adt[sel])
    v = np.concatenate(vals)
    ceiling = 100.0 * (v > 0).mean()
    out = [f"  interfacial faces: {v.size:,}   median alpha_tilde_D = {np.median(v):.3f}",
           f"  ceiling as Co -> 0: {ceiling:.1f}%   ({100 - ceiling:.1f}% of faces carry",
           "  alpha_tilde_D = 0 identically -- the donor cell is already pure, so the",
           "  scheme is upwind there whatever Co is, and they never saturate)",
           "  fraction where Hyper-C is saturated (alpha_tilde_D > Co):"]
    for co in LADDER_CO + (0.20, 0.30, 0.44, 0.60):
        tag = "   <- ladder" if co in LADDER_CO else ""
        out.append(f"     Co = {co:.3f}   {100 * (v > co).mean():5.1f}%{tag}")
    out.append("  The first four are the ladder's OWN Courant numbers, 075 to 150 at")
    out.append("  dt = 1 us (analyze_hyperc_dt.py).  Across them the saturated fraction")
    out.append("  moves under two points, against twelve over Co = 0.2 -> 0.44, because")
    out.append("  the whole ladder sits within four points of the ceiling above.  So the")
    out.append("  clipping does NOT weaken appreciably along the ladder and does not")
    out.append("  account for the narrowing gamma_f = 1 advantage.  Measured directly it")
    out.append("  runs the other way: beta_f's four-fold content at gamma_f = 1 FALLS")
    out.append("  7.06 -> 6.45% over the same range (analyze_cicsam_beta.py --co).")
    return "\n".join(out)


def main():
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.2))

    # ---- (a) one ramp, three sampling rates ------------------------------
    # Plotted against s/W rather than s/h.  The three profiles share their
    # endpoints and their midpoint but not their shape (the axis one is exactly
    # linear, the diagonals sigmoidal); what matters is where the U, D, A
    # samples land, at 1, 1/2 and 1/3 of the ramp width per step.
    s = np.linspace(-1.2, 1.2, 801)
    for lab, n, col in ORI:
        nn = n / np.linalg.norm(n)
        w = np.abs(nn).sum()
        a = alpha_of_s(n, s * w)                  # s given in units of W
        ax[0].plot(s, a, color=col, lw=1.6, alpha=0.85,
                   label=f"{lab}:  $W={w:.2f}h$")
        d_ = abs(nn[0]) / w                       # sample spacing in units of W
        for k in (-1, 0, 1):
            ax[0].plot(k * d_, np.interp(k * d_, s, a), "o", color=col,
                       ms=8, mec="k", mew=0.7, zorder=5)
    ax[0].axvspan(-0.5, 0.5, color="0.9", zorder=0)
    ax[0].set_xlabel(r"offset along the normal, in units of the ramp width $W$")
    ax[0].set_ylabel(r"cell volume fraction  $\alpha$")
    ax[0].set_title("(a)", fontsize=9.5, loc="left")
    ax[0].legend(fontsize=7.5, loc="upper left")
    ax[0].set_xlim(-1.2, 1.2)

    # ---- (c) why removing compression makes it worse ----------------------
    ad = np.linspace(0, 1, 400)
    hyperc = np.minimum(1.0, ad / CO)
    uq = np.minimum(CO * ad + (1 - CO) * (6 * ad + 3) / 8, hyperc)
    ax[1].plot(ad, hyperc, color="C0", lw=2.0,
               label=r"Hyper-C: $\min(1,\ \tilde\alpha_D/\mathrm{Co})$")
    ax[1].plot(ad, uq, color="C3", lw=2.0, label="ULTIMATE-QUICKEST")
    ax[1].axvspan(CO, 1.0, color="C0", alpha=0.10)
    ax[1].axvline(CO, color="k", ls="--", lw=1.0, alpha=0.7)
    ax[1].annotate(r"$\mathrm{Co}$", xy=(CO, 0.05), xytext=(CO + 0.03, 0.05),
                   fontsize=9)
    ax[1].set_xlabel(r"$\tilde\alpha_D$")
    ax[1].set_ylabel(r"$\tilde\alpha_f$")
    ax[1].set_title("(b)", fontsize=9.5, loc="left")
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1.08)

    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"ramp_sampling.{ext}"),
                    dpi=200, bbox_inches="tight")

    print("ramp widths ||n||_1 and sampling rates Delta:")
    for lab, n, _ in ORI:
        nn = n / np.linalg.norm(n)
        print(f"  {lab:<20} ||n||_1 = {np.abs(nn).sum():.4f}   "
              f"Delta = {abs(nn[0])/np.abs(nn).sum():.4f}")
    print()
    print(saturated_fraction_report())
    print("written: pics/ramp_sampling.{pdf,png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
