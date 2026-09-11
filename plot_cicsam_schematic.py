#!/usr/bin/env python3
"""Two-dimensional schematic of the CICSAM stencil at two interface orientations
and two instants of the donor's filling.

    python3 plot_cicsam_schematic.py

Companion to the gradient-stencil schematic of Section 3: the same kind of
picture, for the limiter rather than for the temperature gradient.

  Top row, the donor half full.  (a) a mesh axis along the interface normal n;
  (b) the mesh turned by 45 degrees about the donor centre while n and the front
  stay where they are.  In both the stencil reads (1, 0.5, 0): identical inputs.

  Bottom row, the same two meshes with the donor 90% full.  (c) on the axis the
  acceptor is still empty and the D-A face is still dry: the front reaches that
  face only when the donor is full.  (d) on the diagonal the D-A face is the
  edge running from D's bottom corner to its right corner, its near end is level
  with the donor centre, so the front has been crossing it since the donor was
  half full: 55% of the face is wetted and the acceptor already holds 0.153.

  Both panels of a row are drawn with n pointing to the right.  The mesh, not
  the normal, is what is rotated between them; this is the same geometry as
  rotating n on a fixed mesh, and it means "along n" is "horizontal" in every
  panel.  W, the extent of the donor along n, is the distance the front travels
  while the donor fills (one cell width on an axis, sqrt(2) on the diagonal);
  lambda is the advance of the front from one cell centre of the stencil to the
  next (h on the axis, h cos 45 on the diagonal).  The D-A face is drawn thick,
  its wetted part in orange.

Volume fractions and wetted fractions are closed-form for the geometry drawn
(a strip cut by a line on the axis, a diamond cut by a line on the diagonal),
so the numbers shown are exact rather than illustrative.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR",
                            os.path.join(os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling"), "pics"))

plt.rcParams.update({"font.size": 11, "font.family": "serif"})

XCLIP, YCLIP = 2.55, 1.25                  # window the cells are cut to


def alpha_cell(i, j, th, s0):
    """Area fraction of the unit cell (i,j) lying on the alpha side of the
    interface, i.e. behind the front, for a mesh at angle th to the normal.
    Closed form for the two orientations drawn (th = 0 and th = 45 degrees):
    a strip for the axis, a diamond cut by a vertical line for the diagonal."""
    e1 = np.array([np.cos(th), -np.sin(th)])
    e2 = np.array([np.sin(th), np.cos(th)])
    xc = (i * e1 + j * e2)[0]                      # cell centre, along n
    u = s0 - xc
    if abs(np.sin(th)) < 1e-12:                    # axis: unit square
        return float(np.clip(u + 0.5, 0.0, 1.0))
    a = 0.5 * (abs(np.cos(th)) + abs(np.sin(th)))  # half-extent along n
    if abs(abs(np.cos(th)) - abs(np.sin(th))) > 1e-12:
        raise ValueError("closed form implemented for 0 and 45 degrees only")
    if u <= -a:
        return 0.0
    if u >= a:
        return 1.0
    return float((a + u) ** 2 if u <= 0 else 1.0 - (a - u) ** 2)


def front_for_donor_fill(th, target):
    """Front position s0 (drawn frame, along n) at which the donor holds
    `target` of the phase behind the front; inverse of alpha_cell(0, 0)."""
    if abs(np.sin(th)) < 1e-12:
        return target - 0.5
    a = 0.5 * (abs(np.cos(th)) + abs(np.sin(th)))
    return np.sqrt(target) - a if target <= 0.5 else a - np.sqrt(1.0 - target)


def cells(ax, ang_deg, title, fill, show_lengths):
    """One panel: the cell block, the stencil, the D-A face with its wetted
    part, and optionally W and lambda as the two horizontal lengths they are."""
    th = np.radians(ang_deg)
    e1 = np.array([np.cos(th), -np.sin(th)])       # mesh x-axis, drawn frame
    e2 = np.array([np.sin(th), np.cos(th)])        # mesh y-axis, drawn frame
    W = abs(np.cos(th)) + abs(np.sin(th))          # donor's extent along n
    dd = abs(np.cos(th))                           # centre-to-centre, along n
    s_off = front_for_donor_fill(th, fill)

    clip = plt.Rectangle((-XCLIP, -YCLIP), 2 * XCLIP, 2 * YCLIP,
                         transform=ax.transData, fc="none", ec="none")
    ax.add_patch(clip)

    rng = range(-4, 5)
    for i in rng:
        for j in rng:
            c = i * e1 + j * e2
            if abs(c[0]) > XCLIP + 1.2 or abs(c[1]) > YCLIP + 1.2:
                continue
            a = alpha_cell(i, j, th, s_off)
            poly = plt.Polygon([c - .5 * e1 - .5 * e2, c + .5 * e1 - .5 * e2,
                                c + .5 * e1 + .5 * e2, c - .5 * e1 + .5 * e2],
                               fc=plt.cm.Blues(0.08 + 0.42 * a),
                               ec="0.6", lw=0.9, zorder=0)
            poly.set_clip_path(clip)
            ax.add_patch(poly)

    # the D-A face: the edge of D perpendicular to e1, and its wetted part
    p1 = 0.5 * e1 - 0.5 * e2
    p2 = 0.5 * e1 + 0.5 * e2
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="0.15", lw=4.2, zorder=2,
            solid_capstyle="butt")
    if abs(p2[0] - p1[0]) < 1e-9:
        wet = 1.0 if s_off > p1[0] else 0.0
    else:
        wet = float(np.clip((s_off - p1[0]) / (p2[0] - p1[0]), 0.0, 1.0))
    if wet > 0:
        q = p1 + wet * (p2 - p1)
        ax.plot([p1[0], q[0]], [p1[1], q[1]], color="C1", lw=4.2, zorder=3,
                solid_capstyle="butt")

    ln, = ax.plot([s_off, s_off], [-YCLIP, YCLIP], color="C0", lw=3.2,
                  zorder=4, solid_capstyle="butt")
    ln.set_clip_path(clip)

    if show_lengths:
        # the two extreme front positions: donor exactly empty, exactly full
        for xv, lab in ((-W / 2, r"$\alpha_D=0$"), (W / 2, r"$\alpha_D=1$")):
            ax.plot([xv, xv], [-YCLIP - 0.06, 1.27], color="C0", lw=1.2,
                    ls=(0, (5, 3)), alpha=0.85, zorder=4)
            ax.annotate(lab, xy=(xv, 1.31), ha="center", va="bottom",
                        fontsize=9, color="C0")
        ax.annotate("", xy=(W / 2, 1.66), xytext=(-W / 2, 1.66),
                    arrowprops=dict(arrowstyle="<|-|>", color="C0", lw=1.8,
                                    shrinkA=0, shrinkB=0), zorder=6)
        ax.annotate(rf"extent along $\mathbf{{n}}$  $W=h\|\mathbf{{n}}\|_1={W:.2f}h$",
                    xy=(0, 1.77), ha="center", va="bottom", fontsize=10.5,
                    color="C0")

    for k, (cc, lab, col) in enumerate(((-e1, "U", "0.25"), (0 * e1, "D", "C3"),
                                        (e1, "A", "0.25"))):
        a = alpha_cell(k - 1, 0, th, s_off)
        ax.plot(cc[0], cc[1], "o", color=col, ms=8, zorder=6)
        ax.annotate(lab, xy=(cc[0], cc[1] + 0.11), ha="center", va="bottom",
                    fontsize=12.5, color=col, zorder=7, fontweight="bold")
        ha, dx = ("center", 0.0) if (ang_deg < 1 or lab != "D") else ("right", -0.14)
        ax.annotate(f"$\\alpha={a:.3f}$", xy=(cc[0] + dx, cc[1] - 0.13), ha=ha,
                    va="top", fontsize=9.5, color=col, zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.8))

    if show_lengths:
        # lambda: the same step, seen along n
        for cc in (-e1, np.zeros(2)):
            ax.plot([cc[0], cc[0]], [min(cc[1], 0) - 0.30, -1.42], color="0.35",
                    lw=1.0, ls=(0, (1, 2.5)), zorder=4)
        ax.annotate("", xy=(0, -1.55), xytext=(-dd, -1.55),
                    arrowprops=dict(arrowstyle="<|-|>", color="0.25", lw=1.8,
                                    shrinkA=0, shrinkB=0), zorder=6)
        ax.annotate(rf"advance per cell  $\lambda=h\cos\theta_f={dd:.2f}h$",
                    xy=(-dd / 2, -1.68), ha="center", va="top", fontsize=10.5,
                    color="0.25")
    else:
        ax.annotate(f"face D\N{EN DASH}A: ${100 * wet:.0f}\\%$ wetted",
                    xy=(0.0, -1.62), ha="center", va="top", fontsize=10.5,
                    color="C1" if wet > 0 else "0.25")

    # d, n and the angle between them
    L = 0.80 if ang_deg < 1 else 0.86
    ax.annotate("", xy=tuple(L * e1), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="C2", lw=2.6), zorder=6)
    off = np.array([0.0, 0.30]) if ang_deg < 1 else np.zeros(2)
    ax.annotate("", xy=(L + off[0], off[1]), xytext=tuple(off),
                arrowprops=dict(arrowstyle="-|>", color="C1", lw=2.6), zorder=6)
    if ang_deg < 1:
        ax.annotate(r"$\mathbf{d}$", xy=(0.42, 0.03), color="C2", fontsize=13,
                    ha="center", va="bottom", zorder=7)
        ax.annotate(r"$\mathbf{n}$", xy=(0.42, 0.33), color="C1", fontsize=13,
                    ha="center", va="bottom", zorder=7)
    else:
        ax.annotate(r"$\mathbf{d}$", xy=(0.22, -0.50), color="C2", fontsize=13,
                    ha="center", va="top", zorder=7)
        ax.annotate(r"$\mathbf{n}$", xy=(0.30, 0.06), color="C1", fontsize=13,
                    ha="center", va="bottom", zorder=7)
        arc = np.linspace(-th, 0.0, 60)
        ax.plot(0.58 * np.cos(arc), 0.58 * np.sin(arc), color="k", lw=1.1, zorder=6)
        ax.annotate(r"$\theta_f$", xy=(0.30, -0.24), fontsize=12, zorder=7,
                    ha="left", va="center")

    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlim(-XCLIP - 0.12, XCLIP + 0.12)
    ax.set_ylim(-2.10, 2.05)
    ax.set_aspect("equal")
    ax.axis("off")
    return s_off, wet


def main():
    fig = plt.figure(figsize=(11.6, 9.9))
    gs = fig.add_gridspec(2, 2, hspace=0.12, wspace=0.06)
    out = {}
    out["a"] = cells(fig.add_subplot(gs[0, 0]), 0.0,
                     r"(a) $\mathbf{n}$ on a mesh axis,  $\theta_f=0$,  $\gamma_f=1$;"
                     "  donor half full\n ", 0.5, True)
    out["b"] = cells(fig.add_subplot(gs[0, 1]), 45.0,
                     r"(b) $\mathbf{n}$ on the diagonal,  $\theta_f=45^\circ$,  $\gamma_f=1/2$;"
                     "  donor half full\n" r"     (the mesh is turned, not $\mathbf{n}$)",
                     0.5, True)
    out["c"] = cells(fig.add_subplot(gs[1, 0]), 0.0,
                     r"(c) axis,  donor $90\%$ full" "\n ", 0.9, False)
    out["d"] = cells(fig.add_subplot(gs[1, 1]), 45.0,
                     r"(d) diagonal,  donor $90\%$ full" "\n ", 0.9, False)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"cicsam_schematic.{ext}"),
                    dpi=200, bbox_inches="tight")
    for k, (s, w) in out.items():
        print(f"panel ({k}): front at {s:+.3f} h from the donor centre, D-A face {100 * w:.1f}% wetted")
    print("written: pics/cicsam_schematic.{pdf,png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
