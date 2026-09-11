#!/usr/bin/env python3
"""
Non-orthogonal correction methods schematic for the paper.
Generates overrelaxed.png / .pdf showing three decomposition methods:
  (a) Minimal:     alpha_f = n_f . s_f
  (b) Orthogonal:  alpha_f = 1
  (c) Overrelaxed: alpha_f = 1 / (n_f . s_f)

Each panel shows face point f with n_f decomposed into
  implicit part  alpha_f s_f   and   explicit part  n_f - alpha_f s_f.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Arc
import os

plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'savefig.dpi': 300,
})

# Paths — override via environment variables, e.g.:
#   export OUTPUT_DIR=/path/to/paper/pics
OUTPUT_DIR = os.environ.get("OUTPUT_DIR",
                            os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
#  Geometry setup — 28 deg non-orthogonality angle
# ---------------------------------------------------------------------------
THETA_NF = np.radians(72)   # face-normal direction (from +x)
THETA_SF = np.radians(42)   # cell-centre connection direction
SCALE    = 1.0              # vector length scale

nf_hat = np.array([np.cos(THETA_NF), np.sin(THETA_NF)])
sf_hat = np.array([np.cos(THETA_SF), np.sin(THETA_SF)])
DOT_NS = np.dot(nf_hat, sf_hat)   # n_f . s_f

nf = SCALE * nf_hat
sf = SCALE * sf_hat


def arrow(ax, start, end, color='k', lw=1.6, zorder=3):
    """Draw an arrow using FancyArrowPatch for clean heads."""
    a = FancyArrowPatch(
        posA=tuple(start), posB=tuple(end),
        arrowstyle='->,head_width=5,head_length=5',
        color=color, linewidth=lw, zorder=zorder,
        mutation_scale=1)
    ax.add_patch(a)


def draw_panel(ax, title, alpha_f, lab_impl, lab_expl,
               impl_xytext, expl_xytext):
    """Draw one decomposition panel.

    impl_xytext / expl_xytext : (dx, dy) offset in points for each label,
        relative to the midpoint of the respective vector.
    """

    f = np.array([0.0, 0.0])

    # Implicit and explicit vectors
    v_impl = alpha_f * sf
    v_expl = nf - v_impl
    tip_impl = f + v_impl
    tip_nf   = f + nf

    # --- face line (perpendicular to n_f through f) ---
    tangent = np.array([-nf_hat[1], nf_hat[0]])
    face_half = 0.65
    pf1 = f - face_half * tangent
    pf2 = f + face_half * tangent
    ax.plot([pf1[0], pf2[0]], [pf1[1], pf2[1]],
            'k-', linewidth=2.5, zorder=1, solid_capstyle='round')

    # --- dashed s_f guide (P-Q direction through f) ---
    ext_back = 0.55
    ext_fwd  = max(np.linalg.norm(v_impl) + 0.12, 0.65)
    ax.plot([f[0] - ext_back * sf_hat[0], f[0] + ext_fwd * sf_hat[0]],
            [f[1] - ext_back * sf_hat[1], f[1] + ext_fwd * sf_hat[1]],
            color='gray', linewidth=0.6, linestyle=(0, (4, 3)), zorder=0)

    # --- cell-centre labels P and Q ---
    pP = f - 0.48 * sf_hat
    pQ = f + 0.48 * sf_hat
    ax.plot(*pP, 'ko', markersize=4, zorder=4)
    ax.plot(*pQ, 'ko', markersize=4, zorder=4)
    ax.annotate(r'$P$', pP, fontsize=14, ha='right', va='top',
                xytext=(-6, -5), textcoords='offset points')
    ax.annotate(r'$Q$', pQ, fontsize=14, ha='left', va='bottom',
                xytext=(5, 4), textcoords='offset points')

    # --- n_f arrow (black, thick) ---
    arrow(ax, f, tip_nf, color='k', lw=2.2, zorder=4)
    ax.annotate(r'$\mathbf{n}_f$', tip_nf,
                fontsize=16, ha='left', va='bottom',
                xytext=(5, 3), textcoords='offset points', zorder=5)

    # --- implicit arrow (blue) ---
    arrow(ax, f, tip_impl, color='C0', lw=1.6, zorder=3)
    mid_i = f + 0.5 * v_impl
    ax.annotate(lab_impl, mid_i,
                fontsize=13, color='C0', ha='center', va='center',
                xytext=impl_xytext, textcoords='offset points', zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          ec='none', alpha=0.85))

    # --- explicit arrow (red) ---
    arrow(ax, tip_impl, tip_nf, color='C3', lw=1.6, zorder=3)
    mid_e = 0.5 * (tip_impl + tip_nf)
    ax.annotate(lab_expl, mid_e,
                fontsize=13, color='C3', ha='center', va='center',
                xytext=expl_xytext, textcoords='offset points', zorder=5,
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          ec='none', alpha=0.85))

    # --- dashed parallelogram helper: from tip_nf back along -v_impl ---
    ax.plot([tip_nf[0], tip_nf[0] - v_impl[0]],
            [tip_nf[1], tip_nf[1] - v_impl[1]],
            color='gray', linewidth=0.5, linestyle=':', alpha=0.4, zorder=0)
    # from f along v_expl (completes parallelogram)
    ax.plot([f[0], f[0] + v_expl[0]],
            [f[1], f[1] + v_expl[1]],
            color='gray', linewidth=0.5, linestyle=':', alpha=0.4, zorder=0)

    # --- angle arc between s_f and n_f ---
    arc_r = 0.20
    arc = Arc(f, 2*arc_r, 2*arc_r, angle=0,
              theta1=np.degrees(THETA_SF),
              theta2=np.degrees(THETA_NF),
              color='gray', linewidth=0.7, zorder=2)
    ax.add_patch(arc)

    # --- point f (on top) ---
    ax.plot(*f, 'ko', markersize=6, zorder=6)
    ax.annotate(r'$f$', f, fontsize=15, fontweight='bold',
                ha='right', va='top',
                xytext=(-8, -5), textcoords='offset points', zorder=6)

    ax.set_title(title, fontsize=15, pad=10)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Compute uniform axis limits across all three panels.
    # The overrelaxed case (largest alpha_f) determines the extent.
    alpha_max = 1.0 / DOT_NS
    v_impl_max = alpha_max * sf
    v_expl_max = nf - v_impl_max
    tangent = np.array([-nf_hat[1], nf_hat[0]])
    all_x = [0, nf[0], v_impl_max[0], -0.48*sf_hat[0], 0.48*sf_hat[0],
             -0.65*tangent[0], 0.65*tangent[0],
             nf[0] - v_impl_max[0], v_expl_max[0]]
    all_y = [0, nf[1], v_impl_max[1], -0.48*sf_hat[1], 0.48*sf_hat[1],
             -0.65*tangent[1], 0.65*tangent[1],
             nf[1] - v_impl_max[1], v_expl_max[1]]
    pad = 0.40
    xlim = [min(all_x) - pad, max(all_x) + pad]
    ylim = [min(all_y) - pad, max(all_y) + pad]

    # (a) Minimal:  implicit is short, explicit is large and points "left"
    draw_panel(
        axes[0], '(a) Minimal correction',
        alpha_f=DOT_NS,
        lab_impl=r'$(\mathbf{n}_f \!\cdot\! \mathbf{s}_f)\,\mathbf{s}_f$',
        lab_expl=r'$\mathbf{n}_f - (\mathbf{n}_f \!\cdot\! \mathbf{s}_f)\,\mathbf{s}_f$',
        impl_xytext=(0, -18),
        expl_xytext=(55, 8))

    # (b) Orthogonal:  implicit = s_f, explicit = n_f - s_f
    draw_panel(
        axes[1], '(b) Orthogonal correction',
        alpha_f=1.0,
        lab_impl=r'$\mathbf{s}_f$',
        lab_expl=r'$\mathbf{n}_f - \mathbf{s}_f$',
        impl_xytext=(0, -18),
        expl_xytext=(40, 10))

    # (c) Overrelaxed:  implicit = s_f / (n_f.s_f), longest implicit
    draw_panel(
        axes[2], '(c) Overrelaxed correction',
        alpha_f=1.0 / DOT_NS,
        lab_impl=r'$\mathbf{s}_f / (\mathbf{n}_f \!\cdot\! \mathbf{s}_f)$',
        lab_expl=r'$\mathbf{n}_f - \mathbf{s}_f / (\mathbf{n}_f \!\cdot\! \mathbf{s}_f)$',
        impl_xytext=(0, -18),
        expl_xytext=(65, 8))

    # Apply uniform limits to all panels
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    plt.tight_layout(w_pad=0.5)
    outpath = os.path.join(OUTPUT_DIR, "overrelaxed.png")
    plt.savefig(outpath, bbox_inches='tight', facecolor='white')
    plt.savefig(outpath.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white')
    print(f"Saved: {outpath}")
    print(f"Saved: {outpath.replace('.png', '.pdf')}")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
