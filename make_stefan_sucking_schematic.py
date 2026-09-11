#!/usr/bin/env python3
"""
Stefan & Sucking problem schematic (Figure 2).

Single-axes layout:
  - Top region:   T-x graph with arrowed T and x axes
  - x-axis:       sits between graph and domain, with 0, x_gamma, x_b labels
  - Bottom region: physical domain box (Wall | Vapour | Liquid | Outlet)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.special import erf
import os

plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'savefig.dpi': 300,
})

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(os.environ.get("PAPER_ROOT", "/home/jan/papers/UnstructuredBoiling"), "pics"))

# ── Horizontal positions (shared between graph and domain) ──
x_wall = 0.0
x_intf = 0.35
x_out  = 1.0

# ── Vertical layout (in data coordinates of one axes) ──
# Domain box: y = -0.65 to -0.25
# x-axis:     y = -0.15  (between domain top and graph bottom)
# Graph:      y =  0.0 to 0.90  (temperature curves)

DOM_BOT  = -0.65
DOM_TOP  = -0.25
X_AXIS_Y = -0.12
T_ORIGIN =  0.0    # bottom of T-x graph area
T_TOP    =  0.90   # top of T-x graph area

# ── Temperature levels (mapped to y-coordinates in graph region) ──
T_sat = 0.20
T_hi  = 0.70       # T_wall for Stefan / T_out for Sucking

wall_w = 0.03      # hatching strip width
out_w  = 0.03


def stefan_profile(x):
    """Stefan: linear decrease in vapour, constant T_sat in liquid."""
    return np.where(
        x <= x_intf,
        T_hi + (T_sat - T_hi) * (x - x_wall) / (x_intf - x_wall),
        T_sat)


def sucking_profile(x):
    """Sucking: constant T_sat in vapour, erf rise in liquid."""
    sigma = 0.10
    return np.where(
        x <= x_intf,
        T_sat,
        T_sat + (T_hi - T_sat) * erf((x - x_intf) / sigma))


def axis_arrow(ax, start, end, **kw):
    """Draw an axis line with arrowhead."""
    ax.add_patch(FancyArrowPatch(
        posA=start, posB=end,
        arrowstyle='->,head_width=4,head_length=6',
        color='black', linewidth=1.2, mutation_scale=1, zorder=5, **kw))


def main():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(-0.12, 1.14)
    ax.set_ylim(DOM_BOT - 0.12, T_TOP + 0.08)
    ax.axis('off')

    # ==================================================================
    #  TEMPERATURE PROFILES (upper region)
    # ==================================================================
    x = np.linspace(x_wall, x_out, 500)

    ax.plot(x, stefan_profile(x), 'r-', linewidth=2.8, zorder=3,
            clip_on=False)
    ax.plot(x, sucking_profile(x), 'b-', linewidth=2.8, zorder=3,
            clip_on=False)

    # ── T axis (vertical arrow) ──
    axis_arrow(ax, (x_wall, T_ORIGIN - 0.02), (x_wall, T_TOP))
    ax.text(x_wall - 0.03, T_TOP, r'$T$', fontsize=16,
            ha='right', va='top')

    # ── x axis (horizontal arrow, between graph and domain) ──
    axis_arrow(ax, (x_wall - 0.03, X_AXIS_Y),
               (x_out + 0.08, X_AXIS_Y))
    ax.text(x_out + 0.10, X_AXIS_Y, r'$x$', fontsize=16,
            ha='left', va='center')

    # Tick marks on x-axis
    for xp, lab in [(x_wall, r'$0$'), (x_intf, r'$x_\gamma$'),
                     (x_out, r'$x_b$')]:
        ax.plot([xp, xp], [X_AXIS_Y - 0.02, X_AXIS_Y + 0.02],
                'k-', linewidth=1.0, zorder=5)
        ax.text(xp, X_AXIS_Y - 0.04, lab, fontsize=14,
                ha='center', va='top')

    # Dashed vertical at interface (from x-axis up to T_sat)
    ax.plot([x_intf, x_intf], [X_AXIS_Y, T_sat], 'k--',
            linewidth=0.9, zorder=2)

    # T_sat label
    ax.annotate(r'$T_\mathrm{sat}$',
                xy=(x_intf, T_sat),
                xytext=(x_intf + 0.13, T_sat + 0.18),
                fontsize=15, ha='left', va='bottom',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
                zorder=6)

    # Left-side T labels
    ax.text(-0.06, T_hi, r'$T_\mathrm{wall}$', fontsize=14,
            color='red', ha='right', va='center')
    ax.plot([x_wall - 0.01, x_wall + 0.015], [T_hi, T_hi],
            'r-', linewidth=0.8)

    ax.text(-0.06, T_sat, r'$T_\mathrm{wall}$', fontsize=14,
            color='blue', ha='right', va='center')
    ax.plot([x_wall - 0.01, x_wall + 0.015], [T_sat, T_sat],
            'b-', linewidth=0.8)

    # Right-side T labels
    ax.text(x_out + 0.03, T_hi, r'$T_\mathrm{out}$', fontsize=14,
            color='blue', ha='left', va='center')
    ax.text(x_out + 0.03, T_sat, r'$T_\mathrm{out}$', fontsize=14,
            color='red', ha='left', va='center')

    # Legend at top (closer to graph, anchored to avoid overlap)
    ax.text(0.46, 0.99, 'Red:  Stefan problem', fontsize=14,
            color='red', ha='right', va='bottom', fontweight='bold',
            transform=ax.transAxes)
    ax.text(0.54, 0.99, 'Blue:  Sucking problem', fontsize=14,
            color='blue', ha='left', va='bottom', fontweight='bold',
            transform=ax.transAxes)

    # ==================================================================
    #  INTERFACE ARROW (between x-axis and domain)
    # ==================================================================
    intf_arr_y = (X_AXIS_Y + DOM_TOP) / 2 - 0.02
    ax.annotate('', xy=(x_intf + 0.10, intf_arr_y),
                xytext=(x_intf + 0.01, intf_arr_y),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(x_intf + 0.12, intf_arr_y, 'interface', fontsize=14,
            ha='left', va='center', style='italic')

    # ==================================================================
    #  DOMAIN BOX (lower region)
    # ==================================================================
    dom_h = DOM_TOP - DOM_BOT

    # Vapour (white) — starts at x=0 (inner wall face)
    ax.add_patch(Rectangle(
        (x_wall, DOM_BOT), x_intf - x_wall, dom_h,
        facecolor='white', edgecolor='black', linewidth=1.2, zorder=2))

    # Liquid (light blue)
    ax.add_patch(Rectangle(
        (x_intf, DOM_BOT), x_out - x_intf, dom_h,
        facecolor='#C8DEF0', edgecolor='black', linewidth=1.2, zorder=2))

    # Wall hatching — sits to the LEFT of x=0
    ax.add_patch(Rectangle(
        (x_wall - wall_w, DOM_BOT), wall_w, dom_h,
        facecolor='white', edgecolor='black', linewidth=1.2,
        zorder=3, hatch='////'))

    # Outlet arrows — outward-pointing to indicate open boundary
    n_arrows = 5
    arrow_y_pos = np.linspace(DOM_BOT + 0.06, DOM_TOP - 0.06, n_arrows)
    for ay in arrow_y_pos:
        ax.annotate('', xy=(x_out + out_w + 0.01, ay),
                    xytext=(x_out, ay),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    # Interface line inside domain
    ax.plot([x_intf, x_intf], [DOM_BOT, DOM_TOP], 'k-',
            linewidth=1.2, zorder=3)

    # Phase labels
    mid_y = (DOM_BOT + DOM_TOP) / 2
    ax.text((x_wall + x_intf) / 2, mid_y, 'Vapour',
            fontsize=18, ha='center', va='center', fontweight='bold')
    ax.text((x_intf + x_out) / 2, mid_y, 'Liquid',
            fontsize=18, ha='center', va='center',
            fontweight='bold', color='#1F4E79')

    # Boundary labels
    ax.text(x_wall - wall_w - 0.02, mid_y, 'Wall', fontsize=15,
            ha='right', va='center', rotation=90, fontweight='bold')
    ax.text(x_out + out_w + 0.02, mid_y, 'Outlet', fontsize=15,
            ha='left', va='center', rotation=270, fontweight='bold')

    # ── Save ──
    plt.savefig(os.path.join(OUTPUT_DIR, "stefan_sucking_schematic.png"),
                bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, "stefan_sucking_schematic.pdf"),
                bbox_inches='tight', facecolor='white')
    print("Saved stefan_sucking_schematic.png/.pdf")
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()
