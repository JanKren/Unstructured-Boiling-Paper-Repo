#!/usr/bin/env python3
"""
Schematic: standard vs front-modified gradient stencil on a hex mesh.

Integrated view: temperature profile (top) aligned with cell layout (bottom).
Shows the erf profile, gradient lines, and how the front modification
overshoots the true gradient in the heat-sink limit.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.special import erf
import os

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'savefig.dpi': 300,
})

# --- Parameters (normalised: h = 1, T_sat = 0, ΔT = 1) ---
h = 1.0
d = 0.4 * h           # front-to-cell-centre distance
dT = 3.25 * h          # thermal penetration depth δ_T / h ≈ 3.25

# Positions (interface at x = 0)
x_vap = d - h          # vapour cell centre  (-0.6)
x_P   = d              # Cell P centre       ( 0.4)
x_liq = d + h          # liquid cell centre  ( 1.4)

# Cell face positions (cell width = h)
face = [x_vap - h/2, x_vap + h/2, x_P + h/2, x_liq + h/2]
#       -1.1          -0.1          0.9         1.9

# Temperature profile
x_prof = np.linspace(-1.3, 2.3, 600)
T_prof = np.where(x_prof >= 0, erf(x_prof / dT), 0.0)

# Stencil-point temperatures
T_P_phys = erf(d / dT)          # physical T_P from erf
T_far    = erf((d + h) / dT)    # liquid neighbour

# Gradient magnitudes
g_true  = 2.0 / (np.sqrt(np.pi) * dT)
g_std   = T_far / (2.0 * h)
g_fm_hs = T_far * h / (d**2 + h**2)

# --- Colours ---
c_vap    = '#cce0ff'
c_cellP  = '#ffffff'
c_liq    = '#fff2cc'
c_front  = '#c0392b'
c_std    = '#2471a3'
c_true   = '#222222'
c_erf    = '#666666'

# =====================================================================
fig = plt.figure(figsize=(5.5, 4.5))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.2], hspace=0.02,
                       left=0.12, right=0.97, top=0.95, bottom=0.08)
ax_T = fig.add_subplot(gs[0])
ax_C = fig.add_subplot(gs[1], sharex=ax_T)

# =====================================================================
#  TOP — Temperature profile with gradient lines
# =====================================================================
# erf curve
ax_T.plot(x_prof, T_prof, '-', color=c_erf, lw=2.2, zorder=2,
          label=r'$T(x)$')

# g_std: secant line through the two standard stencil points
ax_T.plot([x_vap, x_liq], [0, T_far], '-', color=c_std, lw=1.8, zorder=3,
          label=r'$g_\mathrm{std}$')

# g_fm: gradient line from interface with heat-sink slope
x_gl = np.linspace(0, 2.0, 50)
ax_T.plot(x_gl, g_fm_hs * x_gl, '-', color=c_front, lw=1.8, zorder=3,
          label=r'$g_\mathrm{fm}$  ($T_P\!\to\!T_\mathrm{sat}$)')

# stencil-point markers
ax_T.plot(x_vap, 0, 'o', color=c_std, ms=7, zorder=5, clip_on=False)
ax_T.plot(x_liq, T_far, 'o', color=c_std, ms=7, zorder=5)

# physical T_P on the erf curve
ax_T.plot(x_P, T_P_phys, 'o', color='#27ae60', ms=7, zorder=5)

# heat-sink T_P at T_sat
ax_T.plot(x_P, 0, 's', color=c_front, ms=7, zorder=5)

# front intersection
ax_T.plot(0, 0, '*', color=c_front, ms=13, zorder=6,
          markeredgecolor='#8b0000', markeredgewidth=0.5)

# labels (positioned to avoid overlap)
ax_T.annotate(r'$T_\mathrm{sat}$', xy=(x_vap, 0),
              xytext=(x_vap - 0.05, 0.06), fontsize=9, color=c_std,
              ha='right')
ax_T.annotate(r'$T_P^{\,\mathrm{phys}}$', xy=(x_P, T_P_phys),
              xytext=(x_P + 0.1, T_P_phys + 0.03), fontsize=9,
              color='#27ae60', ha='left')
ax_T.annotate(r'$T_P \!\approx\! T_\mathrm{sat}$',
              xy=(x_P, 0), xytext=(x_P + 0.1, -0.04),
              fontsize=9, color=c_front, ha='left')
ax_T.annotate(r'$T_\mathrm{far}$', xy=(x_liq, T_far),
              xytext=(x_liq + 0.08, T_far), fontsize=9,
              color='#8b4513', ha='left', va='center')

# interface line (faint)
ax_T.axvline(0, color=c_front, lw=1.0, ls=':', alpha=0.4, zorder=1)

ax_T.set_ylabel(r'$T - T_\mathrm{sat}$', fontsize=11)
ax_T.set_xlim(-1.3, 2.3)
ax_T.set_ylim(-0.08, 0.62)
ax_T.legend(loc='upper left', fontsize=8.5, framealpha=0.9,
            handlelength=1.8)
plt.setp(ax_T.get_xticklabels(), visible=False)

# =====================================================================
#  BOTTOM — Cell schematic (aligned x-axis)
# =====================================================================
ch = 0.45   # cell visual height
y0 = 0.15

# draw cells
for (xl, xr), col in zip(
    [(face[0], face[1]), (face[1], face[2]), (face[2], face[3])],
    [c_vap, c_cellP, c_liq]
):
    ax_C.add_patch(mpatches.Rectangle(
        (xl, y0), xr - xl, ch,
        facecolor=col, edgecolor='#333333', lw=1.4, zorder=1))

# vapour shading inside Cell P (left of front)
ax_C.add_patch(mpatches.Rectangle(
    (face[1], y0), 0 - face[1], ch,
    facecolor=c_vap, edgecolor='none', alpha=0.65, zorder=2))

# front line
ax_C.plot([0, 0], [y0 - 0.02, y0 + ch + 0.02],
          color=c_front, lw=3, zorder=4)

# front star
ax_C.plot(0, y0 + ch / 2, '*', color=c_front, ms=13, zorder=6,
          markeredgecolor='#8b0000', markeredgewidth=0.5)

# cell-centre dots
for x in [x_vap, x_P, x_liq]:
    ax_C.plot(x, y0 + ch / 2, 'o', color='k', ms=4.5, zorder=5)

# phase / cell labels (above cells)
ax_C.text((face[0] + face[1]) / 2, y0 + ch + 0.04, 'Vapour',
          ha='center', va='bottom', fontsize=8, color='#555555')
ax_C.text((0 + face[2]) / 2, y0 + ch + 0.04, 'Cell $P$',
          ha='center', va='bottom', fontsize=8, color='#555555')
ax_C.text((face[2] + face[3]) / 2, y0 + ch + 0.04, 'Liquid',
          ha='center', va='bottom', fontsize=8, color='#555555')

# dimension lines
dy = y0 - 0.05
tk = 0.04
for x1, x2, lbl, c in [(0, x_P, r'$d$', c_front),
                         (x_P, x_liq, r'$h$', '#333333')]:
    ax_C.plot([x1, x2], [dy, dy], '-', color=c, lw=1.1)
    ax_C.plot([x1, x1], [dy - tk, dy + tk], '-', color=c, lw=1.1)
    ax_C.plot([x2, x2], [dy - tk, dy + tk], '-', color=c, lw=1.1)
    ax_C.text((x1 + x2) / 2, dy - 0.06, lbl, ha='center', va='top',
              fontsize=11, color=c, fontweight='bold')

ax_C.set_xlim(-1.3, 2.3)
ax_C.set_ylim(-0.15, y0 + ch + 0.18)
ax_C.set_xlabel(r'Distance from interface  $[h]$', fontsize=10)
ax_C.set_yticks([])
ax_C.spines['top'].set_visible(False)
ax_C.spines['right'].set_visible(False)
ax_C.spines['left'].set_visible(False)

# =====================================================================
out = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(out, 'gradient_overestimate_schematic.pdf'),
            bbox_inches='tight')
plt.savefig(os.path.join(out, 'gradient_overestimate_schematic.png'),
            bbox_inches='tight')
plt.close()
print("Saved: gradient_overestimate_schematic.pdf/.png")
