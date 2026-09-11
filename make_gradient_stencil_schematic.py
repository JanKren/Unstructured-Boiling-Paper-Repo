#!/usr/bin/env python3
"""
Schematic of the front-modified least-squares gradient reconstruction
on polyhedral (Voronoi) cells.

Shows:
  - Irregular polyhedral cells with phase colouring (vapour / liquid)
  - Curved interface (alpha = 0.5) cutting through cells
  - One highlighted cell with displacement vectors:
      Blue arrows  -> unmodified r_cc' to same-phase neighbours
      Red arrows   -> shortened  r_cf  to front intersection points
      Red dashed   -> removed portion beyond the front
  - Temperature labels: T_c, T_c', T_sat
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon as MplPolygon
from matplotlib.lines import Line2D
from scipy.spatial import Voronoi
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

# ── Interface: large-radius circle -> gentle arc through mid-domain ──
CX, CY, R_INTF = 0.5, -1.65, 2.0


def interface_y(x):
    """y-coordinate of the interface arc at horizontal position x."""
    return CY + np.sqrt(np.maximum(R_INTF**2 - (x - CX)**2, 0))


def is_liquid(x, y):
    """Liquid phase is above the interface."""
    return y > interface_y(x)


def segment_circle_intersection(x1, y1, x2, y2):
    """First intersection of segment (x1,y1)-(x2,y2) with interface circle."""
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - CX, y1 - CY
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - R_INTF**2
    disc = b * b - 4 * a * c
    if disc < 0 or a < 1e-15:
        return None
    sq = np.sqrt(disc)
    for t in [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]:
        if 0.0 <= t <= 1.0:
            return (x1 + t * dx, y1 + t * dy)
    return None


# ── Sutherland-Hodgman polygon clipping ──
def _clip_side(poly, inside_fn, intersect_fn):
    out = []
    n = len(poly)
    for i in range(n):
        curr, prev = poly[i], poly[i - 1]
        c_in, p_in = inside_fn(curr), inside_fn(prev)
        if p_in:
            if c_in:
                out.append(curr)
            else:
                out.append(intersect_fn(prev, curr))
        elif c_in:
            out.append(intersect_fn(prev, curr))
            out.append(curr)
    return out


def clip_polygon(verts, box):
    """Clip polygon vertices to [xmin, xmax, ymin, ymax]."""
    xmin, xmax, ymin, ymax = box

    def lx(a, b, x):
        if abs(b[0] - a[0]) < 1e-15:
            return [x, a[1]]
        t = (x - a[0]) / (b[0] - a[0])
        return [x, a[1] + t * (b[1] - a[1])]

    def ly(a, b, y):
        if abs(b[1] - a[1]) < 1e-15:
            return [a[0], y]
        t = (y - a[1]) / (b[1] - a[1])
        return [a[0] + t * (b[0] - a[0]), y]

    p = [[v[0], v[1]] for v in verts]
    p = _clip_side(p, lambda q: q[0] >= xmin, lambda a, b: lx(a, b, xmin))
    p = _clip_side(p, lambda q: q[0] <= xmax, lambda a, b: lx(a, b, xmax))
    p = _clip_side(p, lambda q: q[1] >= ymin, lambda a, b: ly(a, b, ymin))
    p = _clip_side(p, lambda q: q[1] <= ymax, lambda a, b: ly(a, b, ymax))
    return np.array(p) if len(p) >= 3 else None


def main():
    # ── Voronoi seeds: perturbed grid ──
    np.random.seed(42)
    nx, ny = 6, 6
    sp = 1.0 / nx
    real_seeds = []
    for i in range(nx):
        for j in range(ny):
            x = (i + 0.5) * sp + np.random.uniform(-0.055, 0.055)
            y = (j + 0.5) * sp + np.random.uniform(-0.055, 0.055)
            real_seeds.append([x, y])

    n_real = len(real_seeds)

    # Mirror padding to produce bounded interior regions
    all_seeds = list(real_seeds)
    for s in real_seeds:
        all_seeds.append([s[0], -s[1]])
        all_seeds.append([s[0], 2.0 - s[1]])
        all_seeds.append([-s[0], s[1]])
        all_seeds.append([2.0 - s[0], s[1]])

    seeds = np.array(all_seeds)
    vor = Voronoi(seeds)

    BOX = [0.0, 1.0, 0.0, 1.0]

    # ── Build cell data ──
    cells = {}
    for idx in range(n_real):
        reg = vor.regions[vor.point_region[idx]]
        if -1 in reg or len(reg) < 3:
            continue
        clipped = clip_polygon(vor.vertices[reg], BOX)
        if clipped is None:
            continue
        sx, sy = seeds[idx]
        cells[idx] = {
            'vertices': clipped,
            'center': (sx, sy),
            'liquid': is_liquid(sx, sy),
        }

    # ── Neighbour map & ridge data ──
    nbrs = {i: [] for i in cells}
    ridges = {}
    for ri, (i, j) in enumerate(vor.ridge_points):
        if i in cells and j in cells:
            nbrs[i].append(j)
            nbrs[j].append(i)
            rv = vor.ridge_vertices[ri]
            if -1 not in rv:
                v1, v2 = vor.vertices[rv[0]], vor.vertices[rv[1]]
                ridges[(i, j)] = (v1, v2)
                ridges[(j, i)] = (v1, v2)

    # ── Choose the highlighted cell ──
    # Liquid cell near interface, central in domain, with cross-phase neighbours
    best = None
    best_score = -np.inf
    for idx, cell in cells.items():
        if not cell['liquid']:
            continue
        cx, cy = cell['center']
        d = cy - interface_y(cx)
        if d < 0.02 or d > 0.14:
            continue
        n_same = sum(1 for n in nbrs[idx]
                     if n in cells and cells[n]['liquid'])
        n_cross = sum(1 for n in nbrs[idx]
                      if n in cells and not cells[n]['liquid'])
        if n_same < 1 or n_cross < 1:
            continue
        # Score: prefer central x, close to interface, many neighbours
        centrality = 1.0 - 4.0 * (cx - 0.5)**2   # peaks at x=0.5
        closeness = 1.0 - d / 0.18                 # peaks at interface
        richness = (n_same + n_cross) / 6.0         # more nbrs = better
        score = centrality + 0.5 * closeness + 0.3 * richness
        if score > best_score:
            best_score = score
            best = idx

    # ── Create figure ──
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.axis('off')

    COL_LIQ = '#C8DEF0'
    COL_VAP = '#FFF8E7'
    COL_HI = '#FFE0B2'
    COL_INT = '#2E7D32'

    # ── Cell fills ──
    for idx, cell in cells.items():
        if idx == best:
            fc, ec, lw = COL_HI, '#D84315', 2.2
        else:
            fc = COL_LIQ if cell['liquid'] else COL_VAP
            ec, lw = '#666666', 0.8
        ax.add_patch(MplPolygon(cell['vertices'], closed=True,
                                facecolor=fc, edgecolor=ec,
                                linewidth=lw, zorder=1))

    # ── Cell centre dots ──
    for idx, cell in cells.items():
        cx, cy = cell['center']
        if idx == best:
            ax.plot(cx, cy, 'ko', ms=6, zorder=10)
        else:
            ax.plot(cx, cy, '.', color='#999999', ms=3, zorder=2)

    # ── Interface arc ──
    x_arc = np.linspace(-0.02, 1.02, 300)
    y_arc = interface_y(x_arc)
    ax.plot(x_arc, y_arc, color=COL_INT, linewidth=3.5, zorder=4)
    ax.text(0.88, interface_y(0.88) + 0.03, r'$\alpha = 0.5$',
            fontsize=13, color=COL_INT, ha='center', va='bottom',
            fontweight='bold', zorder=6)

    # ── Phase labels ──
    ax.text(0.12, 0.88, 'Liquid', fontsize=24, color='#1F4E79',
            ha='center', va='center', fontweight='bold', alpha=0.5)
    ax.text(0.88, 0.12, 'Vapour', fontsize=24, color='#8B6914',
            ha='center', va='center', fontweight='bold', alpha=0.5)

    # ── Stencil arrows for the highlighted cell ──
    if best is not None:
        hx, hy = cells[best]['center']

        # Collect all arrows first to choose best ones for labelling
        blue_arrows = []   # (nb_x, nb_y) for same-phase
        red_arrows = []    # (int_pt, nb_x, nb_y) for cross-phase

        for nb in nbrs[best]:
            if nb not in cells:
                continue
            nbx, nby = cells[nb]['center']
            same_phase = cells[nb]['liquid'] == cells[best]['liquid']

            if not same_phase:
                # x_f is found by linear interpolation along the
                # cell-centre connection line (not on the face itself)
                int_pt = segment_circle_intersection(hx, hy, nbx, nby)
                if int_pt is not None:
                    red_arrows.append((int_pt, nbx, nby))
            else:
                blue_arrows.append((nbx, nby))

        # Draw all blue arrows
        for i, (bx, by) in enumerate(blue_arrows):
            ax.add_patch(FancyArrowPatch(
                posA=(hx, hy), posB=(bx, by),
                arrowstyle='->,head_width=5,head_length=5',
                color='blue', linewidth=2.5, mutation_scale=1, zorder=8))
            ax.plot(bx, by, 'bs', ms=7, zorder=9)

        # Draw all red arrows
        for i, (ipt, nbx, nby) in enumerate(red_arrows):
            # Full c -> c' connection line (thin gray) to show
            # that x_f lies on this line by linear interpolation
            ax.plot([hx, nbx], [hy, nby], color='#AAAAAA',
                    ls='-', lw=0.8, zorder=6)
            # Red arrow: shortened to front intersection
            ax.add_patch(FancyArrowPatch(
                posA=(hx, hy), posB=ipt,
                arrowstyle='->,head_width=5,head_length=5',
                color='red', linewidth=2.5, mutation_scale=1, zorder=8))
            ax.plot(*ipt, 'r*', ms=15, zorder=9)
            # Dashed continuation beyond front to neighbour
            ax.plot([ipt[0], nbx], [ipt[1], nby],
                    'r--', lw=1.2, alpha=0.5, zorder=7)
            ax.plot(nbx, nby, 'o', color='#AAAAAA', ms=5, zorder=7)

        # ── Labels with offset annotations to avoid overlap ──

        # T_P at cell centre — offset to the right (away from arrows)
        ax.annotate(r'$T_P$', xy=(hx, hy),
                    xytext=(hx + 0.08, hy - 0.01), fontsize=15,
                    fontweight='bold', ha='left', va='center', zorder=11,
                    arrowprops=dict(arrowstyle='-', color='black',
                                    lw=0.8, shrinkA=2, shrinkB=2))

        # r_PQ label on the most upward-going blue arrow
        if blue_arrows:
            # Pick the arrow going most clearly upward (highest by)
            bi = max(range(len(blue_arrows)),
                     key=lambda k: blue_arrows[k][1] - hy)
            bx, by = blue_arrows[bi]
            mx, my = (hx + bx) / 2, (hy + by) / 2
            dx, dy = bx - hx, by - hy
            norm = np.hypot(dx, dy)
            # Perpendicular offset (to the right of the arrow direction)
            px, py = dy / norm * 0.04, -dx / norm * 0.04
            ax.text(mx + px, my + py, r"$\mathbf{r}_{PQ}$", fontsize=16,
                    color='blue', ha='center', va='center', zorder=11,
                    bbox=dict(facecolor='white', edgecolor='none',
                              alpha=0.8, pad=1.5))

            # T_Q at the topmost blue endpoint
            ax.annotate(r"$T_Q$", xy=(bx, by),
                        xytext=(bx - 0.06, by + 0.04), fontsize=14,
                        color='#0D47A1', ha='right', va='bottom', zorder=11,
                        arrowprops=dict(arrowstyle='-', color='#0D47A1',
                                        lw=0.8, shrinkA=2, shrinkB=2))

        # r_Pf label on the red arrow — offset to the left
        if red_arrows:
            ri = 0  # usually only 1 red arrow now
            ipt, _, _ = red_arrows[ri]
            mx, my = (hx + ipt[0]) / 2, (hy + ipt[1]) / 2
            dx, dy = ipt[0] - hx, ipt[1] - hy
            norm = max(np.hypot(dx, dy), 1e-10)
            # Perpendicular offset to the left
            px, py = dy / norm * 0.04, -dx / norm * 0.04
            ax.text(mx + px, my + py, r'$\mathbf{r}_{Pf}$', fontsize=16,
                    color='red', ha='center', va='center', zorder=11,
                    bbox=dict(facecolor='white', edgecolor='none',
                              alpha=0.8, pad=1.5))

            # T_sat at intersection — offset into vapour region
            ax.annotate(r'$T_\mathrm{sat}$', xy=ipt,
                        xytext=(ipt[0] + 0.07, ipt[1] - 0.04),
                        fontsize=14, color='#B71C1C',
                        ha='left', va='top', zorder=11,
                        arrowprops=dict(arrowstyle='-', color='#B71C1C',
                                        lw=0.8, shrinkA=2, shrinkB=2))

    # ── Legend ──
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2.5, marker='s', ms=7,
               markerfacecolor='blue', label=r'Unmodified $\mathbf{r}_{PQ}$'),
        Line2D([0], [0], color='red', lw=2.5, marker='*', ms=13,
               markerfacecolor='red',
               label=r'Interface-shortened $\mathbf{r}_{Pf}$'),
        Line2D([0], [0], color=COL_INT, lw=3, label='Interface'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
              framealpha=0.9, edgecolor='#CCCCCC')

    # ── Save ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        outpath = os.path.join(OUTPUT_DIR,
                               f"gradient_stencil_schematic.{ext}")
        plt.savefig(outpath, bbox_inches='tight', facecolor='white')
    print("Saved gradient_stencil_schematic.png/.pdf")
    plt.close()


if __name__ == "__main__":
    main()
