#!/usr/bin/env python3
"""What the bubble actually looks like, with and without the transport.

The quantitative figure (`plot_noadvection.py`) reports the axis-to-diagonal
difference as a number.  This one shows the surface it is measured on, coloured
by the local departure from the mean radius, so the cubic symmetry of the
deformation is visible rather than inferred.

Three panels, because two would mislead in opposite directions:

  (a) transport active, on a +-5% scale.  Eight lobes towards the cube
      corners -- the bubble is visibly square.
  (b) transport removed, on the SAME +-5% scale.  Essentially uniform.  This
      is the honest comparison and the headline of the figure.
  (c) transport removed again, on its own +-1% scale.  Without this a reader
      could conclude the residual is zero or that the run is featureless; it
      is neither.  The residual is still faintly cubic, and it is what the
      +1.45% reconstruction area bias produces.

Both surfaces are taken at matched radius (119.6 against 118.9 um) so the
comparison is at matched R/h, on which the four-fold mode collapses.

Usage:  python3 plot_bubble_shape.py [--out pics/bubble_shape.png]
"""

import argparse
import glob
import os

import numpy as np
import pyvista as pv

NOADV = "Data/inflate-unif"          # transport removed
FULL = "Data/unif-transport"         # uniform mdot, transport ACTIVE


def load(path):
    """Front surface with a per-point radial departure from the mean radius."""
    m = pv.read(path).clean()
    sized = m.compute_cell_sizes(length=False, area=True, volume=False)
    a = np.asarray(sized.cell_data["Area"], float)
    cc = np.asarray(sized.cell_centers().points, float)
    ctr = (a[:, None] * cc).sum(0) / a.sum()          # area-weighted centre
    rc = np.linalg.norm(cc - ctr, axis=1)
    R = float((a * rc).sum() / a.sum())               # area-weighted mean radius
    p = np.asarray(m.points, float)
    m.point_data["dev"] = 100.0 * (np.linalg.norm(p - ctr, axis=1) - R) / R
    # normalise to unit radius: the camera would otherwise sit 1 m from a
    # 120 um object and render it as a sub-pixel dot
    m.points = (p - ctr) / R
    return m, R


def pick(directory, target):
    """The saved front closest to a target radius."""
    best = None
    for f in sorted(glob.glob(os.path.join(directory, "*front-ts*.pvtu"))):
        try:
            m, R = load(f)
        except Exception:
            continue
        if m.n_cells == 0:
            continue
        if best is None or abs(R - target) < abs(best[2] - target):
            best = (f, m, R)
    return best


def render(mesh, clim, out, cmap="coolwarm"):
    pl = pv.Plotter(off_screen=True, window_size=(900, 900))
    pl.add_mesh(mesh, scalars="dev", cmap=cmap, clim=clim,
                show_scalar_bar=False, smooth_shading=True,
                specular=0.25, specular_power=18)
    pl.set_background("white")
    # isometric-ish: a cube corner towards the viewer, so the eight lobes of a
    # body-diagonal bulge are all foreshortened equally rather than one facing
    pl.view_vector((1.0, 0.85, 0.75), viewup=(0, 0, 1))
    pl.reset_camera()
    pl.camera.zoom(1.3)
    pl.screenshot(out, transparent_background=False)
    pl.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="pics/bubble_shape.png")
    ap.add_argument("--target", type=float, default=119.0e-6)
    a = ap.parse_args()

    pv.start_xvfb()
    pv.OFF_SCREEN = True

    fn, mn, Rn = pick(NOADV, a.target)
    ff, mf, Rf = pick(FULL, a.target)
    print(f"\n  transport removed : {os.path.basename(fn)}  R = {Rn*1e6:.2f} um")
    print(f"  transport active  : {os.path.basename(ff)}  R = {Rf*1e6:.2f} um")
    print(f"  radius mismatch   : {abs(Rn-Rf)*1e6:.2f} um")
    for lab, m in (("active", mf), ("removed", mn)):
        d = m.point_data["dev"]
        print(f"  {lab:<8} departure from mean radius: "
              f"{d.min():+.2f}% to {d.max():+.2f}%")

    tmp = "/tmp/_bubble"
    os.makedirs(tmp, exist_ok=True)
    render(mf, (-5, 5), f"{tmp}/a.png")
    render(mn, (-5, 5), f"{tmp}/b.png")
    render(mn, (-1, 1), f"{tmp}/c.png")

    # ---- assemble with a colourbar under each panel ----------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    plt.rcParams.update({'font.size': 9, 'font.family': 'serif',
                         'savefig.dpi': 300})

    fig = plt.figure(figsize=(7.4, 3.1))
    panels = [("a.png", (-5, 5), rf"(a) transport active, $R={Rf*1e6:.0f}\,\mu$m"),
              ("b.png", (-5, 5), rf"(b) transport removed, $R={Rn*1e6:.0f}\,\mu$m"),
              ("c.png", (-1, 1), r"(c) as (b), scale $\times 5$")]
    for i, (img, clim, title) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.imshow(plt.imread(f"{tmp}/{img}"))
        ax.set_title(title, fontsize=8.5, pad=4)
        ax.axis("off")
        cax = ax.inset_axes([0.12, -0.04, 0.76, 0.045])
        cb = fig.colorbar(cm.ScalarMappable(colors.Normalize(*clim), "coolwarm"),
                          cax=cax, orientation="horizontal")
        cb.set_ticks([clim[0], 0, clim[1]])
        cb.ax.tick_params(labelsize=7, length=2, pad=1)
        cb.set_label(r"$(r-\langle R\rangle)/\langle R\rangle$ [%]",
                     fontsize=7, labelpad=1)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\n  wrote {a.out} and {a.out.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
