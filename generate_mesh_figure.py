#!/usr/bin/env python3
"""
Standalone script to regenerate the annular mesh figure (Figure 14).
3-panel PyVista render: (a) domain geometry, (b) B&W cross-section, (c) inner wall mesh.
Uses: Annular/verySmallPart.vtu (full 7.2M-cell mesh)
Coordinate system (r, y, theta) overlaid on panel (a) via matplotlib.
"""
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gc
from pathlib import Path
import os

# Paths — override via environment variables, e.g.:
#   export OUTPUT_DIR=/path/to/paper/pics
#   export ANNULAR_VTU=/path/to/Annular/verySmallPart.vtu
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR",
                                 os.path.dirname(os.path.abspath(__file__))))

pv.OFF_SCREEN = True

# Load full mesh
print("Loading full mesh (7.2M cells)...")
ANNULAR_VTU = os.environ.get("ANNULAR_VTU", "")
if not ANNULAR_VTU:
    raise RuntimeError("ANNULAR_VTU environment variable is not set. "
                       "Point it to the Annular verySmallPart.vtu file.")
mesh = pv.read(ANNULAR_VTU)
print(f"Cells: {mesh.n_cells}, Points: {mesh.n_points}")

# Extract surface and classify boundaries geometrically
print("Extracting surface boundaries...")
surf = mesh.extract_surface()
pts_surf = surf.cell_centers().points
R_surf = np.sqrt(pts_surf[:, 0]**2 + pts_surf[:, 2]**2)
theta_surf = np.degrees(np.arctan2(pts_surf[:, 2], pts_surf[:, 0]))

inner_mask = R_surf < 0.0096          # R < 9.6 mm
outer_mask = R_surf > 0.0155          # R > 15.5 mm
sym_mask = ((theta_surf < 1.0) | (theta_surf > 44.0)) & ~inner_mask & ~outer_mask

inner_wall = surf.extract_cells(np.where(inner_mask)[0])
outer_wall = surf.extract_cells(np.where(outer_mask)[0])
sym_planes = surf.extract_cells(np.where(sym_mask)[0])

print(f"Inner wall: {inner_wall.n_cells} faces")
print(f"Outer wall: {outer_wall.n_cells} faces")
print(f"Symmetry:   {sym_planes.n_cells} faces")

# Cross-section at mid-height
print("Slicing at y = 26 mm...")
y_slice = mesh.slice(normal='y', origin=(0, 0.026, 0))
pts = y_slice.cell_centers().points
R = np.sqrt(pts[:, 0]**2 + pts[:, 2]**2)
y_slice['R_mm'] = R * 1000
print(f"y-slice:    {y_slice.n_cells} faces")

# ---- Extract mesh metrics before freeing memory ----
print("\n" + "=" * 70)
print("MESH QUALITY METRICS")
print("=" * 70)

wall_dist = mesh.cell_data['Grid Wall Distance [m]']
cell_vol  = np.abs(mesh.cell_data['Grid Cell Volume [m^3]'])
cell_ctrs = mesh.cell_data['Grid Cell Centers [m]']
n_nodes   = mesh.cell_data['Grid Number Of Nodes [1]']

R_cell = np.sqrt(cell_ctrs[:, 0]**2 + cell_ctrs[:, 2]**2)
char_edge = cell_vol**(1.0/3.0)

# Cell topology breakdown
n_hex = np.sum(n_nodes == 8)
n_poly = np.sum(n_nodes != 8)
unique_nodes, counts = np.unique(n_nodes, return_counts=True)
print(f"\n1. CELL TOPOLOGY ({mesh.n_cells:,} total)")
for nn, cc in zip(unique_nodes, counts):
    label = "hex" if nn == 8 else "poly"
    print(f"   {nn} nodes ({label}): {cc:,} ({100*cc/mesh.n_cells:.1f}%)")

# Overall cell size
print(f"\n2. OVERALL CELL SIZE")
print(f"   Characteristic edge (V^{{1/3}}):")
print(f"     Min:    {char_edge.min()*1e6:.1f} µm")
print(f"     Mean:   {char_edge.mean()*1e6:.1f} µm")
print(f"     Median: {np.median(char_edge)*1e6:.1f} µm")
print(f"     Max:    {char_edge.max()*1e6:.1f} µm")

# Wall distance
print(f"\n3. WALL DISTANCE (all cells)")
print(f"   Min:    {wall_dist.min()*1e6:.1f} µm")
print(f"   Mean:   {wall_dist.mean()*1e3:.3f} mm")
print(f"   Median: {np.median(wall_dist)*1e3:.3f} mm")

# First cell layer: smallest wall distances
first_layer = wall_dist < np.percentile(wall_dist, 1)  # ~1% closest
print(f"\n4. FIRST CELL LAYER (wall_dist < {np.percentile(wall_dist, 1)*1e6:.1f} µm, {first_layer.sum():,} cells)")
print(f"   Wall distance: {wall_dist[first_layer].min()*1e6:.1f} - {wall_dist[first_layer].max()*1e6:.1f} µm")
print(f"   Cell edge:     {char_edge[first_layer].min()*1e6:.1f} - {char_edge[first_layer].max()*1e6:.1f} µm (mean {char_edge[first_layer].mean()*1e6:.1f} µm)")

# Radial binning
R_WALL = 0.0095  # 9.5 mm
DELTA_MEAN = 0.00019  # 0.19 mm mean film thickness
bins = [
    ("Near inner wall (R < 9.7 mm)",   R_cell < 0.0097),
    ("Film region (R < 9.69 mm)",      R_cell < R_WALL + DELTA_MEAN),
    ("Mid-gap (10–13 mm)",             (R_cell > 0.010) & (R_cell < 0.013)),
    ("Near outer wall (R > 15.5 mm)",  R_cell > 0.0155),
]
print(f"\n5. CELL SIZE BY RADIAL REGION")
for label, mask in bins:
    n = mask.sum()
    if n > 0:
        print(f"   {label}: {n:,} cells")
        print(f"     Edge: min {char_edge[mask].min()*1e6:.1f}, mean {char_edge[mask].mean()*1e6:.1f}, max {char_edge[mask].max()*1e6:.1f} µm")
        print(f"     Wall dist: min {wall_dist[mask].min()*1e6:.1f}, mean {wall_dist[mask].mean()*1e6:.1f} µm")

# Wall-normal profile: sort by R near inner wall, compute layer spacing
near_wall = R_cell < 0.0105  # within ~1 mm of wall
R_sorted = np.sort(R_cell[near_wall])
# Bin into thin radial shells to find layer spacing
r_edges = np.linspace(R_WALL, 0.0105, 100)
r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])
cell_counts_per_shell = np.histogram(R_sorted, bins=r_edges)[0]
dr_shell = (r_edges[1] - r_edges[0]) * 1e6  # µm
print(f"\n6. RADIAL CELL DENSITY (R = 9.5 to 10.5 mm, {dr_shell:.1f} µm shells)")
print(f"   Cells in first 5 shells (wall-adjacent):")
for i in range(min(5, len(cell_counts_per_shell))):
    print(f"     R = {r_centres[i]*1e3:.3f} mm: {cell_counts_per_shell[i]:,} cells")
print(f"   Cells in last 5 shells (mid-gap):")
for i in range(max(0, len(cell_counts_per_shell)-5), len(cell_counts_per_shell)):
    print(f"     R = {r_centres[i]*1e3:.3f} mm: {cell_counts_per_shell[i]:,} cells")

# Estimate number of cells across the film
film_cells = (R_cell >= R_WALL) & (R_cell <= R_WALL + DELTA_MEAN)
# Average number of radial layers: count unique R bins
film_R = R_cell[film_cells]
n_film_layers = len(np.unique(np.round(film_R, decimals=7)))
print(f"\n7. FILM RESOLUTION")
print(f"   Cells within mean film (δ = {DELTA_MEAN*1e3:.2f} mm): {film_cells.sum():,}")
print(f"   Approximate radial layers: {n_film_layers}")
print(f"   Mean cell edge in film: {char_edge[film_cells].mean()*1e6:.1f} µm")

print("=" * 70 + "\n")

del mesh, surf
gc.collect()

# Font size for panel labels
FONT_SIZE = 28

# ---- THREE PANEL ----
plotter = pv.Plotter(off_screen=True, shape=(1, 3), window_size=[3600, 1400])

# (a) 3D perspective with colored boundaries
plotter.subplot(0, 0)
plotter.set_background('white')
plotter.add_mesh(outer_wall, color='lightsteelblue', opacity=0.15,
                 show_edges=True, edge_color=[0.6, 0.6, 0.7], line_width=0.2)
plotter.add_mesh(inner_wall, color='salmon', opacity=0.8,
                 show_edges=True, edge_color=[0.3, 0.15, 0.15], line_width=0.3,
                 label='Inner wall (heated)')
plotter.add_mesh(sym_planes, color='lightyellow', opacity=0.2,
                 show_edges=True, edge_color=[0.5, 0.5, 0.4], line_width=0.15)
plotter.add_mesh(y_slice, scalars='R_mm', cmap='plasma',
                 show_edges=True, edge_color=[0.1, 0.1, 0.15], line_width=0.3,
                 opacity=1.0, show_scalar_bar=False)
plotter.camera_position = [(0.045, 0.06, 0.035), (0.0115, 0.026, 0.005), (0, 1, 0)]
plotter.camera.zoom(1.5)
plotter.add_text("(a) Domain geometry", position='upper_left',
                 font_size=FONT_SIZE, color='black')

# (b) Cross-section face-on — black and white (white faces, black edges)
plotter.subplot(0, 1)
plotter.set_background('white')
plotter.add_mesh(y_slice, color='white',
                 show_edges=True, edge_color='black', line_width=0.5,
                 opacity=1.0)
cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
cz = (pts[:, 2].min() + pts[:, 2].max()) / 2
span = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 2].max() - pts[:, 2].min())
plotter.enable_parallel_projection()
plotter.camera_position = [(cx, 0.126, cz), (cx, 0.026, cz), (-1, 0, 0)]
plotter.camera.parallel_scale = span * 0.6
plotter.add_text("(b) Cross-section (y = 26 mm)", position='upper_left',
                 font_size=FONT_SIZE, color='black')

# (c) Inner wall mesh detail - zoomed
plotter.subplot(0, 2)
plotter.set_background('white')
plotter.add_mesh(inner_wall, color='lightyellow', show_edges=True,
                 edge_color=[0.15, 0.15, 0.2], line_width=0.6, opacity=1.0)
theta_mid = np.radians(22.5)
R_cam = 0.006
cam_x = R_cam * np.cos(theta_mid)
cam_z = R_cam * np.sin(theta_mid)
focus_x = 0.0095 * np.cos(theta_mid)
focus_z = 0.0095 * np.sin(theta_mid)
plotter.enable_parallel_projection()
plotter.camera_position = [
    (cam_x, 0.026, cam_z),
    (focus_x, 0.026, focus_z),
    (0, 1, 0),
]
plotter.camera.parallel_scale = 0.006
plotter.add_text("(c) Inner wall mesh", position='upper_left',
                 font_size=FONT_SIZE, color='black')

# Free memory
gc.collect()

print("Rendering main panels...")
tmp_path = str(OUTPUT_DIR / 'annular_mesh_raw.png')
plotter.screenshot(tmp_path)
plotter.close()
gc.collect()

# ---- Matplotlib overlay: coordinate system on panel (a), labels on panel (b) ----
print("Adding coordinate system overlay...")
img = np.array(Image.open(tmp_path))
h, w = img.shape[:2]
panel_w = w // 3  # three panels

fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
ax.imshow(img)
ax.axis('off')
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Coordinate system in panel (a) — lower-right corner
ox, oy = panel_w * 0.72, h * 0.82
alen = min(panel_w, h) * 0.18
# y-axis (upward)
ax.annotate('', xy=(ox, oy - alen), xytext=(ox, oy),
            arrowprops=dict(arrowstyle='->', color='black', lw=3.5))
ax.text(ox + 6, oy - alen - 12, '$y$', fontsize=30, fontweight='bold',
        ha='center', va='bottom', color='black')
# r-axis (lower-right in perspective — pointing outward from axis)
ax.annotate('', xy=(ox + alen * 0.75, oy + alen * 0.5), xytext=(ox, oy),
            arrowprops=dict(arrowstyle='->', color='black', lw=3.5))
ax.text(ox + alen * 0.75 + 14, oy + alen * 0.5 + 6, '$r$', fontsize=30,
        fontweight='bold', ha='left', va='center', color='black')
# theta-axis (upper-right in perspective)
ax.annotate('', xy=(ox + alen * 0.75, oy - alen * 0.35), xytext=(ox, oy),
            arrowprops=dict(arrowstyle='->', color='black', lw=3.5))
ax.text(ox + alen * 0.75 + 12, oy - alen * 0.35 - 6, r'$\theta$', fontsize=30,
        fontweight='bold', ha='left', va='center', color='black')

# Boundary condition annotations
BC_FONT = 24
bc_bbox = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.8)


# Boundary highlights and labels on panel (b) — cross-section
# Camera mapping: row = 700 + (x - cx)*S, col = panel_w + 600 - (z - cz)*S
# with S = 700 / (span*0.6), +x down, +z left
scale_px = 700.0 / (span * 0.6)
theta_arr = np.linspace(0, np.pi / 4, 200)

# Coloured boundary arcs
COL_IN  = 'crimson'
COL_OUT = 'steelblue'
COL_SYM = 'forestgreen'
BND_LW  = 4

# Inner wall arc (R_inner ≈ 9.5 mm)
R_in = R.min()
bx = R_in * np.cos(theta_arr);  bz = R_in * np.sin(theta_arr)
ax.plot(panel_w + 600 - (bz - cz) * scale_px,
        700 + (bx - cx) * scale_px, color=COL_IN, lw=BND_LW, zorder=5)

# Outer wall arc (R_outer)
R_out = R.max()
bx = R_out * np.cos(theta_arr);  bz = R_out * np.sin(theta_arr)
ax.plot(panel_w + 600 - (bz - cz) * scale_px,
        700 + (bx - cx) * scale_px, color=COL_OUT, lw=BND_LW, zorder=5)

# Symmetry edges (θ = 0° and θ = 45°)
R_edge = np.array([R.min(), R.max()])
for theta_val in [0, np.pi / 4]:
    bx = R_edge * np.cos(theta_val);  bz = R_edge * np.sin(theta_val)
    ax.plot(panel_w + 600 - (bz - cz) * scale_px,
            700 + (bx - cx) * scale_px, color=COL_SYM, lw=BND_LW, zorder=5)

# Labels with matching colours and boundary conditions
ax.annotate(r'Inner wall' '\n' r'$T_w = T_{\mathrm{sat}} + 1\,\mathrm{K}$, no slip',
            xy=(panel_w * 1.66, h * 0.32), xytext=(panel_w * 1.82, h * 0.14),
            fontsize=BC_FONT, ha='center', va='center', color=COL_IN,
            bbox=bc_bbox,
            arrowprops=dict(arrowstyle='->', color=COL_IN, lw=2.5))

ax.annotate('Outer wall\nAdiabatic, no slip',
            xy=(panel_w * 1.46, h * 0.74), xytext=(panel_w * 1.20, h * 0.92),
            fontsize=BC_FONT, ha='center', va='center', color=COL_OUT,
            bbox=bc_bbox,
            arrowprops=dict(arrowstyle='->', color=COL_OUT, lw=2.5))

ax.annotate('Symmetry',
            xy=(panel_w * 1.11, h * 0.35), xytext=(panel_w * 1.12, h * 0.14),
            fontsize=BC_FONT, ha='center', va='center', color=COL_SYM,
            bbox=bc_bbox,
            arrowprops=dict(arrowstyle='->', color=COL_SYM, lw=2.5))

plt.savefig(str(OUTPUT_DIR / 'annular_mesh.png'), dpi=100, bbox_inches='tight',
            pad_inches=0, facecolor='white')
plt.close()
print(f"Saved: {OUTPUT_DIR / 'annular_mesh.png'}")
