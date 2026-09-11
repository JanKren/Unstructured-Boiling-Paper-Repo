#!/usr/bin/env python3
"""Standalone script to regenerate only the annular schematic (Figure 16)."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Paths — override via environment variables, e.g.:
#   export OUTPUT_DIR=/path/to/paper/pics
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR",
                                 os.path.dirname(os.path.abspath(__file__))))
R_INNER_DISPLAY = 0.0095  # 9.50 mm

fig, ax = plt.subplots(figsize=(10, 5))

# Wall
ax.axhline(R_INNER_DISPLAY * 1000, color='gray', lw=8, label='Heated wall')
ax.fill_between([0, 52], 0, R_INNER_DISPLAY * 1000, color='lightgray', alpha=0.5)

# Wavy interface
y_wave = np.linspace(0, 52, 500)
wave = 0.12 * np.sin(2*np.pi*y_wave/26) + 0.05 * np.sin(2*np.pi*y_wave/13 + 0.5)
r_interface_schematic = R_INNER_DISPLAY * 1000 + 0.12 + wave

# Liquid film
ax.fill_between(y_wave, R_INNER_DISPLAY * 1000, r_interface_schematic, color='steelblue', alpha=0.7, label='Liquid film')
ax.plot(y_wave, r_interface_schematic, 'b-', lw=2)

# Gas core
ax.fill_between(y_wave, r_interface_schematic, 11, color='lightyellow', alpha=0.5, label='Vapour core')

# Velocity arrows - gas
for y_pos in [8, 22, 36]:
    ax.annotate('', xy=(y_pos+6, 10.6), xytext=(y_pos, 10.6),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
ax.text(30, 10.85, '$U_v \\approx 18.5$ m/s', fontsize=15, ha='center', color='darkred')

# Velocity arrows - liquid
for y_pos in [8, 22, 36]:
    idx = np.argmin(np.abs(y_wave - y_pos))
    r_liq = (R_INNER_DISPLAY * 1000 + r_interface_schematic[idx]) / 2
    ax.annotate('', xy=(y_pos+2.5, r_liq), xytext=(y_pos, r_liq),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
ax.text(45, 9.58, '$U_l \\approx 0.6$ m/s', fontsize=13, ha='center', color='darkblue')

# Wave propagation
ax.annotate('', xy=(42, 9.75), xytext=(28, 9.75),
            arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
ax.text(35, 9.65, '$c \\approx 1.6$ m/s', fontsize=15, ha='center', color='darkgreen',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Heat flux arrows
for y_pos in [5, 15, 25, 35, 45]:
    ax.annotate('', xy=(y_pos, R_INNER_DISPLAY*1000 + 0.05), xytext=(y_pos, R_INNER_DISPLAY*1000 - 0.15),
                arrowprops=dict(arrowstyle='->', color='orange', lw=1.5, alpha=0.7))
ax.text(2, R_INNER_DISPLAY*1000 - 0.25, '$q_{wall}$', fontsize=14, color='darkorange')

# Evaporation at trough
trough_y = 39
ax.annotate('', xy=(trough_y, 9.7), xytext=(trough_y, 9.55),
            arrowprops=dict(arrowstyle='->', color='purple', lw=2))
ax.annotate('', xy=(trough_y-1.5, 9.68), xytext=(trough_y-1.5, 9.55),
            arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))
ax.annotate('', xy=(trough_y+1.5, 9.68), xytext=(trough_y+1.5, 9.55),
            arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))
ax.text(trough_y, 9.85, 'High\nevaporation', fontsize=12, ha='center', color='purple')

# Annotations
ax.annotate('Thick film\n(low MTR)', xy=(22, 9.85), xytext=(12, 10.15),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=13, ha='center')
ax.annotate('Thin film\n(high MTR)', xy=(39, 9.57), xytext=(47, 9.35),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=13, ha='center')

ax.set_xlim([-1, 53])
ax.set_ylim([9.1, 11.2])
ax.set_xlabel('Axial position $y$ [mm]', fontsize=15)
ax.set_ylabel('Radial position $r$ [mm]', fontsize=15)
ax.set_title('Wave-modulated evaporation in upward annular flow', fontsize=16, fontweight='bold')
ax.legend(loc='upper left', fontsize=13)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'annular_schematic.pdf')
plt.savefig(OUTPUT_DIR / 'annular_schematic.png')
plt.close()
print(f"Saved: {OUTPUT_DIR / 'annular_schematic.pdf'}")
