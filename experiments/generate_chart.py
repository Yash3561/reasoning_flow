#!/usr/bin/env python3
"""
Generate the "Money Shot" visualization for the presentation.
Clean, professional bar chart showing the 41% curvature drop.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set up professional styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14

# Data from L2-normalized analysis
metrics = ['Euclidean\n(Standard)', 'Poincaré\n(Hyperbolic)']
curvatures = [6.5, 3.9]  # Average curvatures
colors = ['#E74C3C', '#27AE60']  # Red for Euclidean, Green for Poincaré

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

# Create bars
bars = ax.bar(metrics, curvatures, color=colors, width=0.6, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, val in zip(bars, curvatures):
    height = bar.get_height()
    ax.annotate(f'{val:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 8),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=20, fontweight='bold')

# Add the "41% drop" annotation
ax.annotate('', xy=(1, 3.9), xytext=(0, 6.5),
            arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=3))
ax.text(0.5, 5.4, '41% Reduction', ha='center', fontsize=16, 
        fontweight='bold', color='#2C3E50',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F7DC6F', edgecolor='#2C3E50', alpha=0.9))

# Styling
ax.set_ylabel('Menger Curvature (Lower = Straighter Path)', fontsize=14, fontweight='bold')
ax.set_title('Reasoning Curvature: Euclidean vs Hyperbolic\n(L2-Normalized LLM Hidden States)', 
             fontsize=18, fontweight='bold', pad=20)
ax.set_ylim(0, 8)

# Add interpretation text at bottom
ax.text(0.5, -0.15, 
        'Lower curvature = Reasoning paths are straighter (more geodesic-like)',
        transform=ax.transAxes, ha='center', fontsize=12, style='italic', color='#555')

# Clean up
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

# Save to results folder
output_path = 'results/curvature_comparison_chart.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'✅ Chart saved to: {output_path}')

# Also save as PNG for easy viewing
png_path = 'results/curvature_comparison_chart.png'
plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'✅ PNG saved to: {png_path}')

plt.close()
