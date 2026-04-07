#!/usr/bin/env python3
"""Generate Menger Curvature diagram for presentation."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set up figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ============= Left Panel: High Curvature (Sharp Turn) =============
ax1 = axes[0]
ax1.set_title("High Curvature = Sharp Turn", fontsize=16, fontweight='bold', color='#E74C3C')

# Three points forming a sharp angle
p1 = np.array([0.2, 0.3])
p2 = np.array([0.5, 0.7])
p3 = np.array([0.8, 0.35])

# Plot points
for i, (p, label) in enumerate([(p1, r'$y_{t-1}$'), (p2, r'$y_t$'), (p3, r'$y_{t+1}$')]):
    ax1.scatter(*p, s=200, c='#3498DB', zorder=5, edgecolors='black', linewidth=2)
    ax1.annotate(label, (p[0]+0.03, p[1]+0.05), fontsize=14, fontweight='bold')

# Plot trajectory lines
ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=2, zorder=3)
ax1.plot([p2[0], p3[0]], [p2[1], p3[1]], 'k-', lw=2, zorder=3)
ax1.annotate('', xy=(p2[0]-0.02, p2[1]+0.02), xytext=(p1[0]+0.02, p1[1]+0.02),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax1.annotate('', xy=(p3[0]-0.02, p3[1]+0.02), xytext=(p2[0]+0.02, p2[1]-0.02),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Calculate circumcircle (small radius = high curvature)
# Using circumcircle formula
def circumcircle(p1, p2, p3):
    ax = p1[0]; ay = p1[1]
    bx = p2[0]; by = p2[1]
    cx = p3[0]; cy = p3[1]
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-10:
        return None, None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    center = np.array([ux, uy])
    radius = np.linalg.norm(p1 - center)
    return center, radius

center1, r1 = circumcircle(p1, p2, p3)
circle1 = plt.Circle(center1, r1, fill=False, color='#E74C3C', linestyle='--', lw=2, zorder=2)
ax1.add_patch(circle1)
ax1.scatter(*center1, s=50, c='#E74C3C', marker='x', zorder=4)

# Annotation
ax1.text(0.5, 0.05, f'Small radius (r = {r1:.2f})\n→ High Curvature (κ = 1/r)', 
        ha='center', fontsize=12, style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FADBD8', edgecolor='#E74C3C'))

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_aspect('equal')
ax1.axis('off')

# ============= Right Panel: Low Curvature (Straight Line) =============
ax2 = axes[1]
ax2.set_title("Low Curvature = Straight Path", fontsize=16, fontweight='bold', color='#27AE60')

# Three points nearly collinear
q1 = np.array([0.15, 0.4])
q2 = np.array([0.5, 0.5])
q3 = np.array([0.85, 0.6])

# Plot points
for i, (p, label) in enumerate([(q1, r'$y_{t-1}$'), (q2, r'$y_t$'), (q3, r'$y_{t+1}$')]):
    ax2.scatter(*p, s=200, c='#3498DB', zorder=5, edgecolors='black', linewidth=2)
    ax2.annotate(label, (p[0]+0.03, p[1]+0.05), fontsize=14, fontweight='bold')

# Plot trajectory lines
ax2.plot([q1[0], q2[0]], [q1[1], q2[1]], 'k-', lw=2, zorder=3)
ax2.plot([q2[0], q3[0]], [q2[1], q3[1]], 'k-', lw=2, zorder=3)
ax2.annotate('', xy=(q2[0]-0.02, q2[1]+0.01), xytext=(q1[0]+0.02, q1[1]+0.01),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax2.annotate('', xy=(q3[0]-0.02, q3[1]+0.01), xytext=(q2[0]+0.02, q2[1]+0.01),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Large circumcircle (large radius = low curvature)
center2, r2 = circumcircle(q1, q2, q3)
# Only show partial arc since circle is huge
theta = np.linspace(np.arctan2(q1[1]-center2[1], q1[0]-center2[0]),
                    np.arctan2(q3[1]-center2[1], q3[0]-center2[0]), 50)
arc_x = center2[0] + r2 * np.cos(theta)
arc_y = center2[1] + r2 * np.sin(theta)
ax2.plot(arc_x, arc_y, '--', color='#27AE60', lw=2, zorder=2)

# Annotation
ax2.text(0.5, 0.15, f'Large radius (r = {r2:.2f})\n→ Low Curvature (κ ≈ 0)', 
        ha='center', fontsize=12, style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#D5F5E3', edgecolor='#27AE60'))

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_aspect('equal')
ax2.axis('off')

# Overall title
fig.suptitle("Menger Curvature: Measuring the 'Turn' in a Thought", 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save
output_path = 'results/menger_curvature_diagram.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'✅ Saved to: {output_path}')
plt.close()
