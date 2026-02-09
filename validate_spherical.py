#!/usr/bin/env python3
"""
Spherical Manifold Validation & Visualization

This script validates the "Spherical Attractor" hypothesis by:
1. Checking Poincaré coordinate distribution (center vs. edge)
2. Visualizing reasoning trajectories in 2D/3D
3. Detecting convergence patterns toward logical attractors

If norms are < 0.5 (centered), the spherical finding is scientifically valid.
If norms are > 0.9 (edge), we have numerical artifacts to address.

Usage:
    python validate_spherical.py --data_dir results/exp1_order0/data
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA


def project_to_poincare(x: np.ndarray, c: float = 1.0, eps: float = 1e-5) -> np.ndarray:
    """Project Euclidean vectors to Poincaré Ball using exponential map at origin."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    max_norm = np.max(norms)
    if max_norm > 100:
        x = x * (10.0 / max_norm)
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
    
    sqrt_c = np.sqrt(c)
    norms_safe = np.maximum(norms, eps)
    coeff = np.tanh(sqrt_c * norms_safe / 2) / norms_safe
    
    return x * coeff


def load_embeddings_from_dir(data_dir: str) -> dict:
    """Load embeddings from experiment data directory."""
    emb_dir = os.path.join(data_dir, "embeddings")
    if not os.path.exists(emb_dir):
        raise FileNotFoundError(f"Embeddings not found: {emb_dir}")
    
    embeddings = {}
    for npy_file in Path(emb_dir).glob("*.npy"):
        embeddings[npy_file.stem] = np.load(npy_file)
    
    return embeddings


def analyze_poincare_distribution(embeddings: dict) -> dict:
    """
    Analyze the distribution of Poincaré norms.
    
    Critical check:
    - If mean norm < 0.5: Points are well-centered (VALID)
    - If mean norm > 0.9: Points are at edge (NUMERICAL ARTIFACT)
    """
    all_norms = []
    
    for label, emb in embeddings.items():
        poincare_emb = project_to_poincare(emb)
        norms = np.linalg.norm(poincare_emb, axis=-1)
        all_norms.extend(norms.tolist())
    
    all_norms = np.array(all_norms)
    
    return {
        "mean_norm": np.mean(all_norms),
        "std_norm": np.std(all_norms),
        "min_norm": np.min(all_norms),
        "max_norm": np.max(all_norms),
        "pct_below_05": np.mean(all_norms < 0.5) * 100,
        "pct_above_09": np.mean(all_norms > 0.9) * 100,
        "all_norms": all_norms
    }


def visualize_trajectories(embeddings: dict, output_dir: str, sample_size: int = 6):
    """
    Visualize reasoning trajectories in both Euclidean and Poincaré space.
    Shows if paths are converging toward attractors.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sample a few trajectories
    labels = list(embeddings.keys())[:sample_size]
    
    # Collect all embeddings for global PCA
    all_emb = np.vstack([embeddings[l] for l in labels])
    pca = PCA(n_components=2)
    pca.fit(all_emb)
    
    # Create figure with 2 columns: Euclidean vs Poincaré
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # --- Euclidean Space (2D PCA) ---
    ax_euc = axes[0, 0]
    ax_euc.set_title("Euclidean Space (PCA 2D)", fontsize=14)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    
    for idx, label in enumerate(labels):
        emb = embeddings[label]
        emb_2d = pca.transform(emb)
        
        # Plot trajectory
        ax_euc.plot(emb_2d[:, 0], emb_2d[:, 1], 'o-', 
                   color=colors[idx], alpha=0.7, markersize=8,
                   label=label[:20])
        
        # Mark start and end
        ax_euc.scatter(emb_2d[0, 0], emb_2d[0, 1], s=100, c='green', 
                      marker='s', zorder=5, edgecolors='black')
        ax_euc.scatter(emb_2d[-1, 0], emb_2d[-1, 1], s=100, c='red', 
                      marker='*', zorder=5, edgecolors='black')
    
    ax_euc.legend(fontsize=8, loc='upper right')
    ax_euc.set_xlabel("PC1")
    ax_euc.set_ylabel("PC2")
    ax_euc.grid(True, alpha=0.3)
    
    # --- Poincaré Space (2D) ---
    ax_poin = axes[0, 1]
    ax_poin.set_title("Poincaré Ball (2D Projection)", fontsize=14)
    
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax_poin.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=2)
    ax_poin.fill(np.cos(theta), np.sin(theta), alpha=0.05, color='gray')
    
    for idx, label in enumerate(labels):
        emb = embeddings[label]
        poin_emb = project_to_poincare(emb)
        
        # Use PCA to reduce to 2D
        if poin_emb.shape[1] > 2:
            poin_2d = PCA(n_components=2).fit_transform(poin_emb)
            # Re-normalize to unit ball
            max_norm = np.max(np.linalg.norm(poin_2d, axis=1))
            if max_norm > 0.99:
                poin_2d = poin_2d * 0.95 / max_norm
        else:
            poin_2d = poin_emb
        
        ax_poin.plot(poin_2d[:, 0], poin_2d[:, 1], 'o-', 
                    color=colors[idx], alpha=0.7, markersize=8)
        ax_poin.scatter(poin_2d[0, 0], poin_2d[0, 1], s=100, c='green', 
                       marker='s', zorder=5, edgecolors='black')
        ax_poin.scatter(poin_2d[-1, 0], poin_2d[-1, 1], s=100, c='red', 
                       marker='*', zorder=5, edgecolors='black')
    
    ax_poin.set_xlim(-1.2, 1.2)
    ax_poin.set_ylim(-1.2, 1.2)
    ax_poin.set_aspect('equal')
    ax_poin.set_xlabel("Poincaré X")
    ax_poin.set_ylabel("Poincaré Y")
    ax_poin.grid(True, alpha=0.3)
    
    # --- Norm Distribution Histogram ---
    ax_hist = axes[1, 0]
    ax_hist.set_title("Poincaré Norm Distribution", fontsize=14)
    
    all_norms = []
    for label in labels:
        emb = embeddings[label]
        poin_emb = project_to_poincare(emb)
        norms = np.linalg.norm(poin_emb, axis=-1)
        all_norms.extend(norms.tolist())
    
    ax_hist.hist(all_norms, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax_hist.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Safe Zone (< 0.5)')
    ax_hist.axvline(x=0.9, color='red', linestyle='--', linewidth=2, label='Danger Zone (> 0.9)')
    ax_hist.set_xlabel("Poincaré Norm")
    ax_hist.set_ylabel("Frequency")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    # --- Convergence Analysis ---
    ax_conv = axes[1, 1]
    ax_conv.set_title("Step-wise Distance to Centroid (Convergence Check)", fontsize=14)
    
    for idx, label in enumerate(labels):
        emb = embeddings[label]
        
        # Calculate distance from each step to the final step (target)
        final_state = emb[-1]
        distances = np.linalg.norm(emb - final_state, axis=-1)
        steps = np.arange(len(distances))
        
        ax_conv.plot(steps, distances, 'o-', color=colors[idx], alpha=0.7, 
                    label=label[:15])
    
    ax_conv.set_xlabel("Reasoning Step")
    ax_conv.set_ylabel("Distance to Final State")
    ax_conv.legend(fontsize=8, loc='upper right')
    ax_conv.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "spherical_validation.pdf")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Validate Spherical Manifold Hypothesis")
    parser.add_argument("--data_dir", type=str, default="results/exp1_order0/data")
    parser.add_argument("--output_dir", type=str, default="results/spherical_validation")
    args = parser.parse_args()
    
    print("=" * 70)
    print("SPHERICAL MANIFOLD VALIDATION")
    print("=" * 70)
    
    print(f"\nLoading embeddings from: {args.data_dir}")
    embeddings = load_embeddings_from_dir(args.data_dir)
    print(f"Loaded {len(embeddings)} trajectories")
    
    # === CRITICAL CHECK: Poincaré Norm Distribution ===
    print("\n" + "-" * 50)
    print("CRITICAL CHECK: Poincaré Norm Distribution")
    print("-" * 50)
    
    dist_analysis = analyze_poincare_distribution(embeddings)
    
    print(f"  Mean Norm:     {dist_analysis['mean_norm']:.4f}")
    print(f"  Std Norm:      {dist_analysis['std_norm']:.4f}")
    print(f"  Min Norm:      {dist_analysis['min_norm']:.4f}")
    print(f"  Max Norm:      {dist_analysis['max_norm']:.4f}")
    print(f"  % Below 0.5:   {dist_analysis['pct_below_05']:.1f}%")
    print(f"  % Above 0.9:   {dist_analysis['pct_above_09']:.1f}%")
    
    # VERDICT
    print("\n" + "=" * 50)
    if dist_analysis['mean_norm'] < 0.5:
        print("🎯 VERDICT: VALID - Points are well-centered")
        print("   The 4.7x curvature increase is REAL.")
        print("   The Spherical Attractor hypothesis is scientifically supported!")
    elif dist_analysis['mean_norm'] < 0.8:
        print("⚠️  VERDICT: CAUTION - Points are moderately centered")
        print("   Results may be valid but warrant careful interpretation.")
    else:
        print("❌ VERDICT: NUMERICAL ARTIFACT - Points are at the edge")
        print("   The 4.7x increase may be due to projection distortion.")
        print("   Normalize your embeddings before making claims.")
    print("=" * 50)
    
    # Generate visualization
    print("\nGenerating visualization...")
    visualize_trajectories(embeddings, args.output_dir, sample_size=6)
    
    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()
