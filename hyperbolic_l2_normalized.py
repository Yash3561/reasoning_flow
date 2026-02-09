#!/usr/bin/env python3
"""
Hyperbolic Curvature Analysis - L2 Normalized Version

This is the corrected version that L2-normalizes embeddings BEFORE
projecting to Poincaré Ball to avoid boundary collapse.

The fix: Normalize each hidden state to unit norm (||x|| = 1)
before applying the exponential map. This ensures points stay
well inside the Poincaré ball, giving numerically stable results.

Usage:
    python hyperbolic_l2_normalized.py --data_dir results/exp1_order0/data
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    L2-normalize each vector to unit norm.
    
    This is THE critical fix. By making ||x|| = 1 for all hidden states,
    we ensure the Poincaré projection doesn't collapse to the boundary.
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + eps)


def project_to_poincare(x: np.ndarray, c: float = 1.0, eps: float = 1e-5) -> np.ndarray:
    """
    Project L2-normalized vectors to Poincaré Ball.
    
    Since inputs are now unit-normalized, tanh(sqrt(c) * 0.5) ≈ 0.46,
    placing points safely in the CENTER of the ball.
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    sqrt_c = np.sqrt(c)
    norms_safe = np.maximum(norms, eps)
    
    # exp_0(v) = tanh(sqrt(c) * ||v|| / 2) * (v / ||v||)
    coeff = np.tanh(sqrt_c * norms_safe / 2) / norms_safe
    return x * coeff


def poincare_distance(x: np.ndarray, y: np.ndarray, c: float = 1.0, eps: float = 1e-5) -> float:
    """Geodesic distance in Poincaré Ball."""
    x = x.flatten()
    y = y.flatten()
    
    sqrt_c = np.sqrt(c)
    
    x_norm_sq = min(np.sum(x ** 2), 1 - eps)
    y_norm_sq = min(np.sum(y ** 2), 1 - eps)
    diff_norm_sq = np.sum((x - y) ** 2)
    
    num = 2 * diff_norm_sq
    denom = (1 - x_norm_sq) * (1 - y_norm_sq)
    
    arg = sqrt_c * np.sqrt(num / (denom + eps))
    arg = min(arg, 1 - eps)
    
    return float((2 / sqrt_c) * np.arctanh(arg))


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x.flatten() - y.flatten()))


def menger_curvature(x1, x2, x3, metric='euclidean', c=1.0) -> float:
    """
    Menger Curvature = 4 * Area / (a * b * c)
    
    Lower curvature = straighter path (geodesic)
    """
    if metric == 'euclidean':
        a = euclidean_distance(x1, x2)
        b = euclidean_distance(x2, x3)
        c_side = euclidean_distance(x3, x1)
    else:
        a = poincare_distance(x1, x2, c=c)
        b = poincare_distance(x2, x3, c=c)
        c_side = poincare_distance(x3, x1, c=c)
    
    s = (a + b + c_side) / 2
    area_sq = s * (s - a) * (s - b) * (s - c_side)
    area = np.sqrt(max(0, area_sq))
    
    denom = a * b * c_side
    if denom < 1e-12:
        return 0.0
    
    return float((4 * area) / denom)


def analyze_trajectory(embeddings: np.ndarray, c: float = 1.0) -> dict:
    """Analyze trajectory with L2-normalized embeddings."""
    T = embeddings.shape[0]
    if T < 3:
        return {"error": "Need at least 3 steps"}
    
    # THE FIX: L2-normalize before projection
    normalized_emb = l2_normalize(embeddings)
    poincare_emb = project_to_poincare(normalized_emb, c=c)
    
    # Check Poincaré norms (should now be ~0.46, not ~0.999)
    poin_norms = np.linalg.norm(poincare_emb, axis=-1)
    
    euclidean_curvatures = []
    poincare_curvatures = []
    
    for i in range(1, T - 1):
        # Use L2-normalized for Euclidean too (fair comparison)
        x1 = normalized_emb[i - 1:i]
        x2 = normalized_emb[i:i + 1]
        x3 = normalized_emb[i + 1:i + 2]
        
        x1_P = poincare_emb[i - 1:i]
        x2_P = poincare_emb[i:i + 1]
        x3_P = poincare_emb[i + 1:i + 2]
        
        euc_curv = menger_curvature(x1, x2, x3, metric='euclidean')
        poin_curv = menger_curvature(x1_P, x2_P, x3_P, metric='poincare', c=c)
        
        euclidean_curvatures.append(euc_curv)
        poincare_curvatures.append(poin_curv)
    
    mean_euc = np.mean(euclidean_curvatures) if euclidean_curvatures else 0
    mean_poin = np.mean(poincare_curvatures) if poincare_curvatures else 0
    
    return {
        "mean_euclidean": mean_euc,
        "mean_poincare": mean_poin,
        "ratio": mean_poin / (mean_euc + 1e-9) if mean_euc > 0 else 0,
        "mean_poincare_norm": np.mean(poin_norms),
        "max_poincare_norm": np.max(poin_norms)
    }


def load_embeddings_from_dir(data_dir: str) -> dict:
    emb_dir = os.path.join(data_dir, "embeddings")
    if not os.path.exists(emb_dir):
        raise FileNotFoundError(f"Embeddings not found: {emb_dir}")
    
    embeddings = {}
    for npy_file in Path(emb_dir).glob("*.npy"):
        embeddings[npy_file.stem] = np.load(npy_file)
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Hyperbolic Analysis with L2 Normalization")
    parser.add_argument("--data_dir", type=str, default="results/exp1_order0/data")
    parser.add_argument("--output", type=str, default="curvature_l2_normalized.csv")
    args = parser.parse_args()
    
    print("=" * 70)
    print("HYPERBOLIC CURVATURE ANALYSIS (L2-NORMALIZED)")
    print("=" * 70)
    
    print(f"\nLoading embeddings from: {args.data_dir}")
    embeddings = load_embeddings_from_dir(args.data_dir)
    print(f"Loaded {len(embeddings)} trajectories\n")
    
    results = []
    all_poin_norms = []
    
    for label, emb in embeddings.items():
        analysis = analyze_trajectory(emb)
        
        if "error" in analysis:
            continue
        
        results.append({
            "Label": label,
            "Euclidean": round(analysis["mean_euclidean"], 6),
            "Poincare": round(analysis["mean_poincare"], 6),
            "Ratio_P/E": round(analysis["ratio"], 3),
            "Poincare_Norm": round(analysis["mean_poincare_norm"], 4),
        })
        
        all_poin_norms.append(analysis["mean_poincare_norm"])
    
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    
    # Summary
    print("=" * 70)
    print("RESULTS WITH L2 NORMALIZATION")
    print("=" * 70)
    
    avg_euc = df["Euclidean"].mean()
    avg_poin = df["Poincare"].mean()
    avg_ratio = df["Ratio_P/E"].mean()
    avg_norm = np.mean(all_poin_norms)
    
    print(f"\n📊 POINCARÉ NORM CHECK:")
    print(f"   Mean Norm: {avg_norm:.4f}")
    if avg_norm < 0.5:
        print("   ✅ Points are CENTERED (numerically stable)")
    else:
        print("   ⚠️  Points still at edge")
    
    print(f"\n📐 CURVATURE COMPARISON:")
    print(f"   Euclidean Curvature:  {avg_euc:.6f}")
    print(f"   Poincaré Curvature:   {avg_poin:.6f}")
    print(f"   Ratio (Poincaré/Euc): {avg_ratio:.3f}")
    
    print("\n" + "=" * 70)
    if avg_ratio < 1.0:
        print("🎯 HYPOTHESIS SUPPORTED!")
        print("   Poincaré curvature < Euclidean curvature")
        print("   Reasoning paths are STRAIGHTER in hyperbolic space.")
        print("   This supports the 'Hyperbolic Geodesic' hypothesis!")
    elif avg_ratio > 1.0 and avg_ratio < 2.0:
        print("📊 INCONCLUSIVE")
        print("   Curvatures are similar in both spaces.")
        print("   Neither hyperbolic nor spherical geometry dominates.")
    else:
        print("🌐 SPHERICAL GEOMETRY INDICATED")
        print(f"   Poincaré curvature is {avg_ratio:.1f}x higher.")
        print("   Reasoning may follow spherical attractors (convergent flow).")
    print("=" * 70)
    
    print(f"\n✅ Results saved to: {args.output}")
    
    # Show top 5 results
    print("\nSample Results (first 5):")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
