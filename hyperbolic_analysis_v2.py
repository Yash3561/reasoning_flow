#!/usr/bin/env python3
"""
Hyperbolic vs Euclidean Curvature Analysis (v2 - Poincaré Ball)

This script tests the hypothesis that LLM reasoning paths are geodesics
(straight lines) in hyperbolic space, not complex curves in Euclidean space.

VERSION 2: Uses Poincaré Ball model instead of Lorentz Hyperboloid.
The Poincaré Ball better preserves local structure when projecting from 
high-dimensional Euclidean space.

Key insight: If the Poincaré curvature is lower than Euclidean curvature,
it suggests the manifold is intrinsically hyperbolic.

Usage:
    python hyperbolic_analysis_v2.py --data_dir results/exp1_order0/data
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path


# ==============================================================================
# Poincaré Ball Model Operations
# ==============================================================================

def project_to_poincare(x: np.ndarray, c: float = 1.0, eps: float = 1e-5) -> np.ndarray:
    """
    Project Euclidean vectors to the Poincaré Ball.
    
    The Poincaré Ball is the open unit ball with a hyperbolic metric.
    We use the exponential map at the origin to project points:
        exp_0(v) = tanh(c^0.5 * ||v|| / 2) * (v / ||v||)
    
    This preserves directional information while mapping to the ball interior.
    
    Args:
        x: (N, D) array of Euclidean vectors
        c: Curvature parameter (default 1.0 for standard hyperbolic)
        eps: Small value for numerical stability
    
    Returns:
        (N, D) array of Poincaré Ball vectors (inside unit ball)
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    # Normalize vectors to have manageable norms
    # (LLM hidden states can have very large norms)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    
    # Scale to reasonable range for tanh
    max_norm = np.max(norms)
    if max_norm > 100:
        x = x * (10.0 / max_norm)
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
    
    # Exponential map at origin: exp_0(v) = tanh(sqrt(c) * ||v|| / 2) * v/||v||
    sqrt_c = np.sqrt(c)
    norms_safe = np.maximum(norms, eps)
    
    # The coefficient: tanh(sqrt(c) * ||v|| / 2) / ||v||
    coeff = np.tanh(sqrt_c * norms_safe / 2) / norms_safe
    
    return x * coeff


def poincare_distance(x: np.ndarray, y: np.ndarray, c: float = 1.0, eps: float = 1e-5) -> float:
    """
    Geodesic distance in the Poincaré Ball.
    
    d(x, y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||−x ⊕ y||)
    
    where ⊕ is the Möbius addition.
    
    Simplified formula:
    d(x,y) = (2/sqrt(c)) * arctanh(sqrt(c) * (2||x-y||^2) / ((1-||x||^2)(1-||y||^2) + ||x-y||^2))^0.5
    
    Args:
        x, y: (1, D) arrays of Poincaré Ball vectors
        c: Curvature parameter
        eps: Numerical stability
    
    Returns:
        Scalar distance
    """
    x = x.flatten()
    y = y.flatten()
    
    sqrt_c = np.sqrt(c)
    
    # Norms
    x_norm_sq = np.sum(x ** 2)
    y_norm_sq = np.sum(y ** 2)
    diff_norm_sq = np.sum((x - y) ** 2)
    
    # Clamp to stay inside ball
    x_norm_sq = min(x_norm_sq, 1 - eps)
    y_norm_sq = min(y_norm_sq, 1 - eps)
    
    # Numerator and denominator
    num = 2 * diff_norm_sq
    denom = (1 - x_norm_sq) * (1 - y_norm_sq)
    
    # Compute distance
    arg = sqrt_c * np.sqrt(num / (denom + eps))
    arg = min(arg, 1 - eps)  # arctanh domain is (-1, 1)
    
    dist = (2 / sqrt_c) * np.arctanh(arg)
    return float(dist)


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Standard Euclidean distance."""
    return float(np.linalg.norm(x.flatten() - y.flatten()))


# ==============================================================================
# Curvature Calculations
# ==============================================================================

def menger_curvature(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, 
                     metric: str = 'euclidean', c: float = 1.0) -> float:
    """
    Calculate Menger Curvature for three points.
    
    Menger Curvature = 4 * Area(triangle) / (a * b * c)
    
    - If points are collinear (geodesic), Area -> 0, Curvature -> 0.
    - Sharp turns = high curvature.
    
    Args:
        x1, x2, x3: Points (Euclidean or Poincaré)
        metric: 'euclidean' or 'poincare'
        c: Curvature parameter for Poincaré
    
    Returns:
        Curvature value
    """
    if metric == 'euclidean':
        a = euclidean_distance(x1, x2)
        b = euclidean_distance(x2, x3)
        c_side = euclidean_distance(x3, x1)
    elif metric == 'poincare':
        a = poincare_distance(x1, x2, c=c)
        b = poincare_distance(x2, x3, c=c)
        c_side = poincare_distance(x3, x1, c=c)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Semi-perimeter
    s = (a + b + c_side) / 2
    
    # Heron's formula
    area_sq = s * (s - a) * (s - b) * (s - c_side)
    area = np.sqrt(max(0, area_sq))
    
    # Menger curvature
    denom = a * b * c_side
    if denom < 1e-12:
        return 0.0
    
    return float((4 * area) / denom)


def analyze_trajectory(embeddings: np.ndarray, c: float = 1.0) -> dict:
    """
    Analyze a reasoning trajectory comparing Euclidean vs Poincaré curvature.
    
    Args:
        embeddings: (T, D) array of reasoning step embeddings
        c: Curvature parameter
    
    Returns:
        Dictionary with curvature comparisons
    """
    T = embeddings.shape[0]
    if T < 3:
        return {"error": "Need at least 3 steps"}
    
    # Project to Poincaré Ball
    poincare_emb = project_to_poincare(embeddings, c=c)
    
    euclidean_curvatures = []
    poincare_curvatures = []
    
    for i in range(1, T - 1):
        x1 = embeddings[i - 1:i]
        x2 = embeddings[i:i + 1]
        x3 = embeddings[i + 1:i + 2]
        
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
        "euclidean_curvatures": euclidean_curvatures,
        "poincare_curvatures": poincare_curvatures,
        "mean_euclidean": mean_euc,
        "mean_poincare": mean_poin,
        "ratio": mean_poin / (mean_euc + 1e-9) if mean_euc > 0 else 0
    }


# ==============================================================================
# Main
# ==============================================================================

def load_embeddings_from_dir(data_dir: str) -> dict:
    """Load embeddings from experiment data directory."""
    emb_dir = os.path.join(data_dir, "embeddings")
    if not os.path.exists(emb_dir):
        raise FileNotFoundError(f"Embeddings not found: {emb_dir}")
    
    embeddings = {}
    for npy_file in Path(emb_dir).glob("*.npy"):
        embeddings[npy_file.stem] = np.load(npy_file)
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Hyperbolic Curvature Analysis (Poincaré Ball)")
    parser.add_argument("--data_dir", type=str, default="results/exp1_order0/data")
    parser.add_argument("--output", type=str, default="curvature_poincare.csv")
    parser.add_argument("--curvature", type=float, default=1.0, help="Hyperbolic curvature parameter")
    args = parser.parse_args()
    
    print(f"Loading embeddings from: {args.data_dir}")
    embeddings = load_embeddings_from_dir(args.data_dir)
    print(f"Loaded {len(embeddings)} trajectories\n")
    
    results = []
    
    for label, emb in embeddings.items():
        analysis = analyze_trajectory(emb, c=args.curvature)
        
        if "error" in analysis:
            continue
        
        results.append({
            "Label": label,
            "Steps": emb.shape[0],
            "Euclidean_Curvature": round(analysis["mean_euclidean"], 6),
            "Poincare_Curvature": round(analysis["mean_poincare"], 6),
            "Ratio_P/E": round(analysis["ratio"], 3),
        })
        
        print(f"{label}: Euc={analysis['mean_euclidean']:.4f}, Poincaré={analysis['mean_poincare']:.4f}, Ratio={analysis['ratio']:.3f}")
    
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: EUCLIDEAN vs POINCARÉ CURVATURE")
    print("=" * 70)
    
    avg_euc = df["Euclidean_Curvature"].mean()
    avg_poin = df["Poincare_Curvature"].mean()
    avg_ratio = df["Ratio_P/E"].mean()
    
    print(f"Average Euclidean Curvature:  {avg_euc:.6f}")
    print(f"Average Poincaré Curvature:   {avg_poin:.6f}")
    print(f"Average Ratio (Poincaré/Euc): {avg_ratio:.3f}")
    print("=" * 70)
    
    if avg_ratio < 1.0:
        print("\n🎯 HYPOTHESIS SUPPORTED!")
        print("   Poincaré curvature < Euclidean curvature")
        print("   Reasoning paths are STRAIGHTER in hyperbolic space.")
    elif avg_ratio > 1.0:
        print("\n⚠️  INTERESTING FINDING!")
        print("   Poincaré curvature > Euclidean curvature")
        print("   This suggests the manifold may be SPHERICAL, not hyperbolic.")
        print("   Consider: Spherical geometry for reasoning?")
    else:
        print("\n📊 Inconclusive - curvatures are similar.")
    
    print(f"\n✅ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
