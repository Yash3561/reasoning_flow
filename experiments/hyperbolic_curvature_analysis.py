#!/usr/bin/env python3
"""
Hyperbolic vs Euclidean Curvature Analysis

This script tests the hypothesis that LLM reasoning paths are geodesics
(straight lines) in hyperbolic space, not complex curves in Euclidean space.

The key insight: If the Lorentzian (Hyperbolic) curvature is significantly 
lower than the Euclidean curvature, it proves that the LLM is following 
straight-line geodesics in a curved space.

Usage:
    python hyperbolic_curvature_analysis.py --data_dir results/exp1_order0/data
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path


# ==============================================================================
# Lorentz Manifold Operations
# ==============================================================================

def project_to_lorentz(x: np.ndarray) -> np.ndarray:
    """
    Project a Euclidean vector onto the Lorentz Hyperboloid.
    
    The Lorentz model places points on the upper sheet of the hyperboloid:
        -x_0^2 + x_1^2 + ... + x_n^2 = -1
    
    Given a spatial vector x_spatial, we compute:
        x_0 = sqrt(1 + ||x_spatial||^2)
    
    Args:
        x: (N, D) array of Euclidean vectors
    
    Returns:
        (N, D+1) array of Lorentzian vectors with x_0 as the first coordinate
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    # Normalize to prevent numerical issues (scale to unit ball first)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    max_norm = np.max(norms)
    if max_norm > 1e-6:
        x_scaled = x / (max_norm + 1e-9)  # Scale to ~unit ball
    else:
        x_scaled = x
    
    # Compute time-like coordinate
    spatial_norm_sq = np.sum(x_scaled ** 2, axis=-1, keepdims=True)
    x_0 = np.sqrt(1 + spatial_norm_sq)
    
    # Combine: [x_0, x_1, ..., x_n]
    return np.concatenate([x_0, x_scaled], axis=-1)


def lorentz_inner_product(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Lorentzian (Minkowski) inner product: <x, y>_L = -x_0*y_0 + sum(x_i*y_i)
    
    Args:
        x, y: (N, D+1) arrays with x_0 as the time-like component (first index)
    
    Returns:
        (N, 1) array of inner products
    """
    time_part = -x[..., 0:1] * y[..., 0:1]
    space_part = np.sum(x[..., 1:] * y[..., 1:], axis=-1, keepdims=True)
    return time_part + space_part


def lorentz_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
    """
    Geodesic distance on the Lorentz hyperboloid: d_L(x, y) = arccosh(-<x, y>_L)
    
    Args:
        x, y: (1, D+1) arrays of Lorentzian vectors
        eps: Small value for numerical stability
    
    Returns:
        Scalar distance
    """
    arg = -lorentz_inner_product(x, y)
    arg = np.clip(arg, 1.0 + eps, None)  # arccosh domain is [1, inf)
    return float(np.arccosh(arg))


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Standard Euclidean distance."""
    return float(np.linalg.norm(x - y))


# ==============================================================================
# Menger Curvature Calculation
# ==============================================================================

def menger_curvature(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, 
                     metric: str = 'euclidean') -> float:
    """
    Calculate Menger Curvature for three points.
    
    Menger Curvature = 4 * Area(triangle) / (a * b * c)
    
    Where a, b, c are the side lengths. 
    - If points are collinear (on a geodesic), Area -> 0, Curvature -> 0.
    - If points form a sharp turn, Curvature is high.
    
    Args:
        x1, x2, x3: Points (in Euclidean or Lorentzian coordinates)
        metric: 'euclidean' or 'lorentz'
    
    Returns:
        Curvature value (lower = straighter path)
    """
    if metric == 'euclidean':
        a = euclidean_distance(x1, x2)
        b = euclidean_distance(x2, x3)
        c = euclidean_distance(x3, x1)
    elif metric == 'lorentz':
        a = lorentz_distance(x1, x2)
        b = lorentz_distance(x2, x3)
        c = lorentz_distance(x3, x1)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Semi-perimeter
    s = (a + b + c) / 2
    
    # Heron's formula for area
    area_sq = s * (s - a) * (s - b) * (s - c)
    area = np.sqrt(max(0, area_sq))
    
    # Menger curvature (avoid division by zero)
    denominator = a * b * c
    if denominator < 1e-12:
        return 0.0
    
    curvature = (4 * area) / denominator
    return float(curvature)


def analyze_trajectory(embeddings: np.ndarray) -> dict:
    """
    Analyze a full reasoning trajectory by computing curvature at each step.
    
    Args:
        embeddings: (T, D) array where T is the number of reasoning steps
    
    Returns:
        Dictionary with Euclidean and Lorentzian curvatures for each triplet
    """
    T = embeddings.shape[0]
    if T < 3:
        return {"error": "Need at least 3 steps"}
    
    # Project to Lorentz manifold
    lorentz_emb = project_to_lorentz(embeddings)
    
    euclidean_curvatures = []
    lorentz_curvatures = []
    
    # Compute curvature for each consecutive triplet (step t-1, t, t+1)
    for i in range(1, T - 1):
        x1 = embeddings[i - 1:i]
        x2 = embeddings[i:i + 1]
        x3 = embeddings[i + 1:i + 2]
        
        x1_L = lorentz_emb[i - 1:i]
        x2_L = lorentz_emb[i:i + 1]
        x3_L = lorentz_emb[i + 1:i + 2]
        
        euc_curv = menger_curvature(x1, x2, x3, metric='euclidean')
        lor_curv = menger_curvature(x1_L, x2_L, x3_L, metric='lorentz')
        
        euclidean_curvatures.append(euc_curv)
        lorentz_curvatures.append(lor_curv)
    
    return {
        "euclidean_curvatures": euclidean_curvatures,
        "lorentz_curvatures": lorentz_curvatures,
        "mean_euclidean": np.mean(euclidean_curvatures),
        "mean_lorentz": np.mean(lorentz_curvatures),
        "reduction_ratio": np.mean(euclidean_curvatures) / (np.mean(lorentz_curvatures) + 1e-9)
    }


# ==============================================================================
# Main Analysis
# ==============================================================================

def load_embeddings_from_dir(data_dir: str) -> dict:
    """
    Load saved embeddings from the experiment data directory.
    
    Args:
        data_dir: Path to 'data/' folder inside an experiment results directory
    
    Returns:
        Dictionary mapping label -> (T, D) embedding array
    """
    emb_dir = os.path.join(data_dir, "embeddings")
    if not os.path.exists(emb_dir):
        raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")
    
    embeddings = {}
    for npy_file in Path(emb_dir).glob("*.npy"):
        label = npy_file.stem
        arr = np.load(npy_file)
        embeddings[label] = arr
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Compare Euclidean vs Hyperbolic Menger Curvature"
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="results/exp1_order0/data",
        help="Path to experiment data directory containing embeddings"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="curvature_comparison.csv",
        help="Output CSV file for results"
    )
    args = parser.parse_args()
    
    print(f"Loading embeddings from: {args.data_dir}")
    embeddings = load_embeddings_from_dir(args.data_dir)
    print(f"Loaded {len(embeddings)} reasoning trajectories")
    
    results = []
    
    for label, emb in embeddings.items():
        print(f"\nAnalyzing: {label} (shape: {emb.shape})")
        
        analysis = analyze_trajectory(emb)
        
        if "error" in analysis:
            print(f"  Skipped: {analysis['error']}")
            continue
        
        results.append({
            "Label": label,
            "Num_Steps": emb.shape[0],
            "Mean_Euclidean_Curvature": round(analysis["mean_euclidean"], 6),
            "Mean_Lorentz_Curvature": round(analysis["mean_lorentz"], 6),
            "Reduction_Ratio": round(analysis["reduction_ratio"], 2),
        })
        
        print(f"  Euclidean Curvature: {analysis['mean_euclidean']:.6f}")
        print(f"  Lorentz Curvature:   {analysis['mean_lorentz']:.6f}")
        print(f"  Reduction Ratio:     {analysis['reduction_ratio']:.2f}x")
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\n✅ Results saved to: {args.output}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("CURVATURE COMPARISON: EUCLIDEAN vs HYPERBOLIC")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    # The punchline
    avg_euc = df["Mean_Euclidean_Curvature"].mean()
    avg_lor = df["Mean_Lorentz_Curvature"].mean()
    ratio = avg_euc / (avg_lor + 1e-9)
    
    print(f"\n📊 OVERALL AVERAGE:")
    print(f"   Euclidean Curvature: {avg_euc:.6f}")
    print(f"   Lorentz Curvature:   {avg_lor:.6f}")
    print(f"   Reduction Factor:    {ratio:.1f}x")
    
    if ratio > 2:
        print("\n🎯 HYPOTHESIS SUPPORTED: Lorentz curvature is significantly lower.")
        print("   This suggests reasoning paths are GEODESICS in hyperbolic space!")
    else:
        print("\n⚠️  Results inconclusive. Curvature reduction is not dramatic.")
        print("   Consider: different projection methods or larger models.")


if __name__ == "__main__":
    main()
