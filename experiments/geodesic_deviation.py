"""
geodesic_deviation.py

EXPERIMENT 1: Baseline — Do lying trajectories deviate more from hyperbolic
geodesics than truthful trajectories?

Computes Menger curvature in both Euclidean and Poincaré space for each
truth and lie trajectory, then runs statistical tests.

Usage:
  python experiments/geodesic_deviation.py \
      --hiddens results/deception_hiddens/qwen \
      --out results/geodesic_deviation/qwen

Outputs:
  - curvature_results.csv    : per-trajectory curvature (E + P) with label
  - summary.json             : mean/std by condition, t-test p-value
  - deviation_plot.png       : box + strip plot
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ── Geometry utilities ────────────────────────────────────────────────────────

def l2_normalize(h: np.ndarray) -> np.ndarray:
    """Normalize each row to unit norm. Shape: (T, d) → (T, d)."""
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return h / norms


def to_poincare(h_normalized: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Project L2-normalized vectors into the Poincaré ball via exponential map at origin.
    Since ||h_hat|| = 1 for all rows, tanh(sqrt(c)*1/2) is constant.
    Output radius ≈ 0.462 for c=1. Shape preserved: (T, d) → (T, d).
    """
    scale = np.tanh(np.sqrt(c) * 0.5)  # ≈ 0.4621
    return scale * h_normalized


def poincare_distance(x: np.ndarray, y: np.ndarray, c: float = 1.0, eps: float = 1e-6) -> float:
    """
    Hyperbolic distance between two points in the Poincaré ball.
    d(x,y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-x) ⊕_c y||)
    Using the simplified form via Möbius addition.
    """
    x_norm_sq = np.dot(x, x)
    y_norm_sq = np.dot(y, y)
    xy_norm_sq = np.dot(x - y, x - y)

    num = 2 * c * xy_norm_sq
    denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    denom = max(denom, eps)

    arg = np.sqrt(c) * np.sqrt(min(num / denom, 1.0 - eps))
    arg = min(arg, 1.0 - eps)
    return (2.0 / np.sqrt(c)) * np.arctanh(arg)


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x - y))


def triangle_area_from_sides(a: float, b: float, c: float) -> float:
    """Heron's formula. Returns 0 if degenerate."""
    s = (a + b + c) / 2.0
    val = s * (s - a) * (s - b) * (s - c)
    return np.sqrt(max(val, 0.0))


def menger_curvature_euclidean(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    """Discrete Menger curvature in Euclidean space for three consecutive points."""
    a = euclidean_distance(q, r)
    b = euclidean_distance(p, r)
    c = euclidean_distance(p, q)
    area = triangle_area_from_sides(a, b, c)
    denom = a * b * c
    if denom < 1e-10:
        return 0.0
    return 4.0 * area / denom


def menger_curvature_poincare(p: np.ndarray, q: np.ndarray, r: np.ndarray,
                               c: float = 1.0) -> float:
    """Discrete Menger curvature using hyperbolic distances."""
    a = poincare_distance(q, r, c)
    b = poincare_distance(p, r, c)
    d = poincare_distance(p, q, c)
    area = triangle_area_from_sides(a, b, d)
    denom = a * b * d
    if denom < 1e-10:
        return 0.0
    return 4.0 * area / denom


def mean_trajectory_curvature(trajectory: np.ndarray, geometry: str = "euclidean") -> float:
    """
    Compute mean Menger curvature over all consecutive triplets in a trajectory.
    trajectory: (T, d) — raw hidden states
    geometry: "euclidean" or "poincare"
    """
    T = trajectory.shape[0]
    if T < 3:
        return 0.0

    # Preprocess
    normed = l2_normalize(trajectory)
    if geometry == "poincare":
        pts = to_poincare(normed)
        curvature_fn = menger_curvature_poincare
    else:
        pts = normed  # use normalized for fair comparison
        curvature_fn = menger_curvature_euclidean

    curvatures = []
    for t in range(T - 2):
        k = curvature_fn(pts[t], pts[t+1], pts[t+2])
        curvatures.append(k)

    return float(np.mean(curvatures)) if curvatures else 0.0


# ── Analysis ──────────────────────────────────────────────────────────────────

def load_trajectories(base_dir: Path) -> dict:
    """Load all trajectory numpy arrays. Returns dict with 'truth' and 'lie' lists."""
    result = {}
    for condition in ["truth", "lie"]:
        cond_dir = base_dir / condition
        files = sorted(cond_dir.glob("pair_*.npy"))
        result[condition] = [np.load(f) for f in files]
        print(f"Loaded {len(result[condition])} {condition} trajectories "
              f"from {cond_dir}")
    return result


def compute_all_curvatures(trajectories: dict) -> pd.DataFrame:
    """Compute E and P curvature for every trajectory."""
    rows = []
    for condition, trajs in trajectories.items():
        for i, traj in enumerate(trajs):
            kappa_e = mean_trajectory_curvature(traj, "euclidean")
            kappa_p = mean_trajectory_curvature(traj, "poincare")
            ratio   = kappa_p / kappa_e if kappa_e > 1e-8 else np.nan
            rows.append({
                "condition": condition,
                "pair_idx":  i,
                "kappa_euclidean": kappa_e,
                "kappa_poincare":  kappa_p,
                "p_e_ratio":       ratio,
            })
    return pd.DataFrame(rows)


def run_stats(df: pd.DataFrame) -> dict:
    """T-test and summary stats between truth and lie conditions."""
    truth = df[df.condition == "truth"]
    lie   = df[df.condition == "lie"]

    summary = {}
    for metric in ["kappa_euclidean", "kappa_poincare", "p_e_ratio"]:
        t_vals = truth[metric].dropna().values
        l_vals = lie[metric].dropna().values
        t_stat, p_val = stats.ttest_ind(t_vals, l_vals, alternative="two-sided")
        # H1: truth curvature < lie curvature (lies deviate more)
        summary[metric] = {
            "truth_mean":  float(np.mean(t_vals)),
            "truth_std":   float(np.std(t_vals)),
            "lie_mean":    float(np.mean(l_vals)),
            "lie_std":     float(np.std(l_vals)),
            "delta":       float(np.mean(l_vals) - np.mean(t_vals)),
            "delta_pct":   float((np.mean(l_vals) - np.mean(t_vals)) / (np.mean(t_vals) + 1e-8) * 100),
            "t_stat":      float(t_stat),
            "p_value":     float(p_val),
            "significant": bool(p_val < 0.05),
        }

    return summary


def make_plot(df: pd.DataFrame, out_dir: Path):
    """Box + strip plot of curvature by condition."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        conditions = ["truth", "lie"]
        colors = {"truth": "#2196F3", "lie": "#F44336"}

        for ax, metric, label in zip(
            axes,
            ["kappa_euclidean", "kappa_poincare"],
            ["Euclidean Menger Curvature (κ_E)", "Poincaré Menger Curvature (κ_P)"],
        ):
            data_by_cond = [df[df.condition == c][metric].values for c in conditions]

            bp = ax.boxplot(
                data_by_cond,
                labels=["Truth", "Lie"],
                patch_artist=True,
                widths=0.5,
                medianprops=dict(color="black", linewidth=2),
            )
            for patch, cond in zip(bp["boxes"], conditions):
                patch.set_facecolor(colors[cond])
                patch.set_alpha(0.7)

            # Strip plot overlay
            for j, (cond, vals) in enumerate(zip(conditions, data_by_cond)):
                jitter = np.random.RandomState(42).uniform(-0.1, 0.1, len(vals))
                ax.scatter(
                    np.full(len(vals), j + 1) + jitter,
                    vals,
                    color=colors[cond],
                    alpha=0.5,
                    s=20,
                    zorder=3,
                )

            ax.set_title(label, fontsize=11)
            ax.set_ylabel("Curvature (κ)", fontsize=10)
            ax.set_xlabel("Condition", fontsize=10)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle(
            "Deception Geodesic Deviation:\nLying trajectories curve more in hyperbolic space",
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plot_path = out_dir / "deviation_plot.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {plot_path}")
        plt.close()

    except ImportError:
        print("matplotlib not available, skipping plot.")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hiddens", type=str, required=True,
                   help="Directory from extract_deception_hiddens.py")
    p.add_argument("--out",     type=str, required=True,
                   help="Output directory for results")
    return p.parse_args()


def main():
    args = parse_args()
    hiddens_dir = Path(args.hiddens)
    out_dir     = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    trajectories = load_trajectories(hiddens_dir)

    # Compute curvatures
    print("\nComputing curvatures (this is fast, pure numpy)...")
    df = compute_all_curvatures(trajectories)

    # Save CSV
    csv_path = out_dir / "curvature_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Curvature table → {csv_path}")
    print(df.groupby("condition")[["kappa_euclidean", "kappa_poincare", "p_e_ratio"]].mean())

    # Stats
    summary = run_stats(df)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print key result
    print("\n── KEY RESULT ──────────────────────────────────────────────────")
    for metric in ["kappa_euclidean", "kappa_poincare"]:
        s = summary[metric]
        sig = "✅ SIGNIFICANT" if s["significant"] else "❌ not significant"
        print(f"{metric}:")
        print(f"  Truth: {s['truth_mean']:.4f} ± {s['truth_std']:.4f}")
        print(f"  Lie:   {s['lie_mean']:.4f} ± {s['lie_std']:.4f}")
        print(f"  Δ: +{s['delta_pct']:.1f}%  |  p = {s['p_value']:.4f}  {sig}")
        print()

    # Plot
    make_plot(df, out_dir)
    print(f"\nAll results → {out_dir}")


if __name__ == "__main__":
    main()
