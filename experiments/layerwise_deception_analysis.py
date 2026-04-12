"""
layerwise_deception_analysis.py

Extracts hidden states from EVERY layer for truth vs lie pairs,
computes geometric separation at each layer in both Euclidean and
Poincaré space, trains a linear probe per layer, and produces
publication-ready heatmap visualizations.

This is the core experiment for the paper — it tells us:
  1. Which layers carry the deception signal most strongly
  2. Whether the signal is model-specific or universal
  3. Which space (Euclidean vs Poincaré) separates best at each layer

Usage:
  python experiments/layerwise_deception_analysis.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --data data/deception_pairs.json \
      --out results/layerwise/qwen \
      --device cuda

  python experiments/layerwise_deception_analysis.py \
      --model meta-llama/Meta-Llama-3.1-8B-Instruct \
      --data data/deception_pairs.json \
      --out results/layerwise/llama \
      --device cuda
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy import stats


# ── Poincaré math ─────────────────────────────────────────────────────────────

def l2_normalize(h: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(h, axis=-1, keepdims=True)
    return h / np.maximum(norms, 1e-8)

def to_poincare(h: np.ndarray, c: float = 1.0) -> np.ndarray:
    """L2-normalize then project to Poincaré ball. Output radius ≈ 0.462."""
    h_hat = l2_normalize(h)
    scale = np.tanh(np.sqrt(c) * 0.5)
    return scale * h_hat

def poincare_distance(x: np.ndarray, y: np.ndarray,
                      c: float = 1.0, eps: float = 1e-6) -> float:
    x_sq = np.dot(x, x)
    y_sq = np.dot(y, y)
    xy_sq = np.dot(x - y, x - y)
    num   = 2 * c * xy_sq
    denom = max((1 - c*x_sq) * (1 - c*y_sq), eps)
    arg   = np.sqrt(c) * np.sqrt(min(num/denom, 1.0 - eps))
    return (2.0 / np.sqrt(c)) * np.arctanh(min(arg, 1.0 - eps))

def mean_poincare_distance_between(A: np.ndarray, B: np.ndarray) -> float:
    """Mean pairwise hyperbolic distance between two sets of vectors."""
    A_p = to_poincare(A)
    B_p = to_poincare(B)
    dists = []
    for a, b in zip(A_p, B_p):
        dists.append(poincare_distance(a, b))
    return float(np.mean(dists))

def mean_euclidean_distance_between(A: np.ndarray, B: np.ndarray) -> float:
    """Mean pairwise L2 distance between L2-normalized vectors."""
    A_n = l2_normalize(A)
    B_n = l2_normalize(B)
    return float(np.mean(np.linalg.norm(A_n - B_n, axis=1)))

def cosine_separation(A: np.ndarray, B: np.ndarray) -> float:
    """Mean cosine distance between truth and lie hidden states."""
    A_n = l2_normalize(A)
    B_n = l2_normalize(B)
    cos_sim = np.sum(A_n * B_n, axis=1)
    return float(np.mean(1.0 - cos_sim))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str, device: str):
    print(f"\nLoading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Loaded. Layers: {n_layers}, Hidden dim: {model.config.hidden_size}")
    return tokenizer, model, n_layers


# ── Hidden state extraction — ALL layers at once ──────────────────────────────

def extract_all_layers(
    messages: list,
    tokenizer,
    model,
    max_new_tokens: int,
    device: str,
) -> np.ndarray:
    """
    Generate a response and return hidden states from ALL layers
    at the LAST generated token position.

    Returns array of shape (n_layers + 1, hidden_dim)
    where index 0 = embedding layer, 1..n_layers = transformer layers.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    # out.hidden_states: tuple(n_generated) of tuple(n_layers+1) of tensors
    # We take the FIRST generated token's hidden states
    # (captures the model's state right as it begins generating)
    # Shape per layer: (batch, seq_len, hidden_dim)
    if not out.hidden_states:
        return None

    first_step = out.hidden_states[0]   # tuple of n_layers+1 tensors
    layer_vecs = []
    for layer_h in first_step:
        # Take last token position, first batch
        h = layer_h[0, -1, :].float().cpu().numpy()  # (hidden_dim,)
        layer_vecs.append(h)

    return np.stack(layer_vecs, axis=0)  # (n_layers+1, hidden_dim)


# ── Per-layer probe ───────────────────────────────────────────────────────────

def probe_layer(
    truth_vecs: np.ndarray,
    lie_vecs: np.ndarray,
    use_poincare: bool = False,
) -> dict:
    """
    Train logistic regression probe on hidden states from one layer.
    Returns accuracy, F1, AUC via 5-fold cross-validation.

    truth_vecs: (N, hidden_dim)
    lie_vecs:   (N, hidden_dim)
    """
    if use_poincare:
        X_truth = to_poincare(truth_vecs)
        X_lie   = to_poincare(lie_vecs)
    else:
        X_truth = l2_normalize(truth_vecs)
        X_lie   = l2_normalize(lie_vecs)

    X = np.concatenate([X_truth, X_lie], axis=0)
    y = np.array([0]*len(X_truth) + [1]*len(X_lie))

    # PCA to 50 dims for probe efficiency (60 samples, 3584+ dims = ill-posed)
    # Use mean-difference direction + top components
    diff = X_lie.mean(0) - X_truth.mean(0)
    diff = diff / (np.linalg.norm(diff) + 1e-8)

    # Project onto difference direction + next 9 PCA components
    from numpy.linalg import svd
    X_centered = X - X.mean(0)
    try:
        U, S, Vt = svd(X_centered, full_matrices=False)
        X_proj = X_centered @ Vt[:50].T  # (N, 50)
    except Exception:
        X_proj = X_centered[:, :50]

    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s, aucs = [], [], []

    for train_idx, test_idx in skf.split(X_proj, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_proj[train_idx])
        X_te = scaler.transform(X_proj[test_idx])
        y_tr, y_te = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_tr, y_tr)

        preds = clf.predict(X_te)
        probs = clf.predict_proba(X_te)[:, 1]

        accs.append((preds == y_te).mean())
        f1s.append(f1_score(y_te, preds, zero_division=0))
        try:
            aucs.append(roc_auc_score(y_te, probs))
        except Exception:
            aucs.append(0.5)

    return {
        "accuracy": float(np.mean(accs)),
        "f1":       float(np.mean(f1s)),
        "auc":      float(np.mean(aucs)),
    }


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_layerwise_analysis(
    pairs: list,
    tokenizer,
    model,
    n_layers: int,
    max_new_tokens: int,
    device: str,
    out_dir: Path,
):
    truth_all = []  # list of (n_layers+1, hidden_dim) arrays
    lie_all   = []

    print(f"\nExtracting hidden states from all {n_layers+1} layers...")
    for i, pair in enumerate(pairs):
        print(f"  [{i+1}/{len(pairs)}] {pair['question'][:55]}")

        t_vecs = extract_all_layers(
            pair["truth_messages"], tokenizer, model, max_new_tokens, device
        )
        l_vecs = extract_all_layers(
            pair["lie_messages"],   tokenizer, model, max_new_tokens, device
        )

        if t_vecs is None or l_vecs is None:
            print(f"    WARNING: empty generation for pair {i}, skipping")
            continue

        truth_all.append(t_vecs)
        lie_all.append(l_vecs)

    truth_all = np.stack(truth_all, axis=0)  # (N, n_layers+1, hidden_dim)
    lie_all   = np.stack(lie_all,   axis=0)

    n_pairs   = truth_all.shape[0]
    n_l       = truth_all.shape[1]
    print(f"\nExtracted {n_pairs} pairs × {n_l} layers")

    # Save raw hidden states
    np.save(out_dir / "truth_all_layers.npy", truth_all)
    np.save(out_dir / "lie_all_layers.npy",   lie_all)
    print(f"Raw hidden states saved.")

    # ── Per-layer metrics ─────────────────────────────────────────────────────
    results = []
    print(f"\nComputing per-layer metrics...")

    for l in range(n_l):
        T = truth_all[:, l, :]  # (N, hidden_dim)
        L = lie_all[:, l, :]

        # Geometric separation
        euc_sep  = mean_euclidean_distance_between(T, L)
        hyp_sep  = mean_poincare_distance_between(T, L)
        cos_sep  = cosine_separation(T, L)

        # T-test on norms
        t_norms  = np.linalg.norm(T, axis=1)
        l_norms  = np.linalg.norm(L, axis=1)
        _, p_norm = stats.ttest_ind(t_norms, l_norms, alternative="two-sided")

        # Cosine similarity between mean truth and mean lie direction
        t_mean = T.mean(0); t_mean /= np.linalg.norm(t_mean) + 1e-8
        l_mean = L.mean(0); l_mean /= np.linalg.norm(l_mean) + 1e-8
        mean_cos = float(np.dot(t_mean, l_mean))

        # Linear probe accuracy
        probe_euc = probe_layer(T, L, use_poincare=False)
        probe_hyp = probe_layer(T, L, use_poincare=True)

        # T-test on cosine separation per sample
        cos_truth = np.sum(l2_normalize(T) * t_mean, axis=1)
        cos_lie   = np.sum(l2_normalize(L) * t_mean, axis=1)
        _, p_cos  = stats.ttest_ind(cos_truth, cos_lie, alternative="two-sided")

        row = {
            "layer":          l,
            "euc_separation": euc_sep,
            "hyp_separation": hyp_sep,
            "cos_separation": cos_sep,
            "mean_cos_sim":   mean_cos,
            "p_norm":         float(p_norm),
            "p_cos":          float(p_cos),
            "probe_euc_acc":  probe_euc["accuracy"],
            "probe_euc_auc":  probe_euc["auc"],
            "probe_hyp_acc":  probe_hyp["accuracy"],
            "probe_hyp_auc":  probe_hyp["auc"],
        }
        results.append(row)

        sig = "✅" if p_cos < 0.05 else "  "
        print(f"  Layer {l:3d} {sig} | "
              f"EucSep={euc_sep:.4f} | HypSep={hyp_sep:.4f} | "
              f"ProbeAUC(E)={probe_euc['auc']:.3f} | "
              f"ProbeAUC(H)={probe_hyp['auc']:.3f} | "
              f"p={p_cos:.4f}")

    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = out_dir / "layerwise_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")

    return df


# ── Visualization ─────────────────────────────────────────────────────────────

def make_plots(df: pd.DataFrame, model_name: str, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        n_layers = len(df)
        layers   = df["layer"].values

        fig = plt.figure(figsize=(16, 14))
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

        short_name = model_name.split("/")[-1]

        # ── Plot 1: Euclidean vs Hyperbolic separation ──
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(layers, df["euc_separation"], color="#2196F3",
                 linewidth=1.8, label="Euclidean separation", alpha=0.85)
        ax1.plot(layers, df["hyp_separation"], color="#E91E63",
                 linewidth=1.8, label="Hyperbolic separation (Poincaré)", alpha=0.85)
        ax1.set_xlabel("Layer", fontsize=11)
        ax1.set_ylabel("Mean Distance (truth vs lie)", fontsize=11)
        ax1.set_title(
            f"{short_name} — Truth vs Lie Geometric Separation Per Layer\n"
            f"Higher = more separable = stronger deception signal",
            fontsize=12, fontweight="bold"
        )
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)

        # Mark top-5 layers by hyperbolic separation
        top5_idx = df["hyp_separation"].nlargest(5).index
        for idx in top5_idx:
            ax1.axvline(x=df.loc[idx, "layer"], color="#E91E63",
                        alpha=0.2, linestyle="--", linewidth=1)

        # ── Plot 2: Probe AUC — Euclidean ──
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.bar(layers, df["probe_euc_auc"], color="#2196F3", alpha=0.75, width=0.8)
        ax2.axhline(0.5, color="red", linestyle="--", linewidth=1.2,
                    label="Random baseline")
        ax2.set_xlabel("Layer", fontsize=10)
        ax2.set_ylabel("AUC", fontsize=10)
        ax2.set_title("Linear Probe AUC\n(Euclidean space)", fontsize=11,
                      fontweight="bold")
        ax2.set_ylim(0.3, 1.0)
        ax2.legend(fontsize=9)
        ax2.grid(axis="y", alpha=0.3)

        # ── Plot 3: Probe AUC — Hyperbolic ──
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.bar(layers, df["probe_hyp_auc"], color="#E91E63", alpha=0.75, width=0.8)
        ax3.axhline(0.5, color="red", linestyle="--", linewidth=1.2,
                    label="Random baseline")
        ax3.set_xlabel("Layer", fontsize=10)
        ax3.set_ylabel("AUC", fontsize=10)
        ax3.set_title("Linear Probe AUC\n(Poincaré space)", fontsize=11,
                      fontweight="bold")
        ax3.set_ylim(0.3, 1.0)
        ax3.legend(fontsize=9)
        ax3.grid(axis="y", alpha=0.3)

        # ── Plot 4: p-values (cosine separation) ──
        ax4 = fig.add_subplot(gs[2, 0])
        p_vals = df["p_cos"].values
        colors = ["#4CAF50" if p < 0.05 else "#FF9800" if p < 0.1 else "#9E9E9E"
                  for p in p_vals]
        ax4.bar(layers, -np.log10(np.clip(p_vals, 1e-10, 1.0)),
                color=colors, alpha=0.85, width=0.8)
        ax4.axhline(-np.log10(0.05), color="red", linestyle="--",
                    linewidth=1.2, label="p=0.05")
        ax4.axhline(-np.log10(0.01), color="darkred", linestyle=":",
                    linewidth=1.2, label="p=0.01")
        ax4.set_xlabel("Layer", fontsize=10)
        ax4.set_ylabel("-log10(p-value)", fontsize=10)
        ax4.set_title("Statistical Significance\nper Layer (cosine separation)",
                      fontsize=11, fontweight="bold")
        ax4.legend(fontsize=9)
        ax4.grid(axis="y", alpha=0.3)

        # ── Plot 5: Hyp vs Euc probe advantage ──
        ax5 = fig.add_subplot(gs[2, 1])
        delta_auc = df["probe_hyp_auc"].values - df["probe_euc_auc"].values
        bar_colors = ["#E91E63" if d > 0 else "#2196F3" for d in delta_auc]
        ax5.bar(layers, delta_auc, color=bar_colors, alpha=0.8, width=0.8)
        ax5.axhline(0, color="black", linewidth=1.0)
        ax5.set_xlabel("Layer", fontsize=10)
        ax5.set_ylabel("ΔAUC (Hyperbolic − Euclidean)", fontsize=10)
        ax5.set_title("Hyperbolic Probe Advantage per Layer\n"
                      "Pink = Hyp better | Blue = Euc better",
                      fontsize=11, fontweight="bold")
        ax5.grid(axis="y", alpha=0.3)

        plt.suptitle(
            f"Layer-wise Deception Geometry Analysis — {short_name}",
            fontsize=14, fontweight="bold", y=1.01
        )

        plot_path = out_dir / "layerwise_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {plot_path}")
        plt.close()

        # ── Heatmap: separation across layers ──
        fig2, axes = plt.subplots(1, 2, figsize=(14, 4))

        for ax, col, title, cmap in zip(
            axes,
            ["euc_separation", "hyp_separation"],
            ["Euclidean Separation", "Poincaré Separation"],
            ["Blues", "RdPu"]
        ):
            vals = df[col].values.reshape(1, -1)
            im = ax.imshow(vals, aspect="auto", cmap=cmap,
                           extent=[0, n_layers, 0, 1])
            ax.set_xlabel("Layer", fontsize=11)
            ax.set_yticks([])
            ax.set_title(f"{title}\n{short_name}", fontsize=11, fontweight="bold")
            plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.2)

        plt.suptitle("Truth vs Lie Separation Heatmap Across Layers",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        heatmap_path = out_dir / "separation_heatmap.png"
        plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved → {heatmap_path}")
        plt.close()

    except Exception as e:
        print(f"Plotting failed: {e}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(df, model_name: str):
    print(f"\n{'='*65}")
    print(f" LAYER-WISE SUMMARY — {model_name.split('/')[-1]}")
    print(f"{'='*65}")

    # Top layers by hyperbolic probe AUC
    top5_hyp = df.nlargest(5, "probe_hyp_auc")[
        ["layer", "probe_hyp_auc", "probe_euc_auc", "hyp_separation", "p_cos"]
    ]
    print(f"\nTop 5 layers by Hyperbolic Probe AUC:")
    print(top5_hyp.to_string(index=False))

    # Significant layers
    sig = df[df["p_cos"] < 0.05]
    print(f"\nStatistically significant layers (p<0.05): "
          f"{sig['layer'].tolist()}")

    # Best single layer overall
    best_layer = df.loc[df["probe_hyp_auc"].idxmax(), "layer"]
    best_auc   = df["probe_hyp_auc"].max()
    print(f"\nBest single layer (Hyperbolic AUC): "
          f"Layer {best_layer} → AUC={best_auc:.3f}")

    # Hyperbolic vs Euclidean advantage
    hyp_wins = (df["probe_hyp_auc"] > df["probe_euc_auc"]).sum()
    print(f"\nHyperbolic probe outperforms Euclidean: "
          f"{hyp_wins}/{len(df)} layers ({hyp_wins/len(df)*100:.0f}%)")

    print(f"{'='*65}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          type=str, required=True)
    p.add_argument("--data",           type=str,
                   default="data/deception_pairs.json")
    p.add_argument("--out",            type=str, required=True)
    p.add_argument("--max_new_tokens", type=int, default=100)
    p.add_argument("--device",         type=str, default="cuda")
    p.add_argument("--limit",          type=int, default=None,
                   help="Limit pairs for quick testing")
    return p.parse_args()


def main():
    args = parse_args()
    import pandas as pd

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(args.data) as f:
        pairs = json.load(f)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"Pairs: {len(pairs)}")

    # Load model
    tokenizer, model, n_layers = load_model(args.model, args.device)

    # Save metadata
    meta = {
        "model":     args.model,
        "n_layers":  n_layers,
        "n_pairs":   len(pairs),
        "hidden_dim": model.config.hidden_size,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Run analysis
    df = run_layerwise_analysis(
        pairs, tokenizer, model, n_layers,
        args.max_new_tokens, args.device, out_dir
    )

    # Summary
    print_summary(df, args.model)

    # Plots
    make_plots(df, args.model, out_dir)

    print(f"\nAll outputs → {out_dir}")
    print("Next: run on second model and compare heatmaps.")


if __name__ == "__main__":
    main()