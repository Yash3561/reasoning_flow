#!/usr/bin/env python3
"""
Hallucination Detection via Hyperbolic Geodesic Deviation

Hypothesis: When an LLM hallucinates or makes an invalid inference, the hidden
state trajectory deviates more from a geodesic path — producing a curvature spike
at that reasoning step.

This script:
1. Extracts per-step Menger curvature profiles from reasoning trajectories
2. Identifies high-curvature steps (potential failure points)
3. If correctness labels provided: computes correlation between curvature and errors
4. Includes a synthetic demonstration using injected noise at step 5
5. Generates visualizations: curvature profiles, heatmaps, per-logic comparison

Usage:
    # Basic mode — visualize curvature profiles
    python experiments/hallucination_probe.py \
        --hf_model Qwen/Qwen2.5-0.5B \
        --data_file data/demo_subset.json \
        --output results/hallucination_probe/

    # With correctness labels
    python experiments/hallucination_probe.py \
        --hf_model Qwen/Qwen2.5-0.5B \
        --data_file data/demo_subset.json \
        --labels_file data/correctness_labels.json \
        --output results/hallucination_probe/
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class LogicItem:
    logic: str
    topic: Optional[str]
    steps: List[str]


def split_cot_steps(text: str) -> List[str]:
    lines = [ln.strip() for ln in re.split(r"\r?\n", text) if ln.strip()]
    if lines:
        return lines
    parts = re.split(r"(?<=[\.!?])\s+|\.(?=\s|$)", text)
    return [p.strip() for p in parts if p and p.strip()]


def load_dataset(path: str, sections: str = "all") -> List[LogicItem]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items: List[LogicItem] = []
    keep = None if sections == "all" else {s.strip() for s in sections.split(",")}
    for logic_key, seq_list in data.items():
        if not isinstance(seq_list, list):
            continue
        if keep and logic_key not in keep:
            continue
        for rec in seq_list:
            if not isinstance(rec, dict) or "steps" not in rec:
                continue
            steps = rec["steps"]
            if isinstance(steps, str):
                steps = split_cot_steps(steps)
            elif isinstance(steps, list) and len(steps) == 1 and isinstance(steps[0], str):
                steps = split_cot_steps(steps[0])
            topic = rec.get("topic")
            items.append(LogicItem(logic=str(logic_key), topic=str(topic) if topic else None, steps=steps))
    return items


# ── Geometry ──────────────────────────────────────────────────────────────────

def menger_curvature(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> float:
    """Menger curvature for 3 consecutive trajectory points."""
    a = float(np.linalg.norm(x2 - x1))
    b = float(np.linalg.norm(x3 - x2))
    c = float(np.linalg.norm(x3 - x1))
    s = (a + b + c) / 2.0
    area_sq = max(0.0, s * (s - a) * (s - b) * (s - c))
    area = np.sqrt(area_sq)
    denom = a * b * c
    return float(4.0 * area / denom) if denom > 1e-12 else 0.0


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + eps)


def compute_curvature_profile(embeddings: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Compute per-step Menger curvature for a trajectory.
    Returns array of T-2 curvature values (undefined at endpoints).
    """
    if normalize:
        embeddings = l2_normalize(embeddings)
    T = embeddings.shape[0]
    if T < 3:
        return np.zeros(max(0, T - 2))
    profile = []
    for i in range(1, T - 1):
        c = menger_curvature(embeddings[i - 1], embeddings[i], embeddings[i + 1])
        profile.append(c)
    return np.array(profile, dtype=np.float32)


def trajectory_stats(profile: np.ndarray) -> dict:
    if len(profile) == 0:
        return {"max_curvature": 0, "mean_curvature": 0, "curvature_variance": 0,
                "spike_count": 0, "geodesic_deviation": 0}
    mean_c = float(np.mean(profile))
    max_c = float(np.max(profile))
    var_c = float(np.var(profile))
    spikes = int(np.sum(profile > 2.0 * mean_c))
    deviation = max_c / (mean_c + 1e-9)
    return {
        "max_curvature": round(max_c, 6),
        "mean_curvature": round(mean_c, 6),
        "curvature_variance": round(var_c, 6),
        "spike_count": spikes,
        "geodesic_deviation": round(deviation, 4),
    }


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        model = AutoModel.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def get_step_vectors(tokenizer, model, steps: List[str], device: str) -> np.ndarray:
    context = ""
    prev_len = 0
    vecs = []
    for t, step in enumerate(steps):
        context = step if t == 0 else (context + "\n" + step)
        enc = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)

        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hs = outputs.last_hidden_state
        else:
            hs = outputs.hidden_states[-1]

        L = input_ids.shape[1]
        start = min(prev_len, L - 1)
        step_slice = hs[:, start:, :]
        if step_slice.shape[1] == 0:
            step_slice = hs[:, -1:, :]
        v = step_slice.mean(dim=1).squeeze(0).detach().float().cpu().numpy()
        vecs.append(v)
        prev_len = L
    return np.stack(vecs)  # [T, D]


# ── Synthetic demonstration ───────────────────────────────────────────────────

def generate_synthetic_demo():
    """
    Synthetic trajectories to illustrate the concept:
    - 'Correct' reasoning: smooth trajectory, low curvature
    - 'Hallucinated' reasoning: noise injected at step 5, curvature spike
    """
    rng = np.random.default_rng(42)
    T, D = 9, 64

    # Correct trajectory: smooth linear trend + small noise
    base_direction = rng.standard_normal(D)
    base_direction /= np.linalg.norm(base_direction)

    correct_trajs = []
    for _ in range(2):
        noise_scale = 0.05
        traj = np.stack([i * base_direction + rng.standard_normal(D) * noise_scale for i in range(T)])
        correct_trajs.append(traj)

    # Hallucinated trajectory: same as correct but with large noise at step 5
    hallucinated_trajs = []
    for _ in range(2):
        noise_scale = 0.05
        traj = np.stack([i * base_direction + rng.standard_normal(D) * noise_scale for i in range(T)])
        # Inject hallucination: large random jump at step 5
        traj[4] += rng.standard_normal(D) * 2.0  # Step 5 (index 4)
        hallucinated_trajs.append(traj)

    return correct_trajs, hallucinated_trajs


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_curvature_profiles(profiles_by_label: Dict[str, np.ndarray], output_path: str, title: str):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Subplot 1: Individual profiles colored by logic type
    logic_colors = {"logicA": "#3498DB", "logicB": "#27AE60", "logicC": "#E74C3C"}
    for label, profile in profiles_by_label.items():
        logic_key = label.split(":")[0].lower() if ":" in label else label[:6].lower()
        color = logic_colors.get(logic_key, "#888888")
        x = np.arange(1, len(profile) + 1)
        axes[0].plot(x, profile, alpha=0.4, color=color, linewidth=1)

    # Add legend proxies
    from matplotlib.lines import Line2D
    legend_lines = [Line2D([0], [0], color=c, linewidth=2, label=k)
                    for k, c in logic_colors.items()]
    axes[0].legend(handles=legend_lines)
    axes[0].set_xlabel("Reasoning Step")
    axes[0].set_ylabel("Menger Curvature")
    axes[0].set_title("Per-Step Curvature Profiles by Logic Type")
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Mean ± std per step across all trajectories
    all_profiles = [p for p in profiles_by_label.values() if len(p) > 0]
    if all_profiles:
        min_len = min(len(p) for p in all_profiles)
        aligned = np.stack([p[:min_len] for p in all_profiles])
        mean_profile = aligned.mean(axis=0)
        std_profile = aligned.std(axis=0)
        x = np.arange(1, min_len + 1)
        axes[1].plot(x, mean_profile, color="#8E44AD", linewidth=2, label="Mean curvature")
        axes[1].fill_between(x, mean_profile - std_profile, mean_profile + std_profile,
                             alpha=0.2, color="#8E44AD")
        axes[1].axhline(mean_profile.mean() * 2, color="#E74C3C", linestyle="--", alpha=0.7,
                        label="Spike threshold (2× mean)")
        axes[1].set_xlabel("Reasoning Step")
        axes[1].set_ylabel("Mean Menger Curvature ± Std")
        axes[1].set_title("Aggregated Curvature Profile (Mean ± Std)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Profile plot saved: {output_path}")


def plot_curvature_heatmap(profiles_by_label: Dict[str, np.ndarray], output_path: str):
    labels = list(profiles_by_label.keys())
    profiles = [profiles_by_label[l] for l in labels]
    if not profiles:
        return
    max_len = max(len(p) for p in profiles)
    padded = np.full((len(profiles), max_len), np.nan)
    for i, p in enumerate(profiles):
        padded[i, :len(p)] = p

    fig, ax = plt.subplots(figsize=(max(8, max_len), max(6, len(labels) * 0.3 + 2)))
    im = ax.imshow(padded, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Menger Curvature")
    ax.set_xlabel("Reasoning Step Index")
    ax.set_ylabel("Trajectory")
    ax.set_yticks(range(len(labels)))
    if len(labels) <= 40:
        ax.set_yticklabels(labels, fontsize=7)
    else:
        ax.set_yticklabels([])
    ax.set_title("Curvature Heatmap: All Trajectories × All Steps\n(Red = High Curvature = Potential Failure Point)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved: {output_path}")


def plot_synthetic_demo(correct_trajs, hallucinated_trajs, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, traj in enumerate(correct_trajs):
        p = compute_curvature_profile(traj)
        axes[0].plot(np.arange(1, len(p)+1), p, label=f"Correct {i+1}", color="#27AE60", lw=2)
    for i, traj in enumerate(hallucinated_trajs):
        p = compute_curvature_profile(traj)
        axes[0].plot(np.arange(1, len(p)+1), p, label=f"Hallucinated {i+1}", color="#E74C3C",
                     lw=2, linestyle="--")
    axes[0].axvline(x=4, color="#E74C3C", alpha=0.3, linestyle=":", label="Injected noise at step 5")
    axes[0].set_xlabel("Reasoning Step")
    axes[0].set_ylabel("Menger Curvature")
    axes[0].set_title("Curvature Profiles: Correct vs Hallucinated")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    correct_maxs = [compute_curvature_profile(t).max() for t in correct_trajs]
    hallu_maxs = [compute_curvature_profile(t).max() for t in hallucinated_trajs]
    categories = (["Correct"] * len(correct_maxs) + ["Hallucinated"] * len(hallu_maxs))
    values = correct_maxs + hallu_maxs
    colors = ["#27AE60" if c == "Correct" else "#E74C3C" for c in categories]
    x_pos = list(range(len(values)))
    axes[1].bar(x_pos, values, color=colors, edgecolor="black")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(categories)
    axes[1].set_ylabel("Max Curvature (Geodesic Deviation)")
    axes[1].set_title("Max Curvature: Correct vs Hallucinated\n(Higher = More Deviation = Potential Hallucination)")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Synthetic Demonstration: Curvature as Hallucination Signal", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Synthetic demo plot saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Hallucination probe via curvature")
    ap.add_argument("--hf_model", required=True, help="HuggingFace model ID")
    ap.add_argument("--data_file", default="data/demo_subset.json")
    ap.add_argument("--sections", default="all")
    ap.add_argument("--labels_file", default=None,
                    help="JSON file mapping label -> 0 (correct) or 1 (incorrect)")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="results/hallucination_probe/")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("HALLUCINATION PROBE: GEODESIC DEVIATION ANALYSIS")
    print("=" * 60)

    # Step 1: Synthetic demo (always runs first)
    print("\n[1/4] Generating synthetic demonstration...")
    correct_trajs, hallucinated_trajs = generate_synthetic_demo()
    plot_synthetic_demo(
        correct_trajs, hallucinated_trajs,
        os.path.join(args.output, "synthetic_demo.png")
    )
    print("  Synthetic demo shows: injected noise at step 5 creates a clear curvature spike")
    print("  This is the EXPECTED pattern for real hallucinations")

    # Step 2: Load real data
    print(f"\n[2/4] Loading dataset: {args.data_file}")
    items = load_dataset(args.data_file, args.sections)
    print(f"  Loaded {len(items)} items")

    # Step 3: Extract embeddings and compute curvature profiles
    print(f"\n[3/4] Loading model: {args.hf_model}")
    tokenizer, model = load_model(args.hf_model, args.device)

    profiles_by_label: Dict[str, np.ndarray] = {}
    stats_rows = []

    print(f"\n  Computing curvature profiles for {len(items)} trajectories...")
    for idx, item in enumerate(items):
        label = f"{item.logic}:{item.topic or 'abstract'}"
        print(f"  [{idx+1}/{len(items)}] {label}")
        emb = get_step_vectors(tokenizer, model, item.steps, args.device)
        profile = compute_curvature_profile(emb, normalize=True)
        profiles_by_label[label] = profile
        stats = trajectory_stats(profile)

        row = {"label": label, "logic": item.logic, "topic": item.topic or "abstract"}
        row.update(stats)
        for i, cv in enumerate(profile):
            row[f"step_{i+2}_curvature"] = round(float(cv), 6)  # steps 2..T-1 (interior)
        stats_rows.append(row)

    # Step 4: Save results and plots
    print(f"\n[4/4] Saving results to: {args.output}")

    df = pd.DataFrame(stats_rows)
    csv_path = os.path.join(args.output, "curvature_profiles.csv")
    df.to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}")

    plot_curvature_profiles(
        profiles_by_label,
        os.path.join(args.output, "curvature_profiles.png"),
        title=f"Hallucination Probe: Per-Step Curvature Profiles\n{args.hf_model}"
    )

    plot_curvature_heatmap(
        profiles_by_label,
        os.path.join(args.output, "curvature_heatmap.png")
    )

    # Logic-type mean curvature bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    logic_means = df.groupby("logic")["mean_curvature"].mean().reset_index()
    colors = ["#3498DB", "#27AE60", "#E74C3C"]
    ax.bar(logic_means["logic"], logic_means["mean_curvature"],
           color=colors[:len(logic_means)], edgecolor="black")
    ax.set_ylabel("Mean Menger Curvature")
    ax.set_title("Mean Curvature by Logic Type\n(Lower = More Geodesic = Stronger Reasoning)")
    for i, (_, row) in enumerate(logic_means.iterrows()):
        ax.text(i, row["mean_curvature"] + 0.01, f"{row['mean_curvature']:.3f}",
                ha="center", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "curvature_by_logic.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Optional: correlation with labels
    if args.labels_file and os.path.exists(args.labels_file):
        print(f"\n  Loading correctness labels: {args.labels_file}")
        with open(args.labels_file, "r") as f:
            labels_map = json.load(f)  # {label: 0/1}

        matched = df[df["label"].isin(labels_map)].copy()
        if len(matched) > 0:
            matched["correct"] = matched["label"].map(labels_map)
            corr = matched["max_curvature"].corr(matched["correct"])
            print(f"\n  Pearson correlation (max_curvature vs error label): {corr:.4f}")
            if abs(corr) > 0.3:
                print("  → SIGNIFICANT: Higher curvature correlates with errors!")
            else:
                print("  → Weak correlation — need more data or larger model")
        else:
            print("  Warning: No label matches found in dataset")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    summary_cols = ["logic", "mean_curvature", "max_curvature", "geodesic_deviation", "spike_count"]
    print(df[summary_cols].groupby("logic").mean().round(4).to_string())

    print(f"\n★ Highest curvature trajectory:")
    worst = df.loc[df["max_curvature"].idxmax()]
    print(f"  {worst['label']} — max={worst['max_curvature']:.4f}, deviation={worst['geodesic_deviation']:.2f}")

    print(f"\n★ Lowest curvature (most geodesic) trajectory:")
    best = df.loc[df["mean_curvature"].idxmin()]
    print(f"  {best['label']} — mean={best['mean_curvature']:.4f}, deviation={best['geodesic_deviation']:.2f}")

    print("\n✅ Done. Check the output folder for:")
    print(f"  {args.output}synthetic_demo.png     — synthetic hallucination demo")
    print(f"  {args.output}curvature_profiles.png — per-step profile plots")
    print(f"  {args.output}curvature_heatmap.png  — all trajectories × all steps")
    print(f"  {args.output}curvature_by_logic.png — mean curvature per logic type")
    print(f"  {args.output}curvature_profiles.csv — full data table")


if __name__ == "__main__":
    main()
