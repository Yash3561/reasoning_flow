#!/usr/bin/env python3
"""
Layer-by-Layer Curvature Analysis

Analyzes Menger curvature at each transformer layer to find which layers
encode reasoning structure most strongly. Low curvature = straighter path =
more structured reasoning.

Usage:
    python experiments/layer_analysis.py \
        --hf_model Qwen/Qwen2.5-0.5B \
        --data_file data/demo_subset.json \
        --output results/layer_analysis/curvature_by_layer.csv
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
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


# ── Data structures (mirrors cot-hidden-dynamic.py) ───────────────────────────

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


# ── Geometry helpers ──────────────────────────────────────────────────────────

def menger_curvature(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> float:
    """Menger curvature κ = 4*Area / (a*b*c) for three consecutive trajectory points."""
    a = float(np.linalg.norm(x2 - x1))
    b = float(np.linalg.norm(x3 - x2))
    c = float(np.linalg.norm(x3 - x1))
    s = (a + b + c) / 2.0
    area_sq = max(0.0, s * (s - a) * (s - b) * (s - c))
    area = np.sqrt(area_sq)
    denom = a * b * c
    return float(4.0 * area / denom) if denom > 1e-12 else 0.0


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str, device: str):
    print(f"  Loading tokenizer: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"  Loading model ({dtype}) ...")
    try:
        model = AutoModel.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()
    print(f"  Model loaded on {device}")
    return tok, model


# ── Hidden state extraction (ALL layers) ─────────────────────────────────────

@torch.no_grad()
def get_all_layer_step_vectors(
    tokenizer, model, steps: List[str], device: str
) -> np.ndarray:
    """
    Returns array of shape [num_layers+1, T, D] where:
      - dim 0: layer index (0 = embedding layer, 1..L = transformer layers)
      - dim 1: step index
      - dim 2: hidden dimension
    """
    context = ""
    prev_len = 0
    layer_step_vecs = []  # list of [num_layers+1, D] per step

    for t, step in enumerate(steps):
        context = step if t == 0 else (context + "\n" + step)
        enc = tokenizer(context, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        all_hs = outputs.hidden_states  # tuple of [1, L_seq, D], length = num_layers + 1

        L_seq = input_ids.shape[1]
        start = min(prev_len, L_seq - 1)

        layer_vecs = []
        for hs in all_hs:
            # step_mean pool over new tokens
            step_slice = hs[0, start:, :]  # [new_tokens, D]
            if step_slice.shape[0] == 0:
                step_slice = hs[0, -1:, :]
            v = step_slice.mean(dim=0).detach().float().cpu().numpy()
            layer_vecs.append(v)

        layer_step_vecs.append(np.stack(layer_vecs))  # [num_layers+1, D]
        prev_len = L_seq

    # Stack over steps: [T, num_layers+1, D] -> transpose to [num_layers+1, T, D]
    arr = np.stack(layer_step_vecs, axis=0)  # [T, num_layers+1, D]
    arr = arr.transpose(1, 0, 2)             # [num_layers+1, T, D]
    return arr


# ── Per-layer analysis ────────────────────────────────────────────────────────

def analyze_layer(embeddings_T_D: np.ndarray, logic_vecs_by_group: Dict[str, List[np.ndarray]]) -> dict:
    """
    embeddings_T_D: shape [T, D] — trajectory for one item at one layer
    logic_vecs_by_group: {logic_key: [array_T_D, ...]} — all trajectories for within/between logic comparison
    """
    T = embeddings_T_D.shape[0]

    # Menger curvature
    curvatures = []
    for i in range(1, T - 1):
        curv = menger_curvature(embeddings_T_D[i - 1], embeddings_T_D[i], embeddings_T_D[i + 1])
        curvatures.append(curv)
    mean_curv = float(np.mean(curvatures)) if curvatures else 0.0

    return {"mean_curvature": mean_curv}


def compute_logic_separability(
    all_layer_data: Dict[str, np.ndarray], layer_idx: int
) -> float:
    """
    Compute within-logic vs between-logic cosine similarity at a given layer.
    Returns separability = mean_between_logic - mean_within_logic
    (more negative = better separated = logic signal stronger)
    Actually returns: mean_within - mean_between (higher = better separated)
    """
    # Group by logic
    groups: Dict[str, List[np.ndarray]] = {}
    for label, arr in all_layer_data.items():
        logic = label.split(":")[0] if ":" in label else label[:6]
        # Use mean-pooled trajectory vector as representation
        traj = arr[layer_idx]  # [T, D]
        rep = traj.mean(axis=0)
        groups.setdefault(logic, []).append(rep)

    logic_keys = list(groups.keys())
    if len(logic_keys) < 2:
        return 0.0

    within_sims, between_sims = [], []
    for i, lk in enumerate(logic_keys):
        vecs = groups[lk]
        for vi in range(len(vecs)):
            for vj in range(vi + 1, len(vecs)):
                within_sims.append(cosine_sim(vecs[vi], vecs[vj]))
        for lk2 in logic_keys[i + 1:]:
            for v1 in vecs:
                for v2 in groups[lk2]:
                    between_sims.append(cosine_sim(v1, v2))

    mean_within = float(np.mean(within_sims)) if within_sims else 0.0
    mean_between = float(np.mean(between_sims)) if between_sims else 0.0
    return mean_within - mean_between  # higher = same logic is more similar than cross-logic


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Layer-by-layer curvature analysis")
    ap.add_argument("--hf_model", required=True, help="HuggingFace model ID")
    ap.add_argument("--data_file", default="data/demo_subset.json")
    ap.add_argument("--sections", default="all")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="results/layer_analysis/curvature_by_layer.csv")
    args = ap.parse_args()

    out_dir = os.path.dirname(args.output)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("LAYER-BY-LAYER CURVATURE ANALYSIS")
    print("=" * 60)

    items = load_dataset(args.data_file, args.sections)
    print(f"\nLoaded {len(items)} reasoning items")

    tokenizer, model = load_model(args.hf_model, args.device)

    # Try to get number of layers
    try:
        num_transformer_layers = model.config.num_hidden_layers
    except Exception:
        num_transformer_layers = None

    print(f"\nExtracting all-layer hidden states for {len(items)} items...")
    all_layer_data: Dict[str, np.ndarray] = {}  # label -> [num_layers+1, T, D]

    for idx, item in enumerate(items):
        label = f"{item.logic}:{item.topic or 'abstract'}"
        print(f"  [{idx+1}/{len(items)}] {label} ({len(item.steps)} steps)")
        arr = get_all_layer_step_vectors(tokenizer, model, item.steps, args.device)
        all_layer_data[label] = arr

    num_layers_plus_1 = next(iter(all_layer_data.values())).shape[0]
    print(f"\nGot {num_layers_plus_1} layer outputs (embedding + {num_layers_plus_1 - 1} transformer layers)")

    print("\nComputing per-layer curvature and logic separability...")
    results = []
    for layer_idx in range(num_layers_plus_1):
        curvatures = []
        for label, arr in all_layer_data.items():
            traj = arr[layer_idx]  # [T, D]
            T = traj.shape[0]
            layer_curvs = []
            for i in range(1, T - 1):
                c = menger_curvature(traj[i - 1], traj[i], traj[i + 1])
                layer_curvs.append(c)
            if layer_curvs:
                curvatures.append(float(np.mean(layer_curvs)))

        mean_curv = float(np.mean(curvatures)) if curvatures else 0.0
        std_curv = float(np.std(curvatures)) if curvatures else 0.0
        sep = compute_logic_separability(all_layer_data, layer_idx)

        results.append({
            "layer": layer_idx,
            "mean_euclidean_curvature": round(mean_curv, 6),
            "std_curvature": round(std_curv, 6),
            "logic_separability": round(sep, 6),
        })
        print(f"  Layer {layer_idx:3d}: curvature={mean_curv:.4f}, separability={sep:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")

    # Identify key layers
    min_curv_layer = df.loc[df["mean_euclidean_curvature"].idxmin(), "layer"]
    max_sep_layer = df.loc[df["logic_separability"].idxmax(), "layer"]
    print(f"\n★ Minimum curvature at layer: {min_curv_layer}")
    print(f"★ Maximum logic separability at layer: {max_sep_layer}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 5))
    color1, color2 = "#E74C3C", "#27AE60"

    ax1.plot(df["layer"], df["mean_euclidean_curvature"], color=color1, lw=2, label="Menger Curvature")
    ax1.fill_between(df["layer"],
                     df["mean_euclidean_curvature"] - df["std_curvature"],
                     df["mean_euclidean_curvature"] + df["std_curvature"],
                     color=color1, alpha=0.15)
    ax1.set_xlabel("Transformer Layer", fontsize=12)
    ax1.set_ylabel("Mean Menger Curvature", color=color1, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axvline(min_curv_layer, color=color1, linestyle="--", alpha=0.5,
                label=f"Min curvature (layer {min_curv_layer})")

    ax2 = ax1.twinx()
    ax2.plot(df["layer"], df["logic_separability"], color=color2, lw=2, linestyle="-.",
             label="Logic Separability")
    ax2.set_ylabel("Logic Separability (within − between)", color=color2, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.axvline(max_sep_layer, color=color2, linestyle="--", alpha=0.5,
                label=f"Max separability (layer {max_sep_layer})")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    plt.title(f"Layer-by-Layer Curvature & Logic Separability\n{args.hf_model}", fontsize=13)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "curvature_by_layer.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")
    plt.close()

    print("\n" + "=" * 60)
    print("TOP 5 LAYERS BY LOGIC SEPARABILITY")
    print("=" * 60)
    print(df.nlargest(5, "logic_separability")[["layer", "mean_euclidean_curvature", "logic_separability"]].to_string(index=False))


if __name__ == "__main__":
    main()
