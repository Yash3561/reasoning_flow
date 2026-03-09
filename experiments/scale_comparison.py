#!/usr/bin/env python3
"""
Model Scale Comparison: Hyperbolic Curvature Reduction Across Model Sizes

Tests whether the hyperbolic geodesic hypothesis (reasoning paths are straighter
in Poincaré space) holds across different model scales.

Usage:
    python experiments/scale_comparison.py \
        --hf_models "Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B,Qwen/Qwen2.5-3B" \
        --data_file data/demo_subset.json \
        --output results/scale_comparison/

Note: Requires sufficient GPU memory. Use --load_in_4bit for large models.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


# ── Hyperbolic geometry ───────────────────────────────────────────────────────

def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + eps)


def project_to_poincare(x: np.ndarray, c: float = 1.0, eps: float = 1e-5) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    sqrt_c = np.sqrt(c)
    norms_safe = np.maximum(norms, eps)
    coeff = np.tanh(sqrt_c * norms_safe / 2) / norms_safe
    return x * coeff


def poincare_distance(x: np.ndarray, y: np.ndarray, c: float = 1.0, eps: float = 1e-5) -> float:
    x, y = x.flatten(), y.flatten()
    sqrt_c = np.sqrt(c)
    x_norm_sq = min(float(np.sum(x ** 2)), 1 - eps)
    y_norm_sq = min(float(np.sum(y ** 2)), 1 - eps)
    diff_norm_sq = float(np.sum((x - y) ** 2))
    num = 2.0 * diff_norm_sq
    denom = (1.0 - x_norm_sq) * (1.0 - y_norm_sq)
    arg = sqrt_c * np.sqrt(num / (denom + eps))
    arg = min(arg, 1.0 - eps)
    return float((2.0 / sqrt_c) * np.arctanh(arg))


def menger_curvature_euclidean(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> float:
    a = float(np.linalg.norm(x2 - x1))
    b = float(np.linalg.norm(x3 - x2))
    c = float(np.linalg.norm(x3 - x1))
    s = (a + b + c) / 2.0
    area_sq = max(0.0, s * (s - a) * (s - b) * (s - c))
    area = np.sqrt(area_sq)
    denom = a * b * c
    return float(4.0 * area / denom) if denom > 1e-12 else 0.0


def menger_curvature_poincare(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, c: float = 1.0) -> float:
    a = poincare_distance(x1, x2, c=c)
    b = poincare_distance(x2, x3, c=c)
    cs = poincare_distance(x3, x1, c=c)
    s = (a + b + cs) / 2.0
    area_sq = max(0.0, s * (s - a) * (s - b) * (s - cs))
    area = np.sqrt(area_sq)
    denom = a * b * cs
    return float(4.0 * area / denom) if denom > 1e-12 else 0.0


def analyze_trajectory_both(embeddings: np.ndarray, c: float = 1.0) -> dict:
    T = embeddings.shape[0]
    if T < 3:
        return {"euc": 0.0, "poin": 0.0, "reduction_pct": 0.0}
    normalized = l2_normalize(embeddings)
    poincare_emb = project_to_poincare(normalized, c=c)

    euc_curvs, poin_curvs = [], []
    for i in range(1, T - 1):
        euc_curvs.append(menger_curvature_euclidean(normalized[i-1], normalized[i], normalized[i+1]))
        poin_curvs.append(menger_curvature_poincare(poincare_emb[i-1], poincare_emb[i], poincare_emb[i+1], c=c))

    mean_euc = float(np.mean(euc_curvs)) if euc_curvs else 0.0
    mean_poin = float(np.mean(poin_curvs)) if poin_curvs else 0.0
    reduction_pct = 100.0 * (1.0 - mean_poin / (mean_euc + 1e-9)) if mean_euc > 0 else 0.0
    return {"euc": mean_euc, "poin": mean_poin, "reduction_pct": reduction_pct}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str, device: str, load_in_4bit: bool = False, load_in_8bit: bool = False):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs = {"trust_remote_code": True}

    if load_in_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["device_map"] = "auto"
    elif load_in_8bit:
        kwargs["load_in_8bit"] = True
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = dtype

    try:
        model = AutoModel.from_pretrained(model_id, **kwargs)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if not (load_in_4bit or load_in_8bit):
        model.to(device)
    model.eval()
    return tok, model


# ── Step vector extraction ────────────────────────────────────────────────────

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


def count_params(model) -> str:
    try:
        n = sum(p.numel() for p in model.parameters())
        if n >= 1e9:
            return f"{n/1e9:.1f}B"
        return f"{n/1e6:.0f}M"
    except Exception:
        return "N/A"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Model scale curvature comparison")
    ap.add_argument("--hf_models", required=True, help="Comma-separated model IDs")
    ap.add_argument("--data_file", default="data/demo_subset.json")
    ap.add_argument("--sections", default="all")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--output", default="results/scale_comparison/")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model_ids = [m.strip() for m in args.hf_models.split(",") if m.strip()]

    print("=" * 60)
    print("MODEL SCALE COMPARISON: HYPERBOLIC CURVATURE REDUCTION")
    print("=" * 60)
    print(f"Models: {model_ids}")

    items = load_dataset(args.data_file, args.sections)
    print(f"Dataset: {len(items)} items from {args.data_file}")

    all_results = []

    for mid in model_ids:
        print(f"\n{'='*60}")
        print(f"Processing: {mid}")
        print(f"{'='*60}")

        tokenizer, model = load_model(mid, args.device, args.load_in_4bit, args.load_in_8bit)
        param_count = count_params(model)
        print(f"Parameters: {param_count}")

        euc_vals, poin_vals, reduction_vals = [], [], []

        for idx, item in enumerate(items):
            print(f"  [{idx+1}/{len(items)}] {item.logic}:{item.topic or 'abstract'}")
            emb = get_step_vectors(tokenizer, model, item.steps, args.device)
            result = analyze_trajectory_both(emb)
            euc_vals.append(result["euc"])
            poin_vals.append(result["poin"])
            reduction_vals.append(result["reduction_pct"])

        mean_euc = float(np.mean(euc_vals))
        mean_poin = float(np.mean(poin_vals))
        mean_reduction = float(np.mean(reduction_vals))

        all_results.append({
            "model": mid.split("/")[-1],
            "model_full": mid,
            "param_count": param_count,
            "mean_euclidean": round(mean_euc, 4),
            "mean_poincare": round(mean_poin, 4),
            "reduction_pct": round(mean_reduction, 2),
        })

        print(f"\n  Euclidean curvature: {mean_euc:.4f}")
        print(f"  Poincaré curvature:  {mean_poin:.4f}")
        print(f"  Reduction:           {mean_reduction:.1f}%")

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output, "scale_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    print("\n" + "=" * 60)
    print("SCALE COMPARISON SUMMARY")
    print("=" * 60)
    print(df[["model", "param_count", "mean_euclidean", "mean_poincare", "reduction_pct"]].to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = range(len(df))
    labels = [f"{r['model']}\n({r['param_count']})" for _, r in df.iterrows()]

    axes[0].bar([i - 0.2 for i in x], df["mean_euclidean"], width=0.4,
                color="#E74C3C", label="Euclidean", edgecolor="black")
    axes[0].bar([i + 0.2 for i in x], df["mean_poincare"], width=0.4,
                color="#27AE60", label="Poincaré", edgecolor="black")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Mean Menger Curvature")
    axes[0].set_title("Curvature by Model Scale")
    axes[0].legend()

    axes[1].bar(list(x), df["reduction_pct"], color="#3498DB", edgecolor="black")
    axes[1].axhline(y=41, color="#E74C3C", linestyle="--", label="Qwen2.5-0.5B baseline (41%)")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Curvature Reduction (%)")
    axes[1].set_title("% Reduction: Euclidean → Poincaré")
    axes[1].legend()
    for i, v in enumerate(df["reduction_pct"]):
        axes[1].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")

    plt.suptitle("Hyperbolic Geodesic Hypothesis: Scale Invariance Test", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(args.output, "scale_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
