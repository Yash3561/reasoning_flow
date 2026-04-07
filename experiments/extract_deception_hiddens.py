"""
extract_deception_hiddens.py

Extracts hidden states from truth and lie generation trajectories.
Saves per-sample hidden state sequences to results/deception_hiddens/

Works with:
  - Qwen/Qwen2.5-7B-Instruct
  - meta-llama/Meta-Llama-3.1-8B-Instruct

Usage:
  python experiments/extract_deception_hiddens.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --data data/deception_pairs.json \
      --out results/deception_hiddens/qwen \
      --max_new_tokens 200 \
      --device cuda
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          type=str, required=True,
                   help="HuggingFace model id, e.g. Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data",           type=str, default="data/deception_pairs.json")
    p.add_argument("--out",            type=str, required=True,
                   help="Output directory for hidden state numpy arrays")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--device",         type=str, default="cuda")
    p.add_argument("--layer",          type=int, default=-1,
                   help="Which layer to extract. -1 = final layer.")
    p.add_argument("--n_steps",        type=int, default=9,
                   help="Number of CoT steps to sample hidden states from.")
    p.add_argument("--limit",          type=int, default=None,
                   help="Limit number of pairs for quick testing.")
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str, device: str):
    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",          # handles multi-GPU automatically
        trust_remote_code=True,
        output_hidden_states=True,  # CRITICAL: return all layer hiddens
    )
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")
    return tokenizer, model


# ── Hidden state extraction ───────────────────────────────────────────────────

def extract_trajectory(
    messages: list,
    tokenizer,
    model,
    layer: int,
    max_new_tokens: int,
    n_steps: int,
    device: str,
) -> np.ndarray:
    """
    Generate a response token-by-token and collect hidden states at evenly
    spaced token positions. Returns array of shape (n_steps, hidden_dim).

    Why token-level sampling?
    We don't have explicit CoT step boundaries in free-form generation,
    so we sample hidden states at n_steps evenly-spaced token positions
    during generation. This gives a trajectory analogous to the reasoning_flow
    step-level trajectory.
    """
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Generate with hidden states
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,         # greedy — deterministic, reproducible
            temperature=1.0,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    # output.hidden_states: tuple of length (n_generated_tokens,)
    # each element: tuple of length (n_layers + 1,) [embedding + each layer]
    # each layer: tensor of shape (batch, seq_len_so_far, hidden_dim)
    hidden_states_per_step = output.hidden_states  # tuple[tuple[tensor]]

    n_generated = len(hidden_states_per_step)
    if n_generated == 0:
        # Fallback: model refused / empty generation
        hidden_dim = model.config.hidden_size
        return np.zeros((n_steps, hidden_dim), dtype=np.float32)

    # Pick n_steps evenly-spaced generated token positions
    indices = np.linspace(0, n_generated - 1, n_steps, dtype=int)

    trajectory = []
    for idx in indices:
        layer_hiddens = hidden_states_per_step[idx]  # tuple of tensors per layer
        # layer -1 = last transformer layer (before LM head)
        h = layer_hiddens[layer]                     # (batch, seq_at_step, hidden)
        # Take the last token's hidden state (the "current" state)
        h_last = h[0, -1, :].float().cpu().numpy()  # (hidden_dim,)
        trajectory.append(h_last)

    return np.stack(trajectory, axis=0)  # (n_steps, hidden_dim)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load data
    with open(args.data) as f:
        pairs = json.load(f)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"Processing {len(pairs)} pairs.")

    # Load model
    tokenizer, model = load_model(args.model, args.device)

    # Determine actual layer index
    n_layers = model.config.num_hidden_layers
    layer_idx = args.layer if args.layer >= 0 else n_layers  # n_layers = last layer
    print(f"Extracting from layer index {layer_idx} / {n_layers}")

    # Output dirs
    out = Path(args.out)
    truth_dir = out / "truth"
    lie_dir   = out / "lie"
    truth_dir.mkdir(parents=True, exist_ok=True)
    lie_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta = {
        "model": args.model,
        "layer": layer_idx,
        "n_steps": args.n_steps,
        "max_new_tokens": args.max_new_tokens,
        "n_pairs": len(pairs),
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Extract
    for i, pair in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {pair['question'][:60]}")

        truth_traj = extract_trajectory(
            pair["truth_messages"], tokenizer, model,
            layer=layer_idx, max_new_tokens=args.max_new_tokens,
            n_steps=args.n_steps, device=args.device,
        )
        lie_traj = extract_trajectory(
            pair["lie_messages"], tokenizer, model,
            layer=layer_idx, max_new_tokens=args.max_new_tokens,
            n_steps=args.n_steps, device=args.device,
        )

        np.save(truth_dir / f"pair_{i:04d}.npy", truth_traj)
        np.save(lie_dir   / f"pair_{i:04d}.npy", lie_traj)

        if (i + 1) % 5 == 0:
            print(f"  truth shape: {truth_traj.shape}, "
                  f"norm range: [{np.linalg.norm(truth_traj, axis=1).min():.1f}, "
                  f"{np.linalg.norm(truth_traj, axis=1).max():.1f}]")

    print(f"\nDone. Hidden states saved to {out}")


if __name__ == "__main__":
    main()
