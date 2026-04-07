#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_deception_pipeline.sh
#
# Full pipeline: dataset → hiddens (Qwen) → hiddens (Llama) → deviation →
#                steering (Qwen) → steering (Llama)
#
# Run this from the root of the reasoning_flow repo on NJIT OnDemand.
# Assumes CUDA is available (A100 80GB).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

QWEN_MODEL="Qwen/Qwen2.5-7B-Instruct"
LLAMA_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE="cuda"
MAX_TOKENS=200
N_STEPS=9
STEER_LAYER=12    # CMU paper identified layers 10-15 as critical

echo "══════════════════════════════════════════════════════════════════"
echo " Deception Geometry Pipeline — NJIT OnDemand A100"
echo " $(date)"
echo "══════════════════════════════════════════════════════════════════"

# ── Install dependencies ──────────────────────────────────────────────────────
echo ""
echo "── [0/6] Installing dependencies ───────────────────────────────────────"
pip install -q geoopt datasets matplotlib scipy pandas --break-system-packages
pip install -q torch transformers accelerate --break-system-packages

# ── Step 1: Generate dataset ──────────────────────────────────────────────────
echo ""
echo "── [1/6] Generating contrastive truth/lie dataset ───────────────────────"
python experiments/generate_deception_dataset.py
echo "Dataset ready: data/deception_pairs.json"

# ── Step 2: Extract hidden states — Qwen ─────────────────────────────────────
echo ""
echo "── [2/6] Extracting hidden states — Qwen2.5-7B ──────────────────────────"
python experiments/extract_deception_hiddens.py \
    --model  "$QWEN_MODEL" \
    --data   data/deception_pairs.json \
    --out    results/deception_hiddens/qwen \
    --max_new_tokens $MAX_TOKENS \
    --n_steps $N_STEPS \
    --device $DEVICE

# ── Step 3: Extract hidden states — Llama ────────────────────────────────────
echo ""
echo "── [3/6] Extracting hidden states — Llama-3.1-8B ────────────────────────"
python experiments/extract_deception_hiddens.py \
    --model  "$LLAMA_MODEL" \
    --data   data/deception_pairs.json \
    --out    results/deception_hiddens/llama \
    --max_new_tokens $MAX_TOKENS \
    --n_steps $N_STEPS \
    --device $DEVICE

# ── Step 4: Geodesic deviation analysis ──────────────────────────────────────
echo ""
echo "── [4/6] Geodesic deviation: truth vs. lie curvature ────────────────────"

echo "  Qwen..."
python experiments/geodesic_deviation.py \
    --hiddens results/deception_hiddens/qwen \
    --out     results/geodesic_deviation/qwen

echo "  Llama..."
python experiments/geodesic_deviation.py \
    --hiddens results/deception_hiddens/llama \
    --out     results/geodesic_deviation/llama

# ── Step 5: Hyperbolic steering — Qwen ───────────────────────────────────────
echo ""
echo "── [5/6] Hyperbolic steering — Qwen2.5-7B ───────────────────────────────"
python experiments/hyperbolic_steering.py \
    --model       "$QWEN_MODEL" \
    --hiddens     results/deception_hiddens/qwen \
    --out         results/steering/qwen \
    --lambdas     -0.2 -0.1 -0.05 -0.01 0.0 0.01 0.05 0.1 0.2 \
    --steer_layer $STEER_LAYER \
    --device      $DEVICE

# ── Step 6: Hyperbolic steering — Llama ──────────────────────────────────────
echo ""
echo "── [6/6] Hyperbolic steering — Llama-3.1-8B ─────────────────────────────"
python experiments/hyperbolic_steering.py \
    --model       "$LLAMA_MODEL" \
    --hiddens     results/deception_hiddens/llama \
    --out         results/steering/llama \
    --lambdas     -0.2 -0.1 -0.05 -0.01 0.0 0.01 0.05 0.1 0.2 \
    --steer_layer $STEER_LAYER \
    --device      $DEVICE

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo " Pipeline complete! $(date)"
echo ""
echo " Key outputs:"
echo "   results/geodesic_deviation/qwen/summary.json"
echo "   results/geodesic_deviation/llama/summary.json"
echo "   results/geodesic_deviation/qwen/deviation_plot.png"
echo "   results/steering/qwen/steering_results.json"
echo "   results/steering/llama/steering_results.json"
echo "   results/steering/qwen/steering_comparison.png"
echo "══════════════════════════════════════════════════════════════════"