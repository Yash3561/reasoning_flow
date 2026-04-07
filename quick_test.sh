#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# quick_test.sh
#
# Smoke test: runs the full pipeline on just 5 pairs with a tiny model
# to verify setup before committing to the full A100 run.
# Should complete in ~5 minutes.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SMALL_MODEL="Qwen/Qwen2.5-0.5B-Instruct"   # Already in your repo cache
DEVICE="cuda"

echo "── Smoke test with $SMALL_MODEL (5 pairs) ────────────────────────────────"

# Step 1: dataset
python experiments/generate_deception_dataset.py

# Step 2: hiddens (only 5 pairs, 50 tokens)
python experiments/extract_deception_hiddens.py \
    --model  "$SMALL_MODEL" \
    --data   data/deception_pairs.json \
    --out    results/deception_hiddens/smoke_test \
    --max_new_tokens 50 \
    --n_steps 9 \
    --limit  5 \
    --device $DEVICE

# Step 3: deviation (fast, pure numpy)
python experiments/geodesic_deviation.py \
    --hiddens results/deception_hiddens/smoke_test \
    --out     results/geodesic_deviation/smoke_test

# Step 4: steering (5 λ values only)
python experiments/hyperbolic_steering.py \
    --model       "$SMALL_MODEL" \
    --hiddens     results/deception_hiddens/smoke_test \
    --out         results/steering/smoke_test \
    --lambdas     -0.1 -0.05 -0.01 0.0 0.01 0.05 0.1 \
    --steer_layer 8 \
    --device      $DEVICE

echo ""
echo "── Smoke test passed! Ready for full pipeline. ──────────────────────────"
echo "   Run: bash run_deception_pipeline.sh"