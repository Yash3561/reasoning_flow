#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_generalization.sh
#
# Runs all generalization experiments:
#   1. CounterfactQA cross-dataset validation
#   2. Implicit lying: salesperson, self-preservation, competitive
#
# Both on Llama and Qwen.
# Total time: ~2-3 hours on A100
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

LLAMA="meta-llama/Meta-Llama-3.1-8B-Instruct"
QWEN="Qwen/Qwen2.5-7B-Instruct"
DEVICE="cuda"

echo "══════════════════════════════════════════════════════════════════"
echo " Generalization Experiments — NJIT OnDemand A100"
echo " $(date)"
echo "══════════════════════════════════════════════════════════════════"

# ── Llama ─────────────────────────────────────────────────────────────────────
echo ""
echo "── [1/4] CounterfactQA — Llama ─────────────────────────────────────────"
python experiments/eval_counterfactqa.py \
    --model  "$LLAMA" \
    --probe  results/detector_200/llama/probe.pkl \
    --out    results/generalization/llama_counterfactqa \
    --device $DEVICE

echo ""
echo "── [2/4] Implicit Lying — Llama ─────────────────────────────────────────"
python experiments/eval_implicit_lying.py \
    --model  "$LLAMA" \
    --probe  results/detector_200/llama/probe.pkl \
    --out    results/generalization/llama_implicit \
    --device $DEVICE

# ── Qwen ──────────────────────────────────────────────────────────────────────
echo ""
echo "── [3/4] CounterfactQA — Qwen ──────────────────────────────────────────"
python experiments/eval_counterfactqa.py \
    --model  "$QWEN" \
    --probe  results/detector_200/qwen/probe.pkl \
    --out    results/generalization/qwen_counterfactqa \
    --device $DEVICE

echo ""
echo "── [4/4] Implicit Lying — Qwen ─────────────────────────────────────────"
python experiments/eval_implicit_lying.py \
    --model  "$QWEN" \
    --probe  results/detector_200/qwen/probe.pkl \
    --out    results/generalization/qwen_implicit \
    --device $DEVICE

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo " All generalization experiments complete! $(date)"
echo ""
echo " Key outputs:"
echo "   results/generalization/llama_counterfactqa/counterfactqa_results.json"
echo "   results/generalization/llama_implicit/implicit_lying_results.json"
echo "   results/generalization/qwen_counterfactqa/counterfactqa_results.json"
echo "   results/generalization/qwen_implicit/implicit_lying_results.json"
echo "══════════════════════════════════════════════════════════════════"