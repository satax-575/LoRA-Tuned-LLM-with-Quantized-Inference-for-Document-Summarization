#!/bin/bash
# ============================================================
# run_benchmark.sh — ROUGE Evaluation & Ablation Studies
# Usage: bash scripts/run_benchmark.sh [--report-only] [--ablation]
# ============================================================

set -e

if [ -f ".env" ]; then source .env; fi

REPORT_ONLY=false
RUN_ABLATION=false
TEST_SAMPLES=500
LLAMA_MODEL="./outputs/merged_model"

for arg in "$@"; do
  case $arg in
    --report-only)  REPORT_ONLY=true ;;
    --ablation)     RUN_ABLATION=true ;;
    --samples=*)    TEST_SAMPLES="${arg#*=}" ;;
  esac
done

echo "╔══════════════════════════════════════════════════════╗"
echo "  QLoRA Llama 3.1 8B — Benchmark Suite"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── ROUGE Benchmark
echo "── ROUGE Benchmark (N=$TEST_SAMPLES) ──"
CMD="python evaluation/benchmark.py --test-samples $TEST_SAMPLES --llama-model $LLAMA_MODEL"
[ "$REPORT_ONLY" = true ] && CMD="$CMD --report-only"
$CMD
echo ""

# ── Ablation Study (optional)
if [ "$RUN_ABLATION" = true ]; then
  echo "── Ablation Study ──"
  python training/ablation.py
  echo ""
fi

echo "✓ Benchmark complete!"
echo "  Results: ./outputs/benchmark/"
