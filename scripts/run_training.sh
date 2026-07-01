#!/bin/bash
# ============================================================
# run_training.sh — QLoRA Training Pipeline
# Usage: bash scripts/run_training.sh [--no-wandb] [--skip-data]
# ============================================================

set -e

# ── Load env
if [ -f ".env" ]; then source .env; fi

# ── Defaults
SKIP_DATA=false
NO_WANDB=false
MERGE_ADAPTER=true
CONFIG="configs/training_config.yaml"
DATA_DIR="./data/processed"

# ── Parse args
for arg in "$@"; do
  case $arg in
    --skip-data)    SKIP_DATA=true ;;
    --no-wandb)     NO_WANDB=true ;;
    --no-merge)     MERGE_ADAPTER=false ;;
    --config=*)     CONFIG="${arg#*=}" ;;
  esac
done

echo "╔══════════════════════════════════════════════════════╗"
echo "  QLoRA Llama 3.1 8B — Training Pipeline"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Check HF Token
if [ -z "$HF_TOKEN" ]; then
  echo "⚠  ERROR: HF_TOKEN not set!"
  echo "  1. Copy .env.example → .env"
  echo "  2. Add your token: HF_TOKEN=hf_..."
  echo "  3. Accept Llama 3.1 license: https://huggingface.co/meta-llama/Llama-3.1-8B"
  exit 1
fi

# ── Check GPU
if command -v nvidia-smi &>/dev/null; then
  echo "GPU Status:"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
  echo ""
fi

# ── Build training command
CMD="python training/train.py --config $CONFIG --data-dir $DATA_DIR"
[ "$SKIP_DATA" = true ]  && CMD="$CMD --skip-data-build"
[ "$NO_WANDB" = true ]   && CMD="$CMD --no-wandb"
[ "$MERGE_ADAPTER" = true ] && CMD="$CMD --merge-adapter"

echo "Running: $CMD"
echo ""

# ── Execute
$CMD

echo ""
echo "✓ Training complete!"
echo ""
echo "Next steps:"
echo "  1. Quantize: bash scripts/run_inference.sh --quantize"
echo "  2. Benchmark: bash scripts/run_benchmark.sh"
echo "  3. Serve API: bash scripts/run_inference.sh"
