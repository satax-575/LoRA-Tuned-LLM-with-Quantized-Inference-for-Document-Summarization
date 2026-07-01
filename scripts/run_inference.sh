#!/bin/bash
# ============================================================
# run_inference.sh — Inference Backend + Quantization
# Usage: bash scripts/run_inference.sh [--quantize] [--docker]
# ============================================================

set -e

if [ -f ".env" ]; then source .env; fi

QUANTIZE=false
USE_DOCKER=false
PORT=${API_PORT:-8000}
MODEL_PATH=${AWQ_MODEL_DIR:-"./outputs/awq_model"}

for arg in "$@"; do
  case $arg in
    --quantize) QUANTIZE=true ;;
    --docker)   USE_DOCKER=true ;;
    --port=*)   PORT="${arg#*=}" ;;
  esac
done

echo "╔══════════════════════════════════════════════════════╗"
echo "  QLoRA Llama 3.1 8B — Inference Stack"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: AWQ Quantization (optional)
if [ "$QUANTIZE" = true ]; then
  echo "── Step 1: AWQ Quantization ──"
  python quantization/awq_quantize.py \
    --model-path ./outputs/merged_model \
    --output-path "$MODEL_PATH" \
    --calib-samples 128 \
    --verify
  echo "✓ AWQ model saved to $MODEL_PATH"
  echo ""
fi

# ── Step 2: Start Redis
if ! USE_DOCKER; then
  echo "── Step 2: Starting Redis ──"
  if ! redis-cli ping &>/dev/null 2>&1; then
    echo "  Starting Redis..."
    redis-server --daemonize yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    sleep 1
  fi
  echo "  Redis: $(redis-cli ping)"
  echo ""
fi

# ── Step 3: Start API
if [ "$USE_DOCKER" = true ]; then
  echo "── Docker Compose ──"
  cd docker
  docker compose up --build -d
  echo "✓ Services started"
  echo "  API: http://localhost:$PORT"
  echo "  Docs: http://localhost:$PORT/docs"
else
  echo "── Step 3: Starting FastAPI Server ──"
  echo "  Model: $MODEL_PATH"
  echo "  Port:  $PORT"
  echo "  Docs:  http://localhost:$PORT/docs"
  echo ""
  uvicorn inference.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --log-level info \
    --timeout-keep-alive 120
fi
