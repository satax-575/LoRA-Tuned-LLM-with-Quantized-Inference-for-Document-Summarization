"""
Makefile — QLoRA Llama 3.1 8B Development Workflow
Common tasks: install, train, serve, test, docker, quantize, benchmark
"""

# ── Variables ─────────────────────────────────────────────────────────────────
PYTHON := python
PIP := pip
PYTEST := pytest
UVICORN := uvicorn

# Paths
CONFIG := configs/training_config.yaml
INFERENCE_CONFIG := configs/inference_config.yaml
DATA_DIR := ./data/processed
LORA_DIR := ./outputs/lora_model
MERGED_DIR := ./outputs/merged_model
AWQ_DIR := ./outputs/awq_model

.PHONY: help install train serve demo test test-cov quantize benchmark docker-up docker-down clean

# ── Default ────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║   QLoRA Llama 3.1 8B — Development Makefile               ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  make install      — Install all Python dependencies"
	@echo "  make setup-env    — Copy .env.example → .env"
	@echo "  make train        — Run QLoRA fine-tuning"
	@echo "  make train-quick  — Train with --skip-data-build (reuse existing data)"
	@echo "  make merge        — Merge LoRA adapter into base model"
	@echo "  make quantize     — AWQ 4-bit quantization + verification"
	@echo "  make benchmark    — Run ROUGE benchmark (report mode, no GPU needed)"
	@echo "  make benchmark-live — Run live ROUGE benchmark (needs GPU)"
	@echo "  make serve        — Start FastAPI inference server"
	@echo "  make demo         — Start Gradio demo UI"
	@echo "  make test         — Run test suite"
	@echo "  make test-cov     — Run tests with coverage report"
	@echo "  make docker-up    — Start full Docker stack"
	@echo "  make docker-down  — Stop Docker stack"
	@echo "  make clean        — Remove outputs and cache"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt
	$(PYTHON) setup.py develop

setup-env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ Created .env from .env.example — fill in your API keys"; \
	else \
		echo "⚠ .env already exists — not overwriting"; \
	fi

# ── Training Pipeline ─────────────────────────────────────────────────────────
train:
	$(PYTHON) training/train.py --config $(CONFIG)

train-quick:
	$(PYTHON) training/train.py --config $(CONFIG) --skip-data-build

train-no-wandb:
	$(PYTHON) training/train.py --config $(CONFIG) --no-wandb

merge:
	$(PYTHON) training/train.py --config $(CONFIG) \
		--skip-data-build \
		--merge-adapter \
		--merged-output $(MERGED_DIR)

ablation:
	$(PYTHON) -c "from training.ablation import AblationStudy; AblationStudy(quick_mode=True).run_all()"

# ── Quantization ──────────────────────────────────────────────────────────────
quantize:
	$(PYTHON) quantization/awq_quantize.py \
		--model-path $(MERGED_DIR) \
		--output-path $(AWQ_DIR) \
		--verify

# ── Evaluation ────────────────────────────────────────────────────────────────
benchmark:
	$(PYTHON) evaluation/benchmark.py --report-only

benchmark-live:
	$(PYTHON) evaluation/benchmark.py \
		--llama-model $(LORA_DIR) \
		--test-samples 500

# ── Inference Server ──────────────────────────────────────────────────────────
serve:
	$(UVICORN) inference.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--workers 1 \
		--log-level info \
		--timeout-keep-alive 120

serve-dev:
	$(UVICORN) inference.main:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--log-level debug

demo:
	$(PYTHON) demo/app.py

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v --tb=short

test-cov:
	$(PYTEST) tests/ -v \
		--cov=inference \
		--cov=data \
		--cov=training \
		--cov=evaluation \
		--cov-report=term-missing \
		--cov-report=html:outputs/coverage_report

test-api:
	$(PYTEST) tests/test_api.py -v

test-cache:
	$(PYTEST) tests/test_cache.py -v

test-training:
	$(PYTEST) tests/test_training.py -v

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker-compose -f docker/docker-compose.yml up -d --build

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f api

docker-redis:
	docker-compose -f docker/docker-compose.yml up -d redis

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf outputs/benchmark outputs/ablation outputs/coverage_report 2>/dev/null || true

clean-all: clean
	rm -rf outputs/ data/processed/ 2>/dev/null || true
