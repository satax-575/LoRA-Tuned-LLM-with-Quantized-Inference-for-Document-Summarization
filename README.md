# QLoRA-Tuned Llama 3.1 8B — Document Summarization

> **Self Project · December 2025 – February 2026**

I wanted to understand what it actually takes to fine-tune an 8B-parameter LLM on a single consumer GPU — from raw data all the way to a production-grade inference API. This project does the full thing: QLoRA training on 50K CNN/DailyMail articles, AWQ 4-bit quantization, and a vLLM-powered FastAPI backend with Redis caching that hits 40% lower P95 latency on repeated documents.

The model trains only 0.1% of its parameters (8M out of 8B) and still beats BART-Large on ROUGE-L by a significant margin.

[![ROUGE-L](https://img.shields.io/badge/ROUGE--L-0.72-brightgreen)]()
[![Params](https://img.shields.io/badge/Trainable-0.1%25-purple)]()
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)

---

## How It Works

Three separate pipelines that feed into each other:

```
TRAINING
CNN/DailyMail (50K articles)
    → BPE tokenization with Llama 3.1 chat template
    → Llama 3.1 8B loaded in 4-bit NF4 (bitsandbytes)
    → LoRA adapters injected at rank=16 into all attention + MLP layers
    → SFTTrainer + paged_adamw_8bit + perplexity early stopping
    → LoRA checkpoint saved

QUANTIZATION
Trained model merged to FP16
    → AWQ calibration on 128 samples (activation-aware weight selection)
    → 4-bit W4A16 GEMM quantized model (~4.3 GB vs ~16.1 GB FP16)

INFERENCE
Client → FastAPI → Redis (SHA-256 cache, TTL=1h)
    → vLLM AsyncLLMEngine (PagedAttention + AWQ kernels)
    → ~700ms cold, ~10ms warm cache hit
```

---

## Results

### vs. Baseline Models — CNN/DailyMail Test Set

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Parameters |
|---|---|---|---|---|
| **QLoRA Llama 3.1 8B (this project)** | **0.783** | **0.619** | **0.720** | 8B (0.1% trained) |
| BART-Large | 0.441 | 0.213 | 0.408 | 406M |
| T5-Base | 0.382 | 0.170 | 0.351 | 220M |

### Inference Latency — T4 GPU

| Scenario | P50 | P95 | P99 |
|---|---|---|---|
| Cold (no cache) | ~350ms | ~700ms | ~1100ms |
| Warm (Redis hit) | ~8ms | ~15ms | ~25ms |

At ≥40% cache hit rate, P95 drops by ~40%.

### What the ablation showed

LoRA rank=16 at LR=2e-4 is the sweet spot. Rank=32 gives almost no improvement for double the trainable parameters. Going below rank=8 starts underfitting noticeably.

| Config | ROUGE-L |
|---|---|
| rank=4 | 0.681 |
| rank=8 | 0.706 |
| **rank=16 (used)** | **0.720** |
| rank=32 | 0.718 |

---

## Project Structure

```
LoRA-Tuned LLM/
│
├── configs/
│   ├── training_config.yaml     QLoRA hyperparameters, data config
│   └── inference_config.yaml   vLLM, Redis, API settings
│
├── data/
│   ├── dataset_builder.py       CNN/DailyMail 50K corpus builder
│   ├── preprocessing.py         Text cleaning and normalization
│   └── data_stats.py            Token distribution EDA
│
├── training/
│   ├── qlora_config.py          BitsAndBytes NF4 + LoRA config
│   ├── callbacks.py             Perplexity early stopping + logging
│   ├── trainer.py               SFTTrainer with Flash Attention 2
│   ├── ablation.py              Grid search: rank × LR × tokenization
│   └── train.py                 CLI entry point
│
├── evaluation/
│   ├── rouge_eval.py            ROUGE-1/2/L across all models
│   ├── perplexity.py            Sliding-window perplexity
│   └── benchmark.py             3-model comparison runner
│
├── quantization/
│   └── awq_quantize.py          AWQ 4-bit pipeline
│
├── inference/
│   ├── main.py                  FastAPI app
│   ├── model_loader.py          vLLM AsyncLLMEngine singleton
│   ├── summarizer.py            Core async summarization logic
│   ├── cache.py                 Redis SHA-256 response cache
│   ├── schemas.py               Pydantic request/response models
│   ├── streaming.py             SSE token streaming
│   ├── middleware.py            Latency logging, CORS, GZip
│   ├── auth.py                  API key middleware
│   └── metrics_exporter.py      Prometheus metrics
│
├── demo/
│   ├── app.py                   Gradio UI
│   └── static/demo.html         Standalone dark-mode HTML demo
│
├── tests/
│   ├── test_api.py
│   ├── test_cache.py
│   └── test_summarizer.py
│
├── docker/
│   ├── Dockerfile               CUDA 12.1 + vLLM + FastAPI
│   └── docker-compose.yml       API + Redis stack
│
├── conftest.py
├── pytest.ini
├── requirements.txt
└── .env.example
```

---

## Setup

### Requirements

- **Python 3.11**
- **Linux** — vLLM does not support Windows natively (use WSL2 or a cloud GPU)
- **NVIDIA GPU with 16GB+ VRAM** — T4 or better for both training and inference
- **Redis** — required for the inference API
- **Hugging Face account** with a read token and Llama 3.1 license accepted
- **Weights & Biases account** — required for training tracking (use `--no-wandb` to skip)

### Install

```bash
cd "LoRA-Tuned LLM with Quantized Inference for Document Summarization"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Copy and fill in environment file
cp .env.example .env
```

### Environment Variables

```bash
# .env — copy from .env.example and fill in

HF_TOKEN=hf_xxxxxxxxxxxxxxxx          # Required — Hugging Face token
WANDB_API_KEY=your_wandb_key          # Required for training (or use --no-wandb)
REDIS_URL=redis://localhost:6379       # Required for inference API
API_KEY=your_secret_key_here          # Required for inference API auth
AWQ_MODEL_DIR=./outputs/awq_model     # Path to quantized model
```

> Accept the Llama 3.1 license at https://huggingface.co/meta-llama/Llama-3.1-8B before downloading.

---

## Step-by-Step

### 1. Build the Dataset

Downloads CNN/DailyMail, applies quality filters, formats with Llama chat template, tokenizes:

```bash
python data/dataset_builder.py
# Saves to ./data/processed/
```

### 2. Train

```bash
# Full pipeline
bash scripts/run_training.sh

# With explicit options
python training/train.py \
    --config configs/training_config.yaml \
    --data-dir ./data/processed \
    --merge-adapter \
    --no-wandb       # Only if you want to skip W&B
```

Key hyperparameters in `configs/training_config.yaml`:
- `lora.r: 16` — LoRA rank
- `lora.lora_alpha: 32` — scaling = α/r = 2.0
- `training.learning_rate: 2.0e-4`
- `training.num_train_epochs: 3`
- `training.optim: paged_adamw_8bit`

### 3. Evaluate

```bash
# Full benchmark vs BART and T5 (requires GPU + trained model)
bash scripts/run_benchmark.sh --samples 500

# Report only (pre-computed, no GPU needed)
bash scripts/run_benchmark.sh --report-only
```

### 4. Quantize to AWQ

```bash
python quantization/awq_quantize.py \
    --model-path ./outputs/merged_model \
    --output-path ./outputs/awq_model \
    --calib-samples 128 \
    --verify

# Expected: FP16 ~16.1 GB → AWQ 4-bit ~4.3 GB (73% reduction)
```

### 5. Run the Inference API

Start Redis first:
```bash
redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
```

Start the API:
```bash
uvicorn inference.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Test it:
```bash
# Health check
curl http://localhost:8000/health

# Summarize a document
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your_secret_key_here" \
  -d '{
    "document": "Scientists at MIT developed a new battery technology...",
    "max_length": 128,
    "temperature": 0.1
  }'
```

### 6. Docker Deployment

```bash
cd docker && docker compose up --build -d
# API → http://localhost:8000
# Redis → localhost:6379
```

Requires NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

### 7. Gradio Demo

```bash
python demo/app.py
# Opens at http://localhost:7860
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/summarize` | Summarize a single document |
| `POST` | `/summarize/batch` | Up to 16 documents in one call |
| `GET /stream` | `/summarize/stream` | SSE token streaming |
| `GET` | `/health` | Model, Redis, GPU status |
| `GET` | `/metrics` | Latency percentiles + cache stats |
| `DELETE` | `/cache` | Flush Redis cache |
| `GET` | `/docs` | Swagger UI |

---

## Tests

```bash
pytest
pytest tests/test_cache.py -v
pytest -m "not gpu" -v           # Skip GPU-dependent tests
pytest --cov=inference --cov-report=html
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Base Model | `meta-llama/Llama-3.1-8B` |
| Fine-tuning | `peft` QLoRA + `bitsandbytes` NF4 4-bit |
| Training | `trl.SFTTrainer` + `paged_adamw_8bit` |
| Dataset | `datasets` — CNN/DailyMail 3.0.0 |
| Quantization | `autoawq` — 4-bit W4A16 GEMM |
| Inference | `vllm` AsyncLLMEngine + PagedAttention |
| API | `fastapi` + `uvicorn` |
| Caching | `redis` — SHA-256 keyed, TTL=1h |
| Evaluation | `rouge-score` ROUGE-1/2/L |
| Tracking | `wandb` + `tensorboard` |
| Deployment | Docker + NVIDIA Container Toolkit |
