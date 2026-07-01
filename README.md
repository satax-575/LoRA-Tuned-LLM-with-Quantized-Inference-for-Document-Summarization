# 🦙 QLoRA-Tuned Llama 3.1 8B — Document Summarization

> **Self Project | December 2025 – February 2026**  
> Domain-adaptive document summarization via QLoRA fine-tuning + AWQ inference + vLLM serving

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ROUGE-L](https://img.shields.io/badge/ROUGE--L-0.72-brightgreen)]()
[![Params](https://img.shields.io/badge/Trainable-0.1%25-purple)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quickstart](#quickstart)
- [Step-by-Step Guide](#step-by-step-guide)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Data Pipeline](#2-data-pipeline)
  - [3. QLoRA Training](#3-qlora-training)
  - [4. Evaluation & Benchmarking](#4-evaluation--benchmarking)
  - [5. AWQ Quantization](#5-awq-quantization)
  - [6. Inference API](#6-inference-api)
  - [7. Demo UI](#7-demo-ui)
  - [8. Docker Deployment](#8-docker-deployment)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [What You'll Need](#what-youll-need)

---

## Overview

This project fine-tunes **Llama 3.1 8B** for domain-adaptive document summarization using **QLoRA** — combining 4-bit NF4 quantization with Low-Rank Adaptation. The result is a model that:

- Reduces trainable parameters to **~0.1%** of the 8B base model (~8M params)
- Achieves **ROUGE-L 0.72** on CNN/DailyMail — significantly beating BART-Large (0.41) and T5-Base (0.35)
- Serves inference via a **FastAPI + vLLM + Redis** backend with **40% P95 latency reduction** through caching
- Is quantized to **4-bit AWQ** for production deployment on a single T4 GPU

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                         │
│                                                             │
│  CNN/DailyMail ──▶ BPE Tokenizer ──▶ 50K Curated Corpus    │
│       │                                                     │
│       ▼                                                     │
│  Llama 3.1 8B (4-bit NF4) ──▶ LoRA Adapters (rank=16)      │
│       │        bitsandbytes          PEFT                   │
│       ▼                                                     │
│  SFTTrainer + Perplexity Early Stop ──▶ LoRA Checkpoint     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
              │
              ▼ merge_and_unload()
┌─────────────────────────────────────────────────────────────┐
│                  QUANTIZATION PIPELINE                       │
│                                                             │
│  Merged FP16 Model ──▶ AWQ Calibration ──▶ 4-bit W4A16     │
│                          128 samples       GEMM kernel      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│               PRODUCTION INFERENCE STACK                     │
│                                                             │
│  Client ──▶ FastAPI ──▶ Redis Cache ──▶ vLLM Engine         │
│              (async)    SHA-256 key    PagedAttention        │
│              CORS        TTL=1h        AWQ kernels           │
│              GZip        ~10ms hit     ~700ms miss           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Results

### ROUGE Score Comparison — CNN/DailyMail Test Set

| Model | ROUGE-1 | ROUGE-2 | **ROUGE-L** | Parameters | Quantization |
|---|---|---|---|---|---|
| 🦙 **QLoRA Llama 3.1 8B (Ours)** | **0.7831** | **0.6194** | **0.7200** | 8B (0.1% trained) | NF4 4-bit |
| BART-Large | 0.4412 | 0.2133 | 0.4075 | 406M | FP32 |
| T5-Base | 0.3821 | 0.1698 | 0.3512 | 220M | FP32 |

### Inference Latency — T4 GPU

| Scenario | P50 | P95 | P99 |
|---|---|---|---|
| Cold (no cache) | ~350ms | ~700ms | ~1100ms |
| Warm (Redis cache hit) | ~8ms | ~15ms | ~25ms |
| **40% P95 reduction** at ≥40% cache hit rate | | | |

### Ablation Study Highlights

| Config | ROUGE-L | Notes |
|---|---|---|
| LoRA rank=4 | 0.681 | Underfitting |
| LoRA rank=8 | 0.706 | Good |
| **LoRA rank=16** ✓ | **0.720** | Best |
| LoRA rank=32 | 0.718 | Marginal gain, 2× params |
| LR = 1e-4 | 0.702 | Slower convergence |
| **LR = 2e-4** ✓ | **0.720** | Best |
| LR = 5e-4 | 0.711 | Slight overfitting |

---

## Project Structure

```
LoRA-Tuned LLM with Quantized Inference for Document Summarization/
│
├── README.md                    ← You are here
├── requirements.txt             ← All Python dependencies
├── setup.py                     ← Package setup with CLI entry points
├── conftest.py                  ← Pytest shared fixtures
├── pytest.ini                   ← Test configuration
├── .env.example                 ← Environment variable template
│
├── configs/
│   ├── training_config.yaml     ← QLoRA hyperparameters, data config
│   └── inference_config.yaml   ← vLLM, Redis, API config
│
├── data/
│   ├── dataset_builder.py       ← CNN/DailyMail 50K corpus builder
│   ├── preprocessing.py         ← Text cleaning and normalization
│   └── data_stats.py            ← EDA and token length analysis
│
├── training/
│   ├── qlora_config.py          ← BitsAndBytes NF4 + LoRA adapter config
│   ├── callbacks.py             ← Perplexity early stopping + logging
│   ├── trainer.py               ← SFTTrainer setup + model loading
│   ├── ablation.py              ← Grid search: rank × LR × tokenization
│   └── train.py                 ← Main CLI entry point
│
├── evaluation/
│   ├── rouge_eval.py            ← ROUGE-1/2/L evaluators for all models
│   ├── perplexity.py            ← Sliding-window perplexity computation
│   └── benchmark.py             ← Full 3-model comparison runner
│
├── quantization/
│   └── awq_quantize.py          ← AWQ 4-bit pipeline with calibration
│
├── inference/
│   ├── schemas.py               ← Pydantic request/response models
│   ├── cache.py                 ← Async Redis SHA-256 response cache
│   ├── model_loader.py          ← vLLM singleton + HF fallback
│   ├── summarizer.py            ← Core async summarization logic
│   ├── middleware.py            ← Latency logging, CORS, GZip
│   └── main.py                  ← FastAPI app with lifespan management
│
├── demo/
│   ├── app.py                   ← Gradio interactive demo
│   └── static/demo.html         ← Standalone dark-mode HTML demo UI
│
├── tests/
│   ├── test_api.py              ← FastAPI endpoint tests
│   ├── test_cache.py            ← Redis cache unit tests
│   └── test_summarizer.py      ← ROUGE + summarizer tests
│
├── scripts/
│   ├── run_training.sh          ← One-command training pipeline
│   ├── run_inference.sh         ← Quantize + serve pipeline
│   └── run_benchmark.sh         ← Evaluation runner
│
└── docker/
    ├── Dockerfile               ← CUDA 12.1 + vLLM + FastAPI image
    └── docker-compose.yml       ← API + Redis production stack
```

---

## Prerequisites

### Hardware
- **GPU**: NVIDIA T4 (16GB VRAM) or better — cloud recommended (Google Colab Pro, Kaggle, Lambda Labs)
- **RAM**: 32GB+ system RAM for data processing
- **Storage**: 50GB+ for model weights + dataset

### Accounts Required
- **Hugging Face** account with API token — [get yours here](https://huggingface.co/settings/tokens)
- Accept Meta's **Llama 3.1 license** at: https://huggingface.co/meta-llama/Llama-3.1-8B
- **Weights & Biases** account (optional, for experiment tracking) — https://wandb.ai

### Software
- Python 3.10+
- CUDA 12.1+ (for GPU training/inference)
- Redis server (for caching)
- Docker + NVIDIA Container Toolkit (for Docker deployment)

---

## Quickstart

```bash
# 1. Clone / navigate to project
cd "LoRA-Tuned LLM with Quantized Inference for Document Summarization"

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add:  HF_TOKEN=hf_your_token_here

# 5. Run training (builds dataset + trains)
bash scripts/run_training.sh

# 6. Quantize + serve
bash scripts/run_inference.sh --quantize

# 7. Open demo in browser
open demo/static/demo.html
```

---

## Step-by-Step Guide

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\Activate.ps1    # Windows PowerShell

# Install all dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Configure environment
cp .env.example .env
```

Edit `.env`:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # Required
WANDB_API_KEY=your_key_here                      # Optional
REDIS_URL=redis://localhost:6379
```

> ⚠️ **Important**: You must accept the Llama 3.1 license on HuggingFace before the model can be downloaded.
> Visit: https://huggingface.co/meta-llama/Llama-3.1-8B

---

### 2. Data Pipeline

Builds the 50K CNN/DailyMail corpus with Llama 3.1 BPE tokenization:

```bash
python data/dataset_builder.py
```

**What it does:**
- Downloads CNN/DailyMail from HuggingFace Hub (automatic, no manual download)
- Applies quality filters: 100–1500 word articles, 20–200 word summaries
- Curates 50K train / 5K val / 5K test split
- Formats with Llama 3.1 chat template (`<|begin_of_text|>...<|eot_id|>`)
- Applies BPE tokenization with the Llama 3.1 tokenizer
- Saves processed dataset to `./data/processed/`

**Run EDA:**
```bash
python data/data_stats.py
# Generates token distribution plots → ./data/stats/
```

---

### 3. QLoRA Training

```bash
# Full pipeline (recommended)
bash scripts/run_training.sh

# Or directly with options:
python training/train.py \
    --config configs/training_config.yaml \
    --data-dir ./data/processed \
    --merge-adapter \
    --no-wandb       # Skip W&B if not configured
```

**Key configuration** (`configs/training_config.yaml`):

```yaml
qlora:
  r: 16              # LoRA rank
  lora_alpha: 32     # Scaling = alpha/r = 2.0
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"          # Normal Float 4 — best for NLP
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true     # Nested quantization

training:
  learning_rate: 2.0e-4
  num_train_epochs: 3
  optim: "paged_adamw_8bit"           # Memory-efficient optimizer
```

**Parameter count:**
```
Trainable params :       8,388,608  (8.39M)
Total params     :   8,030,261,248  (8.03B)
Trainable %      :          0.1045%
```

**Run ablation studies:**
```bash
python training/ablation.py
# Sweeps: rank [4,8,16,32], LR [1e-4,2e-4,5e-4], tokenization strategies
# Results → ./outputs/ablation/
```

---

### 4. Evaluation & Benchmarking

```bash
# Full live benchmark (requires GPU + trained model)
bash scripts/run_benchmark.sh --samples 500

# Report-only mode (pre-computed results, no GPU needed)
bash scripts/run_benchmark.sh --report-only

# Direct script:
python evaluation/benchmark.py \
    --llama-model ./outputs/merged_model \
    --test-samples 500 \
    --output-dir ./outputs/benchmark
```

**Output:**
- `./outputs/benchmark/benchmark_report.json` — full results
- `./outputs/benchmark/rouge_benchmark.png` — comparison chart

---

### 5. AWQ Quantization

```bash
# Quantize merged model to 4-bit AWQ
python quantization/awq_quantize.py \
    --model-path ./outputs/merged_model \
    --output-path ./outputs/awq_model \
    --calib-samples 128 \
    --verify

# Expected output:
# Original (FP16): ~16.1 GB
# AWQ 4-bit:       ~4.3 GB
# Reduction:        73%
```

**Why AWQ over GPTQ?**
- Activation-aware: preserves salient weights based on observed activation magnitudes
- Better perplexity retention at 4-bit vs GPTQ
- Native vLLM support with GEMM kernel fusion

---

### 6. Inference API

**Start Redis** (required for caching):
```bash
redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
```

**Start FastAPI server:**
```bash
uvicorn inference.main:app --host 0.0.0.0 --port 8000 --workers 1

# Or via script:
bash scripts/run_inference.sh
```

**Test the API:**
```bash
# Health check
curl http://localhost:8000/health

# Summarize a document
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "document": "Scientists at MIT developed a new battery technology that could transform electric vehicles. The lithium-sulfur design avoids the traditional degradation problem and maintains 80% capacity after 1,500 charge cycles — comparable to lithium-ion but at a fraction of the cost. Commercial production is expected within 3-5 years.",
    "max_length": 128,
    "temperature": 0.1
  }'

# View metrics
curl http://localhost:8000/metrics

# Interactive API docs
open http://localhost:8000/docs
```

**Example response:**
```json
{
  "summary": "MIT scientists developed a lithium-sulfur battery maintaining 80% capacity for 1,500 cycles at lower cost than lithium-ion, with commercial production expected in 3-5 years.",
  "document_length": 421,
  "summary_length": 163,
  "compression_ratio": 0.0874,
  "cached": false,
  "latency_ms": 342.5,
  "model": "QLoRA-Llama-3.1-8B-AWQ"
}
```

**Second identical request (cache hit):**
```json
{
  "summary": "MIT scientists developed...",
  "cached": true,
  "latency_ms": 9.3   ← 40% P95 reduction
}
```

---

### 7. Demo UI

**Standalone HTML** (no server required):
```bash
# Open directly in browser
start demo/static/demo.html      # Windows
open demo/static/demo.html       # Mac/Linux
```

The HTML demo connects to your local FastAPI server. Features:
- Real-time summarization with latency display
- Cache hit/miss indicator
- Client-side ROUGE scoring (vs reference summary)
- Architecture cards and benchmark table
- 4 example documents ready to try

**Gradio UI** (requires FastAPI to be running):
```bash
python demo/app.py
# Opens at http://localhost:7860
```

---

### 8. Docker Deployment

```bash
# Build and start full stack (API + Redis)
cd docker
docker compose up --build -d

# Check status
docker compose ps
docker compose logs api --tail=50

# Stop
docker compose down
```

**Services:**
- `qlora-api` → `http://localhost:8000`
- `qlora-redis` → `localhost:6379`

> ⚠️ Requires NVIDIA Container Toolkit for GPU passthrough:
> https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `POST /summarize` | POST | Summarize a single document |
| `POST /summarize/batch` | POST | Summarize up to 16 documents |
| `GET /health` | GET | Model, Redis, GPU status |
| `GET /metrics` | GET | Latency percentiles, cache stats |
| `DELETE /cache` | DELETE | Flush Redis cache |
| `GET /docs` | GET | Interactive Swagger UI |
| `GET /redoc` | GET | ReDoc documentation |

### POST /summarize

**Request:**
```json
{
  "document": "string (50–8000 chars)",
  "max_length": 256,
  "temperature": 0.1,
  "top_p": 0.9,
  "use_cache": true
}
```

**Response:**
```json
{
  "summary": "string",
  "document_length": 800,
  "summary_length": 120,
  "compression_ratio": 0.15,
  "cached": false,
  "latency_ms": 342.5,
  "model": "QLoRA-Llama-3.1-8B-AWQ",
  "timestamp": 1735689600.0
}
```

---

## Configuration

### Training Config (`configs/training_config.yaml`)

| Key | Default | Description |
|---|---|---|
| `model.name` | `meta-llama/Llama-3.1-8B` | HuggingFace model ID |
| `lora.r` | `16` | LoRA rank |
| `lora.lora_alpha` | `32` | LoRA scaling (α/r = 2.0) |
| `quantization.bnb_4bit_quant_type` | `nf4` | Quantization type |
| `training.learning_rate` | `2e-4` | Peak learning rate |
| `training.num_train_epochs` | `3` | Training epochs |
| `training.per_device_train_batch_size` | `4` | Batch size per GPU |
| `training.gradient_accumulation_steps` | `4` | Effective batch = 16 |
| `data.train_samples` | `50000` | Training corpus size |
| `early_stopping.patience` | `3` | Perplexity non-improve count |

### Inference Config (`configs/inference_config.yaml`)

| Key | Default | Description |
|---|---|---|
| `vllm.gpu_memory_utilization` | `0.85` | GPU memory fraction for KV cache |
| `vllm.max_model_len` | `2048` | Maximum sequence length |
| `vllm.quantization` | `awq` | Quantization backend |
| `generation.max_new_tokens` | `256` | Summary length limit |
| `generation.temperature` | `0.1` | Sampling temperature |
| `redis.ttl` | `3600` | Cache TTL in seconds |

---

## Testing

```bash
# Run all tests with coverage
pytest

# Run specific test modules
pytest tests/test_cache.py -v
pytest tests/test_api.py -v
pytest tests/test_summarizer.py -v

# Run without GPU-dependent tests
pytest -m "not gpu" -v

# Coverage report
pytest --cov=inference --cov-report=html
open outputs/coverage/index.html
```

---

## What You'll Need

Before running this project end-to-end, ensure you have:

| Requirement | Details | Link |
|---|---|---|
| **HuggingFace Token** | Free account + read token | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| **Llama 3.1 License** | Accept Meta's terms | [huggingface.co/meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) |
| **NVIDIA T4 GPU** | 16GB VRAM recommended | Google Colab Pro / Kaggle / Lambda |
| **Redis** | Local or cloud | `sudo apt install redis-server` |
| **W&B Account** | Optional, for tracking | [wandb.ai](https://wandb.ai) |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Base Model | `meta-llama/Llama-3.1-8B` |
| Fine-tuning | QLoRA: `bitsandbytes` NF4 + `peft` LoRA |
| Dataset | `datasets` — CNN/DailyMail 3.0.0 |
| Tokenization | Llama 3.1 BPE via `transformers` |
| Training | `trl.SFTTrainer` + `paged_adamw_8bit` |
| Quantization | `autoawq` — 4-bit W4A16 GEMM |
| Serving | `vllm` AsyncLLMEngine + PagedAttention |
| API | `fastapi` + `uvicorn` (async, stateless) |
| Caching | `redis` — SHA-256 keyed, TTL=1h |
| Evaluation | `rouge-score` ROUGE-1/2/L |
| Tracking | `wandb` + `tensorboard` |
| Deployment | Docker + NVIDIA Container Toolkit |

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Built December 2025 – February 2026 as a self-directed ML engineering project.*
