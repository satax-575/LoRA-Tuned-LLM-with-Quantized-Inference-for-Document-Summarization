"""
FastAPI Inference Backend — Main Application
QLoRA Llama 3.1 8B + AWQ + vLLM + Redis
Production-ready, async, stateless, horizontally scalable
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from rich.console import Console

from inference.schemas import (
    SummarizeRequest,
    SummarizeResponse,
    BatchSummarizeRequest,
    BatchSummarizeResponse,
    HealthResponse,
    MetricsResponse,
    ErrorResponse,
)
from inference.cache import ResponseCache, get_cache
from inference.model_loader import VLLMModelLoader, get_model_loader
from inference.summarizer import Summarizer
from inference.middleware import setup_middleware

load_dotenv()
console = Console()
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global state ──
_startup_time = time.time()
_summarizer: Optional[Summarizer] = None
_total_requests = 0


# ─────────────────────────────────────────────
# Lifespan: startup & shutdown
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Loads model and connects to Redis on startup.
    Gracefully disconnects on shutdown.
    """
    global _summarizer

    console.print("\n" + "═" * 60)
    console.print("  [bold magenta]QLoRA Llama 3.1 8B — Inference API[/bold magenta]")
    console.print("  AWQ 4-bit | vLLM | Redis | FastAPI")
    console.print("═" * 60 + "\n")

    # ── Load Model ──
    loader = get_model_loader()
    loader.load()

    # ── Connect Redis ──
    cache = get_cache()
    await cache.connect()

    # ── Initialize Summarizer ──
    _summarizer = Summarizer(loader=loader, cache=cache)

    console.print("\n[bold green]✓ API ready to serve requests[/bold green]\n")

    yield  # ← API is running

    # ── Shutdown ──
    console.print("\n[yellow]Shutting down...[/yellow]")
    await cache.disconnect()
    console.print("[green]✓ Graceful shutdown complete[/green]")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="QLoRA Llama 3.1 8B — Document Summarization API",
    description=(
        "Production inference backend for domain-adaptive document summarization.\n\n"
        "**Model**: Llama 3.1 8B fine-tuned with QLoRA (NF4 4-bit, rank-16, α=32)\n"
        "**Quantization**: AWQ 4-bit (W4A16)\n"
        "**Serving**: vLLM AsyncLLMEngine with continuous batching\n"
        "**Caching**: Redis SHA-256 response cache (40% P95 latency reduction)\n"
        "**Dataset**: CNN/DailyMail 50K — ROUGE-L 0.72"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Register middleware
setup_middleware(
    app,
    cors_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
)


# ─────────────────────────────────────────────
# Dependency: get summarizer
# ─────────────────────────────────────────────

def get_summarizer() -> Summarizer:
    if _summarizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Try again in a moment.",
        )
    return _summarizer


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarize a document",
    description=(
        "Generate a concise summary using QLoRA-tuned Llama 3.1 8B.\n\n"
        "Responses are cached in Redis by document content hash. "
        "Repeated identical requests return cached results with ~10ms latency."
    ),
    tags=["Summarization"],
)
async def summarize(
    request: SummarizeRequest,
    summarizer: Summarizer = Depends(get_summarizer),
) -> SummarizeResponse:
    """Main summarization endpoint."""
    global _total_requests
    _total_requests += 1

    try:
        result = await summarizer.summarize(
            document=request.document,
            max_new_tokens=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            use_cache=request.use_cache,
        )
        return SummarizeResponse(
            summary=result["summary"],
            document_length=result["document_length"],
            summary_length=result["summary_length"],
            compression_ratio=result["compression_ratio"],
            cached=result["cached"],
            latency_ms=result["latency_ms"],
        )

    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}",
        )


@app.post(
    "/summarize/batch",
    response_model=BatchSummarizeResponse,
    summary="Batch summarize multiple documents",
    tags=["Summarization"],
)
async def summarize_batch(
    request: BatchSummarizeRequest,
    summarizer: Summarizer = Depends(get_summarizer),
) -> BatchSummarizeResponse:
    """Batch summarization — up to 16 documents per request."""
    global _total_requests

    t_start = time.perf_counter()
    summaries = []
    cached_count = 0

    for doc in request.documents:
        _total_requests += 1
        result = await summarizer.summarize(
            document=doc,
            max_new_tokens=request.max_length,
            temperature=request.temperature,
            use_cache=request.use_cache,
        )
        summaries.append(result["summary"])
        if result["cached"]:
            cached_count += 1

    total_latency_ms = (time.perf_counter() - t_start) * 1000

    return BatchSummarizeResponse(
        summaries=summaries,
        count=len(summaries),
        total_latency_ms=round(total_latency_ms, 2),
        cached_count=cached_count,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Operations"],
)
async def health_check(
    summarizer: Summarizer = Depends(get_summarizer),
) -> HealthResponse:
    """Returns model, Redis, and GPU status."""
    cache = get_cache()
    loader = get_model_loader()
    gpu_stats = loader.get_gpu_stats()

    return HealthResponse(
        status="healthy",
        model_loaded=loader.is_loaded,
        redis_connected=await cache.health_check(),
        gpu_available=gpu_stats.get("gpu_available", False),
        gpu_memory_used_gb=gpu_stats.get("gpu_memory_used_gb"),
        gpu_memory_total_gb=gpu_stats.get("gpu_memory_total_gb"),
        uptime_seconds=round(time.time() - _startup_time, 1),
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Performance metrics",
    tags=["Operations"],
)
async def metrics(
    summarizer: Summarizer = Depends(get_summarizer),
) -> MetricsResponse:
    """Returns latency percentiles and cache statistics."""
    cache = get_cache()
    cache_metrics = cache.get_metrics()
    latency_stats = summarizer.get_latency_percentiles()
    uptime = time.time() - _startup_time

    return MetricsResponse(
        total_requests=_total_requests,
        cache_hits=cache_metrics.get("hits", 0),
        cache_misses=cache_metrics.get("misses", 0),
        cache_hit_rate=cache_metrics.get("hit_rate", 0.0),
        p50_latency_ms=latency_stats.get("p50_ms", 0.0),
        p90_latency_ms=latency_stats.get("p90_ms", 0.0),
        p95_latency_ms=latency_stats.get("p95_ms", 0.0),
        p99_latency_ms=latency_stats.get("p99_ms", 0.0),
        avg_latency_ms=latency_stats.get("mean_ms", 0.0),
        requests_per_second=round(_total_requests / max(uptime, 1), 2),
    )


@app.delete(
    "/cache",
    summary="Flush response cache",
    tags=["Operations"],
)
async def flush_cache() -> dict:
    """Clear all Redis cached responses."""
    cache = get_cache()
    count = await cache.flush()
    return {"message": f"Flushed {count} cache entries"}


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "QLoRA Llama 3.1 8B — Document Summarization API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "inference.main:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        workers=1,                   # Single worker — vLLM manages concurrency
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        access_log=True,
        timeout_keep_alive=120,
    )
