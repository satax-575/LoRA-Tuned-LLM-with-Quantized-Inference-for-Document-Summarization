"""
Pydantic Schemas for the FastAPI Inference Backend
Request/response models with validation
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
import time


# ─────────────────────────────────────────────
# Request Schemas
# ─────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    """Request schema for the /summarize endpoint."""

    document: str = Field(
        ...,
        description="The document text to summarize",
        min_length=50,
        max_length=8000,
        examples=["Climate scientists have issued a stark warning that global temperatures..."],
    )
    max_length: int = Field(
        default=256,
        ge=50,
        le=512,
        description="Maximum length of the generated summary in tokens",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.01,
        le=2.0,
        description="Sampling temperature (lower = more focused)",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Nucleus sampling probability",
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use Redis response caching",
    )

    @field_validator("document")
    @classmethod
    def clean_document(cls, v: str) -> str:
        return v.strip()


class BatchSummarizeRequest(BaseModel):
    """Request schema for the /summarize/batch endpoint."""

    documents: List[str] = Field(
        ...,
        description="List of documents to summarize",
        min_length=1,
        max_length=16,
    )
    max_length: int = Field(default=256, ge=50, le=512)
    temperature: float = Field(default=0.1, ge=0.01, le=2.0)
    use_cache: bool = Field(default=True)


# ─────────────────────────────────────────────
# Response Schemas
# ─────────────────────────────────────────────

class SummarizeResponse(BaseModel):
    """Response schema for the /summarize endpoint."""

    summary: str = Field(..., description="The generated summary")
    document_length: int = Field(..., description="Input document character count")
    summary_length: int = Field(..., description="Generated summary character count")
    compression_ratio: float = Field(..., description="Summary / Document length ratio")
    cached: bool = Field(..., description="Whether response was served from Redis cache")
    latency_ms: float = Field(..., description="Total request latency in milliseconds")
    model: str = Field(default="QLoRA-Llama-3.1-8B-AWQ", description="Model identifier")
    timestamp: float = Field(default_factory=time.time)


class BatchSummarizeResponse(BaseModel):
    """Response schema for batch summarization."""

    summaries: List[str]
    count: int
    total_latency_ms: float
    cached_count: int
    model: str = "QLoRA-Llama-3.1-8B-AWQ"
    timestamp: float = Field(default_factory=time.time)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool
    redis_connected: bool
    gpu_available: bool
    gpu_memory_used_gb: Optional[float]
    gpu_memory_total_gb: Optional[float]
    uptime_seconds: float
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    """Latency and throughput metrics."""

    total_requests: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_latency_ms: float
    requests_per_second: float


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: Optional[str] = None
    status_code: int
    timestamp: float = Field(default_factory=time.time)
