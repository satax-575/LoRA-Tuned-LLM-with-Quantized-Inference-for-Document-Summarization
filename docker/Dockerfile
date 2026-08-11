# ============================================================
# Dockerfile — QLoRA Llama 3.1 8B Inference Backend
# CUDA 12.1 + PyTorch + vLLM + FastAPI
#
# Changes vs original:
#   - Multi-stage build (builder + runtime) to reduce image size
#   - Non-root appuser for security
#   - pip --timeout=300 to prevent hangs on slow networks
#   - Ordered installs (torch → vllm → other) to avoid version conflicts
#   - logs/ directory pre-created with correct permissions
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python alias
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /build

# Copy requirements first (Docker layer cache)
COPY requirements.txt .

# Install Python packages in dependency order to avoid conflicts
# 1. PyTorch (CUDA 12.1)
RUN pip install --no-cache-dir --timeout=300 --upgrade pip && \
    pip install --no-cache-dir --timeout=300 \
        torch==2.3.0+cu121 \
        torchvision==0.18.0+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

# 2. vLLM (depends on specific torch version)
RUN pip install --no-cache-dir --timeout=300 vllm==0.5.4

# 3. Core HF ecosystem
RUN pip install --no-cache-dir --timeout=300 \
    transformers>=4.44.0 \
    peft>=0.12.0 \
    bitsandbytes>=0.43.0 \
    accelerate>=0.33.0 \
    datasets>=2.20.0 \
    huggingface_hub>=0.24.0 \
    tokenizers>=0.19.0

# 4. AWQ
RUN pip install --no-cache-dir --timeout=300 \
    autoawq>=0.2.5 \
    autoawq-kernels>=0.0.6

# 5. FastAPI + serving
RUN pip install --no-cache-dir --timeout=300 \
    fastapi>=0.111.0 \
    uvicorn[standard]>=0.30.0 \
    python-multipart>=0.0.9 \
    httpx>=0.27.0 \
    redis>=5.0.7 \
    hiredis>=2.3.2 \
    slowapi>=0.1.9 \
    prometheus-client>=0.20.0

# 6. Utilities
RUN pip install --no-cache-dir --timeout=300 \
    rouge-score>=0.1.2 \
    python-dotenv>=1.0.1 \
    pyyaml>=6.0.1 \
    rich>=13.7.0 \
    loguru>=0.7.2 \
    numpy>=1.26.0 \
    pydantic>=2.7.0 \
    pydantic-settings>=2.3.0


# ── Stage 2: Runtime ─────────────────────────────────────────
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

# System runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy source code
COPY --chown=appuser:appuser . .

# Create logs directory and set permissions
RUN mkdir -p /app/logs /app/outputs && \
    chown -R appuser:appuser /app/logs /app/outputs

# Switch to non-root user
USER appuser

# Environment variables (override at runtime via docker run -e or .env)
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV LOG_LEVEL=info
ENV REDIS_URL=redis://redis:6379
ENV JSON_LOGS=true

# Expose port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint
CMD ["python", "-m", "uvicorn", "inference.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--timeout-keep-alive", "120"]
