# ============================================================
# Dockerfile — QLoRA Llama 3.1 8B Inference Backend
# CUDA 12.1 + PyTorch + vLLM + FastAPI
# ============================================================

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# ── System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── Python alias
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ── Working directory
WORKDIR /app

# ── Copy requirements first (Docker layer cache)
COPY requirements.txt .

# ── Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir vllm==0.5.4 && \
    pip install --no-cache-dir \
        transformers>=4.44.0 \
        peft>=0.12.0 \
        bitsandbytes>=0.43.0 \
        autoawq>=0.2.5 \
        fastapi>=0.111.0 \
        uvicorn[standard]>=0.30.0 \
        redis>=5.0.7 \
        hiredis>=2.3.2 \
        rouge-score>=0.1.2 \
        python-dotenv>=1.0.1 \
        pyyaml>=6.0.1 \
        rich>=13.7.0 \
        numpy>=1.26.0 \
        pydantic>=2.7.0

# ── Copy source code
COPY . .

# ── Environment variables (override at runtime)
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV LOG_LEVEL=info
ENV REDIS_URL=redis://redis:6379

# ── Expose port
EXPOSE 8000

# ── Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint
CMD ["uvicorn", "inference.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--timeout-keep-alive", "120"]
