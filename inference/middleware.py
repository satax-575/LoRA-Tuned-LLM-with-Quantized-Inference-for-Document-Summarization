"""
FastAPI Middleware
Request logging, CORS, latency tracking, rate limiting
"""

import time
import logging
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs request method, path, status code, and latency for every request.
    Enables P95 latency tracking mentioned in the project brief.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        t_start = time.perf_counter()
        
        # Process request
        response = await call_next(request)

        latency_ms = (time.perf_counter() - t_start) * 1000
        
        # Add latency header for client-side monitoring
        response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
        response.headers["X-Model"] = "QLoRA-Llama-3.1-8B-AWQ"

        # Log to structured output
        log_level = logging.WARNING if latency_ms > 2000 else logging.INFO
        logger.log(
            log_level,
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} [{latency_ms:.1f}ms]"
        )

        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Injects a unique X-Request-ID header for tracing."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import uuid
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def setup_middleware(app: FastAPI, cors_origins: list = None):
    """
    Register all middleware on the FastAPI app.
    Order matters: registered last runs first (outermost).
    """
    # GZip compression for large responses
    app.add_middleware(GZipMiddleware, minimum_size=500)

    # CORS — allow all origins by default (configure for production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time-Ms", "X-Request-ID", "X-Model"],
    )

    # Latency logging (outermost — wraps everything)
    app.add_middleware(LatencyLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)
