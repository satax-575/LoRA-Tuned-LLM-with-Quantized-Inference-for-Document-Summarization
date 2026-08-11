"""
FastAPI Middleware
Request logging, CORS, latency tracking, GZip compression, rate limiting setup.
"""

import time
import logging
from typing import Callable, List

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
    Adds X-Process-Time-Ms and X-Model response headers.
    Warns on requests exceeding 2 seconds.
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
    """Injects a unique X-Request-ID header for distributed tracing."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        import uuid
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def setup_middleware(app: FastAPI, cors_origins: List[str] = None):
    """
    Register all middleware on the FastAPI app.

    Middleware execution order (last registered = outermost):
      1. GZip              — compress large responses
      2. CORS              — set allowed origins
      3. LatencyLogging    — log request + add latency header
      4. RequestID         — inject tracing header (outermost)

    Note: APIKeyMiddleware is registered separately in main.py
    so it runs after RequestID but before route handlers.
    """
    # GZip compression for large responses (summaries can be verbose)
    app.add_middleware(GZipMiddleware, minimum_size=500)

    # CORS — allow all origins by default (configure for production via CORS_ORIGINS env)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time-Ms", "X-Request-ID", "X-Model"],
    )

    # Latency logging
    app.add_middleware(LatencyLoggingMiddleware)

    # Request ID tracing (outermost — wraps everything)
    app.add_middleware(RequestIDMiddleware)
