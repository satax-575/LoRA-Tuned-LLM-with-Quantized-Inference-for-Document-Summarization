"""
API Key Authentication Middleware
Secures all non-public endpoints with X-API-Key header validation.

Public endpoints (no auth required):
  GET  /            — Root info
  GET  /health      — Health check (needed by load balancers)
  GET  /docs        — Swagger UI
  GET  /redoc       — ReDoc UI
  GET  /openapi.json

All other endpoints require:
  Header: X-API-Key: <value of API_KEY env variable>
"""

import os
import logging
from typing import Callable, Set

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Paths that are accessible without an API key
_PUBLIC_PATHS: Set[str] = {
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Validates the X-API-Key header on all protected endpoints.

    Configuration:
        API_KEY env var — the required key value.
        If API_KEY is not set, auth is DISABLED (development mode).

    Usage:
        Set API_KEY=mysecretkeyhere in your .env file.
        Pass X-API-Key: mysecretkeyhere header with all requests to /summarize, /metrics, etc.
    """

    def __init__(self, app: ASGIApp, api_key: str = None):
        super().__init__(app)
        self.api_key = api_key or os.environ.get("API_KEY", "")
        self.auth_enabled = bool(self.api_key)

        if self.auth_enabled:
            logger.info("API key authentication ENABLED")
        else:
            logger.warning(
                "API_KEY not set — authentication DISABLED. "
                "Set API_KEY in .env for production."
            )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth for public endpoints
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # Skip auth if not configured
        if not self.auth_enabled:
            return await call_next(request)

        # Validate key
        provided_key = request.headers.get("X-API-Key", "")
        if not provided_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "detail": "Missing X-API-Key header. "
                              "Include your API key in the X-API-Key request header.",
                    "status_code": 401,
                },
            )

        if provided_key != self.api_key:
            client_host = request.client.host if request.client else "unknown"
            logger.warning(
                f"Invalid API key attempt from {client_host} "
                f"on {request.method} {request.url.path}"
            )
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Forbidden",
                    "detail": "Invalid API key.",
                    "status_code": 403,
                },
            )

        return await call_next(request)
