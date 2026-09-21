"""Punto de entrada de la aplicación FastAPI.

Seguridad (Fase 0, ver el documento de arquitectura híbrida): CORS
restringido a `CORS_ORIGINS`, rate limiting con `slowapi`, y los docs
interactivos ya no son públicos por defecto — se sirven protegidos desde
`src.api.docs` (o se desactivan del todo con `DOCS_ENABLED=false`). La
autenticación por API key vive en `src.api.security` y se aplica por
endpoint en `src.api.routes`.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.docs import build_docs_router
from src.api.routes import router
from src.api.security import limiter
from src.config import settings

# docs_url/redoc_url/openapi_url del FastAPI por defecto quedan siempre
# desactivados: si DOCS_ENABLED, se remontan protegidos más abajo.
app = FastAPI(title=settings.app_name, docs_url=None, redoc_url=None, openapi_url=None)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*", "X-API-Key"],
)

app.include_router(router)
if settings.docs_enabled:
    app.include_router(build_docs_router(app))
