"""Autenticación por API key, rate limiting y utilidades compartidas de la API.

Fase 0, Capa 1 de la arquitectura híbrida de seguridad: todo endpoint bajo
`/api/v1` que lea o escriba datos de negocio requiere la cabecera
`X-API-Key`, comparada con `secrets.compare_digest` (evita un side-channel
de tiempo). `src.config.settings.api_key` nunca es None: si no se
configuró, se generó y registró una clave temporal al arrancar (fail
closed, nunca fail open).
"""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.config import settings

API_KEY_HEADER_NAME = "X-API-Key"

_api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

# Un solo limiter compartido entre main.py (registro del middleware/handler)
# y routes.py (decorador por endpoint), para que ambos vean la misma cuota.
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])


async def verify_api_key(provided: str | None = Security(_api_key_header)) -> str:
    """Dependency de FastAPI: exige `X-API-Key` válida o responde 401."""
    if provided is None or not secrets.compare_digest(provided, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Falta o es inválida la cabecera {API_KEY_HEADER_NAME}.",
        )
    return provided
