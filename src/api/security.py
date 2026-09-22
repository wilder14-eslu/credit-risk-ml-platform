"""Autenticacion por API key, rate limiting y utilidades compartidas de la API.

Fase 0, Capa 1 de la arquitectura hibrida de seguridad: todo endpoint bajo
`/api/v1` que lea o escriba datos de negocio requiere la cabecera
`X-API-Key`, comparada con `secrets.compare_digest` (evita un side-channel
de tiempo). `src.config.settings.api_key` nunca es None: si no se
configuro, se genero y registro una clave temporal al arrancar (fail
closed, nunca fail open).

`/docs`, `/redoc` y `/openapi.json` usan `verify_api_key_for_docs` en vez
de `verify_api_key`: un navegador normal no puede adjuntar cabeceras
personalizadas en una navegacion simple (clic en un enlace o pegar la URL),
asi que esas tres rutas de documentacion tambien aceptan `?api_key=...`
en la URL. Es un trade-off deliberado (la key queda en el historial del
navegador) aceptado solo para las rutas de documentacion; `/predict`,
`/outcomes` y `/monitoring/status` siguen exigiendo unicamente la cabecera.
"""

from __future__ import annotations

import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.config import settings

API_KEY_HEADER_NAME = "X-API-Key"
API_KEY_QUERY_NAME = "api_key"

_api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)
_api_key_query = APIKeyQuery(name=API_KEY_QUERY_NAME, auto_error=False)

# Un solo limiter compartido entre main.py (registro del middleware/handler)
# y routes.py (decorador por endpoint), para que ambos vean la misma cuota.
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])


async def verify_api_key(provided: str | None = Security(_api_key_header)) -> str:
    """Dependency de FastAPI: exige `X-API-Key` valida o responde 401.

    Usada por los endpoints de negocio (`/predict`, `/outcomes`,
    `/monitoring/status`): solo acepta la cabecera, nunca query string.
    """
    if provided is None or not secrets.compare_digest(provided, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Falta o es invalida la cabecera {API_KEY_HEADER_NAME}.",
        )
    return provided


async def verify_api_key_for_docs(
    header_key: str | None = Security(_api_key_header),
    query_key: str | None = Security(_api_key_query),
) -> str:
    """Como `verify_api_key`, pero tambien acepta `?api_key=...` en la URL.

    Solo se usa para /docs, /redoc y /openapi.json (ver docstring del
    modulo). La cabecera tiene prioridad si llegan ambas.
    """
    candidate = header_key or query_key
    if candidate is None or not secrets.compare_digest(candidate, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                f"Falta o es invalida la API key "
                f"(cabecera {API_KEY_HEADER_NAME} o parametro ?{API_KEY_QUERY_NAME}=)."
            ),
        )
    return candidate
