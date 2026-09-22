"""Punto de entrada de la aplicación FastAPI.

Seguridad (Fase 0, ver el documento de arquitectura híbrida): CORS
restringido a `CORS_ORIGINS`, rate limiting con `slowapi`, y los docs
interactivos ya no son públicos por defecto — se sirven protegidos desde
`src.api.docs` (o se desactivan del todo con `DOCS_ENABLED=false`). La
autenticación por API key vive en `src.api.security` y se aplica por
endpoint en `src.api.routes`.

Despliegue "clone-and-run" (Render, ver render.yaml): si `AUTO_TRAIN_MODEL`
está en true y no hay ningún artefacto en `MODEL_PATH`, se entrena uno al
arrancar (`src.ml.bootstrap`), igual que ya hacía `app.py` para Streamlit.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.docs import build_docs_router
from src.api.routes import router
from src.api.security import limiter
from src.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Entrena el modelo en background si AUTO_TRAIN_MODEL=true y falta el artefacto.

    No bloquea el arranque del servidor (el entrenamiento puede tomar
    varios segundos): /predict devuelve 503 mientras tanto, igual que si
    nunca se hubiera configurado, y sirve normal en cuanto termina.
    """
    if settings.auto_train_model:
        from src.ml.bootstrap import ensure_model_trained

        async def _train() -> None:
            try:
                await asyncio.to_thread(ensure_model_trained)
            except Exception:
                logger.exception("Falló el entrenamiento automático del modelo al arrancar.")

        asyncio.create_task(_train())
    yield


# docs_url/redoc_url/openapi_url del FastAPI por defecto quedan siempre
# desactivados: si DOCS_ENABLED, se remontan protegidos más abajo.
app = FastAPI(
    title=settings.app_name,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=_lifespan,
)

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
