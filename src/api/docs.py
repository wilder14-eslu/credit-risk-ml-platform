"""Documentación interactiva del API (Swagger UI / ReDoc), protegida.

Fase 0: `/docs`, `/redoc` y `/openapi.json` estaban corriendo abiertos por
defecto (confirmado en local) — cualquiera podía explorar el schema
completo y probar el API en vivo ("Try it out") sin credenciales. En vez
de desactivarlos, se sirven aquí detrás de la misma `verify_api_key` que
protege el resto del API; `DOCS_ENABLED=false` los quita por completo si
se prefiere no exponerlos ni siquiera autenticados.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from src.api.security import verify_api_key


def build_docs_router(app: FastAPI) -> APIRouter:
    router = APIRouter(include_in_schema=False, dependencies=[Depends(verify_api_key)])

    @router.get("/openapi.json")
    async def protected_openapi() -> dict:
        return get_openapi(title=app.title, version=app.version, routes=app.routes)

    @router.get("/docs")
    async def protected_docs():
        return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} - Docs")

    @router.get("/redoc")
    async def protected_redoc():
        return get_redoc_html(openapi_url="/openapi.json", title=f"{app.title} - ReDoc")

    return router
