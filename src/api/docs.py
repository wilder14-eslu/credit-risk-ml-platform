"""Documentacion interactiva del API (Swagger UI / ReDoc), protegida.

Fase 0: `/docs`, `/redoc` y `/openapi.json` estaban corriendo abiertos por
defecto (confirmado en local) -- cualquiera podia explorar el schema
completo y probar el API en vivo ("Try it out") sin credenciales. En vez
de desactivarlos, se sirven aqui detras de `verify_api_key_for_docs`
(cabecera `X-API-Key` o `?api_key=...`, ver `src/api/security.py`):
un navegador normal no puede mandar cabeceras personalizadas al navegar
directamente a una URL, asi que solo estas tres rutas de documentacion
aceptan tambien la query string. `DOCS_ENABLED=false` las quita por
completo si se prefiere no exponerlas ni siquiera autenticadas.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from src.api.security import verify_api_key_for_docs


def build_docs_router(app: FastAPI) -> APIRouter:
    router = APIRouter(include_in_schema=False)

    @router.get("/openapi.json")
    async def protected_openapi(api_key: str = Depends(verify_api_key_for_docs)) -> dict:
        return get_openapi(title=app.title, version=app.version, routes=app.routes)

    @router.get("/docs")
    async def protected_docs(api_key: str = Depends(verify_api_key_for_docs)):
        # Se propaga la api_key al openapi_url para que el fetch que hace
        # el propio Swagger UI (JS, mismo origen) tambien quede autenticado;
        # de lo contrario la pagina carga pero el schema devuelve 401.
        return get_swagger_ui_html(
            openapi_url=f"/openapi.json?api_key={api_key}",
            title=f"{app.title} - Docs",
        )

    @router.get("/redoc")
    async def protected_redoc(api_key: str = Depends(verify_api_key_for_docs)):
        return get_redoc_html(
            openapi_url=f"/openapi.json?api_key={api_key}",
            title=f"{app.title} - ReDoc",
        )

    return router
