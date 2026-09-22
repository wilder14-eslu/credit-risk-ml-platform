import logging
import os
import secrets
from typing import ClassVar

from dotenv import load_dotenv

# El resto del módulo lee configuración con os.getenv(...) directo, así que
# el .env debe cargarse aquí, antes de que la clase Settings se defina y
# se instancie más abajo. Sin esta línea, crear/editar .env no tenía ningún
# efecto (bug preexistente: python-dotenv ya era una dependencia transitiva
# de pydantic-settings, pero nunca se invocaba).
load_dotenv()

logger = logging.getLogger(__name__)


def _normalize_database_url(raw_url: str) -> str:
    """Fuerza el driver async (asyncpg) en la URL de Postgres.

    Render (y otros proveedores) entregan la connection string como
    `postgres://...` o `postgresql://...`; SQLAlchemy con
    `create_async_engine` (src/database/connection.py) exige el driver
    explícito `postgresql+asyncpg://...`, o falla al primer uso.
    """
    if raw_url.startswith("postgres://"):
        return "postgresql+asyncpg://" + raw_url[len("postgres://") :]
    if raw_url.startswith("postgresql://"):
        return "postgresql+asyncpg://" + raw_url[len("postgresql://") :]
    return raw_url


def _generate_dev_api_key() -> str:
    """Clave temporal para una sola ejecución, cuando API_KEY no está configurada.

    Fase 0, Capa 1: la ausencia de API_KEY nunca debe significar "sin
    autenticación" (fail closed, no fail open). Se registra en el log para
    que quien arranca el servicio localmente pueda usarla de inmediato;
    para un valor estable entre reinicios, definir API_KEY en `.env`.
    """
    key = secrets.token_urlsafe(32)
    logger.warning(
        "API_KEY no está configurada: se generó una clave temporal solo para "
        "esta ejecución (cámbiala a un valor fijo en tu .env para producción). "
        "X-API-Key: %s",
        key,
    )
    return key


class Settings:
    app_name = os.getenv("APP_NAME", "Credit Risk ML Platform")
    database_url = _normalize_database_url(
        os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://credit_risk:credit_risk@localhost:5432/credit_risk",
        )
    )
    model_path = os.getenv("MODEL_PATH", "data/processed/model.joblib")

    # --- Fase 0: seguridad de la API (ver el doc de arquitectura híbrida) ---
    api_key = os.getenv("API_KEY") or _generate_dev_api_key()
    cors_origins: ClassVar[list[str]] = [
        origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()
    ]
    rate_limit = os.getenv("RATE_LIMIT", "60/minute")
    docs_enabled = os.getenv("DOCS_ENABLED", "true").strip().lower() != "false"

    # Entrena un modelo automáticamente al arrancar si no hay ningún
    # artefacto en MODEL_PATH (ver src/ml/bootstrap.py). Pensado para un
    # despliegue "clone-and-run" (Render) donde el modelo no está commiteado
    # a git; en local normalmente se deja en false (falla con 503 en vez de
    # entrenar en segundo plano sin avisar).
    auto_train_model = os.getenv("AUTO_TRAIN_MODEL", "false").strip().lower() == "true"


settings = Settings()
