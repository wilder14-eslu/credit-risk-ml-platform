import logging
import os
import secrets
from typing import ClassVar

logger = logging.getLogger(__name__)


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
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://credit_risk:credit_risk@localhost:5432/credit_risk",
    )
    model_path = os.getenv("MODEL_PATH", "data/processed/model.joblib")

    # --- Fase 0: seguridad de la API (ver el doc de arquitectura híbrida) ---
    api_key = os.getenv("API_KEY") or _generate_dev_api_key()
    cors_origins: ClassVar[list[str]] = [
        origin.strip() for origin in os.getenv("CORS_ORIGINS", "").split(",") if origin.strip()
    ]
    rate_limit = os.getenv("RATE_LIMIT", "60/minute")
    docs_enabled = os.getenv("DOCS_ENABLED", "true").strip().lower() != "false"


settings = Settings()
