"""Configuración del gateway (variables de entorno en Render)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "databricks": consulta el endpoint de Model Serving (producción)
    # "local": carga un modelo joblib o entrena uno de demostración (dev / CI)
    serving_backend: str = "local"
    # "databricks": escribe inferencias en Delta vía SQL Statement Execution API
    # "sqlite" | "memory": desarrollo y tests
    log_sink: str = "sqlite"

    databricks_host: str = ""
    databricks_token: str = ""
    databricks_warehouse_id: str = ""
    serving_endpoint: str = "credit-risk-endpoint"
    uc_catalog: str = "workspace"
    uc_schema: str = "credit_risk"

    api_keys: str = ""  # lista separada por comas; vacío = sin autenticación (solo dev)
    challenger_traffic: float | None = None  # None = usar config/platform.yaml
    local_model_path: str = "artifacts/local_model.joblib"
    sqlite_path: str = "artifacts/inference_log.sqlite"
    request_timeout_seconds: float = 60.0
    flush_every_n: int = 20
    flush_every_seconds: float = 10.0

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def fq(self) -> str:
        return f"{self.uc_catalog}.{self.uc_schema}"


@lru_cache(maxsize=1)
def get_settings() -> GatewaySettings:
    return GatewaySettings()
