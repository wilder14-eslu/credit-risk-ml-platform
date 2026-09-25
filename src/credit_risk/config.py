"""Configuración central: esquema de datos, umbrales y nombres en Unity Catalog.

Un solo lugar define cómo se llaman las tablas Delta, el modelo registrado y
el endpoint de serving, de modo que los jobs de Databricks, el gateway en
Render y los tests usen exactamente los mismos nombres.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.getenv("CREDIT_RISK_CONFIG_DIR", PROJECT_ROOT / "config"))


def _load_yaml(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache(maxsize=1)
def data_schema() -> dict[str, Any]:
    """Esquema canónico (columnas crudas -> features, límites, etiquetas)."""
    return _load_yaml("data_schema.yaml")


@lru_cache(maxsize=1)
def platform_config() -> dict[str, Any]:
    """Umbrales de entrenamiento, gates, A/B testing y monitoreo."""
    return _load_yaml("platform.yaml")


@dataclass(frozen=True)
class UCNames:
    """Nombres completos de los objetos en Unity Catalog (catalog.schema.obj)."""

    catalog: str = field(default_factory=lambda: os.getenv("UC_CATALOG", "workspace"))
    schema: str = field(default_factory=lambda: os.getenv("UC_SCHEMA", "credit_risk"))

    def _fq(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    # Medallion
    @property
    def bronze_applications(self) -> str:
        return self._fq("bronze_applications")

    @property
    def silver_applications(self) -> str:
        return self._fq("silver_applications")

    @property
    def data_quality_log(self) -> str:
        return self._fq("data_quality_log")

    # Feature store (tabla de features con PK en UC)
    @property
    def feature_table(self) -> str:
        return self._fq("gold_credit_features")

    @property
    def production_pool(self) -> str:
        """Solicitantes reservados que nunca se usan para entrenar."""
        return self._fq("gold_production_pool")

    # Producción / feedback loop
    @property
    def inference_log(self) -> str:
        return self._fq("inference_log")

    @property
    def outcomes(self) -> str:
        return self._fq("outcomes")

    @property
    def reference_profile(self) -> str:
        return self._fq("reference_profile")

    # Monitoreo y experimentación
    @property
    def monitoring_metrics(self) -> str:
        return self._fq("monitoring_metrics")

    @property
    def drift_features(self) -> str:
        return self._fq("drift_by_feature")

    @property
    def ab_results(self) -> str:
        return self._fq("ab_test_results")

    @property
    def model_benchmark(self) -> str:
        return self._fq("model_benchmark")

    @property
    def retrain_events(self) -> str:
        return self._fq("retrain_events")

    # Modelo registrado y volumen de artefactos
    @property
    def model_name(self) -> str:
        return self._fq("credit_default_model")

    @property
    def volume(self) -> str:
        return self._fq("artifacts")

    @property
    def volume_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/artifacts"


CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "/Shared/credit-risk-ml-platform")


def feature_definitions() -> dict[str, dict[str, Any]]:
    return data_schema()["features"]


def base_feature_names() -> tuple[str, ...]:
    return tuple(feature_definitions())


def target_name() -> str:
    return data_schema()["target"]


def target_source() -> str:
    return data_schema().get("target_source", target_name())
