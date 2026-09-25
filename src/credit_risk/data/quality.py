"""Ingesta y calidad de datos (expectativas tipo Great Expectations, sin la dependencia).

Las funciones trabajan con pandas para poder probarse sin Spark; los jobs de
Databricks las usan sobre el DataFrame leído desde Delta y persisten el
resultado de las expectativas en la tabla `data_quality_log`.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from credit_risk.config import feature_definitions, target_name, target_source

logger = logging.getLogger(__name__)


@dataclass
class Expectation:
    name: str
    column: str
    passed: bool
    observed: float
    threshold: float
    severity: str  # "error" bloquea el pipeline, "warning" solo se registra

    def to_dict(self) -> dict:
        return asdict(self)


def to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas crudas de Kaggle a nombres canónicos y agrega `applicant_id`.

    El `applicant_id` es un hash estable del índice original, así la misma fila
    siempre recibe el mismo id (clave primaria de la feature table).
    """
    data = raw.copy()
    unnamed = [c for c in data.columns if str(c).startswith("Unnamed") or c == ""]
    if unnamed:
        data = data.rename(columns={unnamed[0]: "source_row_id"})
    elif "source_row_id" not in data.columns:
        data["source_row_id"] = np.arange(1, len(data) + 1)

    rename = {d["source"]: name for name, d in feature_definitions().items()}
    rename[target_source()] = target_name()
    data = data.rename(columns=rename)

    for name in feature_definitions():
        if name not in data.columns:
            raise ValueError(f"Falta la columna requerida: {name}")
        data[name] = pd.to_numeric(data[name], errors="coerce").astype("float64")
    if target_name() in data.columns:
        data[target_name()] = pd.to_numeric(data[target_name()], errors="coerce")

    data["applicant_id"] = data["source_row_id"].map(
        lambda v: "APP-" + hashlib.sha1(str(int(v)).encode()).hexdigest()[:12]
    )
    cols = ["applicant_id", "source_row_id", *feature_definitions()]
    if target_name() in data.columns:
        cols.append(target_name())
    return data[cols]


def run_expectations(data: pd.DataFrame, require_target: bool = True) -> list[Expectation]:
    """Evalúa las expectativas de calidad sobre un lote en formato canónico."""
    results: list[Expectation] = []
    n = max(len(data), 1)

    results.append(Expectation("row_count_min", "*", len(data) >= 1000, float(len(data)), 1000.0, "error"))
    dup = float(data["applicant_id"].duplicated().mean()) if "applicant_id" in data else 0.0
    results.append(Expectation("unique_applicant_id", "applicant_id", dup == 0.0, dup, 0.0, "error"))

    for name, definition in feature_definitions().items():
        values = pd.to_numeric(data[name], errors="coerce")
        null_rate = float(values.isna().mean())
        # Ingreso y dependientes traen ~20% y ~3% de nulos en el dataset original.
        max_null = 0.30 if name in {"monthly_income", "number_dependents"} else 0.01
        results.append(Expectation("null_rate", name, null_rate <= max_null, null_rate, max_null, "error"))
        if "min" in definition:
            rate = float((values.dropna() < definition["min"]).sum() / n)
            results.append(Expectation("min_value", name, rate <= 0.001, rate, 0.001, "warning"))
        if "max" in definition:
            rate = float((values.dropna() > definition["max"]).sum() / n)
            results.append(Expectation("max_value", name, rate <= 0.001, rate, 0.001, "warning"))

    if require_target:
        target = pd.to_numeric(data.get(target_name()), errors="coerce")
        valid = bool(target.dropna().isin([0, 1]).all()) if target is not None else False
        results.append(Expectation("target_binary", target_name(), valid, float(valid), 1.0, "error"))
        rate = float(target.mean()) if target is not None else float("nan")
        results.append(
            Expectation("default_rate_range", target_name(), 0.01 <= rate <= 0.30, rate, 0.30, "error")
        )
    return results


def assert_quality(results: list[Expectation]) -> None:
    failed = [r for r in results if not r.passed and r.severity == "error"]
    for r in results:
        if not r.passed and r.severity == "warning":
            logger.warning("Expectativa en warning: %s(%s)=%.4f", r.name, r.column, r.observed)
    if failed:
        detail = ", ".join(f"{r.name}({r.column})={r.observed:.4f}" for r in failed)
        raise ValueError(f"Gate de calidad de datos fallido: {detail}")


def clean(data: pd.DataFrame) -> pd.DataFrame:
    """Descarta filas fuera de rango (p.ej. edad 0) y recorta outliers extremos.

    `RevolvingUtilization` y `DebtRatio` tienen valores de miles en el dataset
    original (errores de captura); se recortan al percentil 99.9 para que no
    dominen el entrenamiento ni el cálculo de PSI.
    """
    mask = pd.Series(True, index=data.index)
    for name, definition in feature_definitions().items():
        values = data[name]
        if "min" in definition:
            mask &= values.isna() | (values >= definition["min"])
        if "max" in definition:
            mask &= values.isna() | (values <= definition["max"])
    cleaned = data.loc[mask].copy()
    for col in ("revolving_utilization_unsecured", "debt_ratio"):
        upper = cleaned[col].quantile(0.999)
        cleaned[col] = cleaned[col].clip(upper=upper)
    dropped = len(data) - len(cleaned)
    if dropped:
        logger.info("Filas descartadas por rango: %d", dropped)
    return cleaned
