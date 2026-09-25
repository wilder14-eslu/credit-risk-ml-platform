"""Backends de inferencia: Databricks Model Serving (prod) o modelo local (dev/CI)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Protocol

import pandas as pd

from credit_risk.config import PROJECT_ROOT
from credit_risk.models.credit_model import CreditRiskModel
from gateway.databricks_client import DatabricksREST

logger = logging.getLogger(__name__)


class ServingBackend(Protocol):
    name: str

    def predict(self, variant: str, records: list[dict], explain: bool) -> list[dict]: ...

    def versions(self) -> dict[str, str]: ...


class DatabricksServingBackend:
    """Consulta el endpoint de Model Serving de Databricks por variante."""

    name = "databricks"

    def __init__(self, client: DatabricksREST, endpoint: str, cache_seconds: float = 60.0):
        self.client = client
        self.endpoint = endpoint
        self.cache_seconds = cache_seconds
        self._versions: dict[str, str] = {}
        self._fetched_at = 0.0

    def versions(self) -> dict[str, str]:
        if time.monotonic() - self._fetched_at > self.cache_seconds:
            try:
                self._versions = self.client.served_versions(self.endpoint)
                self._fetched_at = time.monotonic()
            except Exception as exc:
                logger.warning("No se pudo leer el endpoint: %s", exc)
        return self._versions

    def predict(self, variant: str, records: list[dict], explain: bool) -> list[dict]:
        rows = self.client.invoke(self.endpoint, variant, records, {"explain": explain})
        for row in rows:
            if isinstance(row.get("top_factors"), str):
                row["top_factors"] = json.loads(row["top_factors"] or "[]")
        return rows


class LocalBackend:
    """Modelo en memoria. Si no hay joblib, entrena uno liviano con el dataset del repo."""

    name = "local"

    def __init__(self, champion: CreditRiskModel, challenger: CreditRiskModel | None = None):
        self.models = {"champion": champion}
        if challenger is not None:
            self.models["challenger"] = challenger

    @classmethod
    def from_path_or_train(cls, path: str) -> LocalBackend:
        import joblib

        file = Path(path)
        if file.exists():
            return cls(joblib.load(file))
        logger.warning("No existe %s: entrenando modelo de demostración", file)
        model = train_demo_model()
        file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, file)
        return cls(model)

    def versions(self) -> dict[str, str]:
        return {k: f"local-{m.algorithm}" for k, m in self.models.items()}

    def predict(self, variant: str, records: list[dict], explain: bool) -> list[dict]:
        model = self.models.get(variant, self.models["champion"])
        frame = model.predict_frame(pd.DataFrame(records), explain=explain)
        out = frame.to_dict(orient="records")
        for row in out:
            row["top_factors"] = json.loads(row.get("top_factors", "[]"))
        return out


def train_demo_model(sample: int = 30_000) -> CreditRiskModel:
    from credit_risk.data.quality import clean, to_canonical
    from credit_risk.models.training import fit_model, make_splits

    raw = pd.read_csv(PROJECT_ROOT / "DATA" / "GiveMeSomeCredit" / "cs-training.csv", nrows=sample)
    data = clean(to_canonical(raw))
    data["target_default"] = data["target_default"].astype(int)
    return fit_model("logistic_regression", {}, make_splits(data))
