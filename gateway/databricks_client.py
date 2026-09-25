"""Cliente REST mínimo para la API de Databricks (sin dependencias pesadas).

Usa tres APIs del workspace:
- Model Serving: `/serving-endpoints/{ep}/served-models/{variant}/invocations`
- SQL Statement Execution: `/api/2.0/sql/statements` (escribir/leer Delta)
- Unity Catalog: `/api/2.1/unity-catalog/models/{name}` (aliases del modelo)
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class DatabricksError(RuntimeError):
    pass


class DatabricksREST:
    def __init__(
        self, host: str, token: str, timeout: float = 60.0, transport: httpx.BaseTransport | None = None
    ):
        if not host or not token:
            raise DatabricksError("DATABRICKS_HOST y DATABRICKS_TOKEN son obligatorios")
        self.host = host.rstrip("/")
        if not self.host.startswith("http"):
            self.host = "https://" + self.host
        self._client = httpx.Client(
            base_url=self.host,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
            transport=transport,
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(path, json=payload)
        if response.status_code >= 400:
            raise DatabricksError(f"{path} -> {response.status_code}: {response.text[:500]}")
        return response.json()

    def invoke(
        self, endpoint: str, served_model: str, records: list[dict], params: dict | None = None
    ) -> list[dict]:
        payload: dict[str, Any] = {"dataframe_records": records}
        if params:
            payload["params"] = params
        body = self._post(f"/serving-endpoints/{endpoint}/served-models/{served_model}/invocations", payload)
        predictions = body.get("predictions", body)
        if isinstance(predictions, dict):  # formato "split"/columnar
            keys = list(predictions)
            predictions = [
                dict(zip(keys, vals, strict=False)) for vals in zip(*predictions.values(), strict=False)
            ]
        return predictions

    def served_versions(self, endpoint: str) -> dict[str, str]:
        response = self._client.get(f"/api/2.0/serving-endpoints/{endpoint}")
        if response.status_code >= 400:
            raise DatabricksError(f"endpoint {endpoint}: {response.status_code} {response.text[:300]}")
        config = response.json().get("config", {})
        return {e.get("name"): str(e.get("entity_version")) for e in config.get("served_entities", [])}

    def model_aliases(self, full_name: str) -> dict[str, str]:
        response = self._client.get(
            f"/api/2.1/unity-catalog/models/{full_name}", params={"include_aliases": "true"}
        )
        if response.status_code >= 400:
            return {}
        return {a["alias_name"]: str(a["version_num"]) for a in response.json().get("aliases", [])}

    def sql(self, warehouse_id: str, statement: str, parameters: list[dict] | None = None) -> list[dict]:
        body = self._post(
            "/api/2.0/sql/statements",
            {
                "warehouse_id": warehouse_id,
                "statement": statement,
                "parameters": parameters or [],
                "wait_timeout": "30s",
                "on_wait_timeout": "CONTINUE",
                "format": "JSON_ARRAY",
                "disposition": "INLINE",
            },
        )
        state = body.get("status", {}).get("state")
        if state == "FAILED":
            raise DatabricksError(body["status"].get("error", {}).get("message", "SQL failed"))
        columns = [c["name"] for c in body.get("manifest", {}).get("schema", {}).get("columns", [])]
        rows = body.get("result", {}).get("data_array", []) or []
        return [dict(zip(columns, row, strict=False)) for row in rows]
