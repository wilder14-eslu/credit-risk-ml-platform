"""Destinos del log de inferencias y outcomes (feedback loop hacia Delta Lake)."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Protocol

from credit_risk.config import base_feature_names
from gateway.databricks_client import DatabricksREST

logger = logging.getLogger(__name__)

LOG_COLUMNS = [
    "request_id",
    "applicant_id",
    "event_ts",
    "source",
    "variant",
    "model_version",
    "probability",
    "decision",
    "risk_band",
    "latency_ms",
    "scenario",
    *base_feature_names(),
]
OUTCOME_COLUMNS = ["applicant_id", "request_id", "actual_default", "observed_ts", "source"]


class Sink(Protocol):
    def log_prediction(self, row: dict) -> None: ...

    def log_outcome(self, row: dict) -> None: ...

    def flush(self) -> None: ...


class MemorySink:
    def __init__(self) -> None:
        self.predictions: list[dict] = []
        self.outcomes: list[dict] = []

    def log_prediction(self, row: dict) -> None:
        self.predictions.append(row)

    def log_outcome(self, row: dict) -> None:
        self.outcomes.append(row)

    def flush(self) -> None:
        return None


class SQLiteSink:
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS inference_log ({', '.join(LOG_COLUMNS)})")
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS outcomes ({', '.join(OUTCOME_COLUMNS)})")
            self.conn.commit()

    def _insert(self, table: str, columns: list[str], row: dict) -> None:
        with self.lock:
            self.conn.execute(
                f"INSERT INTO {table} VALUES ({', '.join('?' for _ in columns)})",
                [row.get(c) for c in columns],
            )
            self.conn.commit()

    def log_prediction(self, row: dict) -> None:
        self._insert("inference_log", LOG_COLUMNS, row)

    def log_outcome(self, row: dict) -> None:
        self._insert("outcomes", OUTCOME_COLUMNS, row)

    def flush(self) -> None:
        return None


def _sql_literal(value) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(float(value)) if isinstance(value, float) else str(value)
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


class DatabricksDeltaSink:
    """Buffer en memoria que se vacía en lotes vía SQL Statement Execution API.

    Evita un INSERT por request (el warehouse 2X-Small de Free Edition tiene
    arranque en frío); se vacía cada `flush_every_n` filas o `flush_every_seconds`.
    """

    def __init__(
        self,
        client: DatabricksREST,
        warehouse_id: str,
        fq_schema: str,
        flush_every_n: int = 20,
        flush_every_seconds: float = 10.0,
    ) -> None:
        self.client = client
        self.warehouse_id = warehouse_id
        self.fq = fq_schema
        self.flush_every_n = flush_every_n
        self.flush_every_seconds = flush_every_seconds
        self._buffer: dict[str, list[dict]] = {"inference_log": [], "outcomes": []}
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()

    def _maybe_flush(self) -> None:
        pending = sum(len(v) for v in self._buffer.values())
        if pending >= self.flush_every_n or time.monotonic() - self._last_flush > self.flush_every_seconds:
            self.flush()

    def log_prediction(self, row: dict) -> None:
        with self._lock:
            self._buffer["inference_log"].append(row)
        self._maybe_flush()

    def log_outcome(self, row: dict) -> None:
        with self._lock:
            self._buffer["outcomes"].append(row)
        self._maybe_flush()

    def flush(self) -> None:
        with self._lock:
            batches = {k: v for k, v in self._buffer.items() if v}
            self._buffer = {"inference_log": [], "outcomes": []}
            self._last_flush = time.monotonic()
        for table, rows in batches.items():
            columns = LOG_COLUMNS if table == "inference_log" else OUTCOME_COLUMNS
            ts_col = "event_ts" if table == "inference_log" else "observed_ts"
            values = []
            for row in rows:
                cells = []
                for c in columns:
                    lit = _sql_literal(row.get(c))
                    cells.append(f"CAST({lit} AS TIMESTAMP)" if c == ts_col and lit != "NULL" else lit)
                values.append("(" + ", ".join(cells) + ")")
            statement = f"INSERT INTO {self.fq}.{table} ({', '.join(columns)}) VALUES " + ", ".join(values)
            try:
                self.client.sql(self.warehouse_id, statement)
                logger.info("%d filas -> %s.%s", len(rows), self.fq, table)
            except Exception as exc:  # no perder datos si el warehouse falla
                logger.error("Fallo al escribir en Delta (%s); se reencolan %d filas", exc, len(rows))
                with self._lock:
                    self._buffer[table].extend(rows)
