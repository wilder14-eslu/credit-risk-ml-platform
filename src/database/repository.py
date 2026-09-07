"""Best-effort persistence of predictions/outcomes to Postgres.

Mirrors the "best-effort" pattern already used for MLflow logging in
`src.ml.train`: the JSONL audit trail (`src.api.monitoring`) is the source
of truth the platform actually reads from (drift/performance checks, the
Streamlit dashboard), so a database that is unreachable must never break a
prediction or outcome request -- it only means that particular row does not
additionally land in Postgres.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker

from src.database.models import Outcome, Prediction

logger = logging.getLogger(__name__)


async def persist_prediction(
    session_factory: async_sessionmaker,
    applicant_id: str | None,
    probability: float,
    decision: str,
    explanation: list[dict[str, Any]] | None = None,
) -> None:
    try:
        async with session_factory() as session:
            session.add(
                Prediction(
                    applicant_id=applicant_id,
                    probability=probability,
                    decision=decision,
                    explanation=explanation,
                )
            )
            await session.commit()
    except Exception:  # pragma: no cover - depends on an external database
        logger.exception("No se pudo persistir la predicción en Postgres.")


async def persist_outcome(
    session_factory: async_sessionmaker,
    applicant_id: str,
    actual_default: bool,
) -> None:
    try:
        async with session_factory() as session:
            session.add(Outcome(applicant_id=applicant_id, actual_default=actual_default))
            await session.commit()
    except Exception:  # pragma: no cover - depends on an external database
        logger.exception("No se pudo persistir el resultado real en Postgres.")
