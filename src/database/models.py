from datetime import datetime

from sqlalchemy import BigInteger, Boolean, DateTime, Float, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """Mirrors `src/database/init_db.sql`; written best-effort alongside the
    JSONL audit log (see `src.database.repository.persist_prediction`)."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    applicant_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(30), nullable=False)
    explanation: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Outcome(Base):
    """Ground truth reported later via `POST /api/v1/outcomes`; joined with
    `predictions` (by `applicant_id`) to compute live model performance."""

    __tablename__ = "outcomes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    applicant_id: Mapped[str] = mapped_column(String(100), nullable=False)
    actual_default: Mapped[bool] = mapped_column(Boolean, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
