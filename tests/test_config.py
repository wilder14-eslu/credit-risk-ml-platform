"""Tests de src.config: normalización de DATABASE_URL para Render."""

from __future__ import annotations

from src.config import _normalize_database_url


def test_normalize_database_url_adds_asyncpg_driver_to_postgres_scheme() -> None:
    """Render (y proveedores similares) entregan `postgres://...`."""
    raw = "postgres://user:pass@host:5432/dbname"
    assert _normalize_database_url(raw) == "postgresql+asyncpg://user:pass@host:5432/dbname"


def test_normalize_database_url_adds_asyncpg_driver_to_postgresql_scheme() -> None:
    raw = "postgresql://user:pass@host:5432/dbname"
    assert _normalize_database_url(raw) == "postgresql+asyncpg://user:pass@host:5432/dbname"


def test_normalize_database_url_leaves_explicit_driver_untouched() -> None:
    raw = "postgresql+asyncpg://user:pass@host:5432/dbname"
    assert _normalize_database_url(raw) == raw
