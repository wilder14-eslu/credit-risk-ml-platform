"""Fase 0, Capa 3: el modelo nunca se carga si su hash no coincide con el manifest."""

from __future__ import annotations

import pytest

from src.ml.integrity import (
    ModelIntegrityError,
    compute_sha256,
    verify_model_integrity,
    write_manifest,
)


def test_write_manifest_then_verify_passes(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"contenido de prueba del modelo")

    manifest_path = write_manifest(model_path)

    assert manifest_path.is_file()
    assert manifest_path.read_text(encoding="utf-8").strip() == compute_sha256(model_path)
    verify_model_integrity(model_path)  # no debe levantar


def test_verify_raises_on_tampered_artifact(tmp_path) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"contenido original")
    write_manifest(model_path)

    model_path.write_bytes(b"contenido alterado despues del manifest")

    with pytest.raises(ModelIntegrityError):
        verify_model_integrity(model_path)


def test_verify_without_manifest_only_warns(tmp_path, caplog) -> None:
    model_path = tmp_path / "model.joblib"
    model_path.write_bytes(b"artefacto sin manifest, ej. entrenado antes de este cambio")

    verify_model_integrity(model_path)  # no debe levantar, solo advertir
    assert "manifest" in caplog.text.lower()
