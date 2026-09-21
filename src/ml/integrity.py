"""Model artifact integrity (Fase 0, Capa 3 de la arquitectura de seguridad).

Cada modelo entrenado se acompaña de un manifest sidecar (`<modelo>.sha256`)
con el hash SHA-256 del artefacto. `verify_model_integrity` se llama antes
de deserializar cualquier `.joblib` (champion o challenger): si el manifest
existe y no coincide, el artefacto pudo haber sido alterado o corrompido y
NUNCA se carga (invariante que truena, nunca se corrige en silencio —
mismo principio que `contract.py` en el proyecto BuyOrWait).

Si no hay manifest (artefacto entrenado antes de este cambio), solo se
registra una advertencia: esto nunca debe bloquear un despliegue que ya
funciona por falta de un manifest que todavía no existía.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_SUFFIX = ".sha256"


class ModelIntegrityError(RuntimeError):
    """El hash del artefacto de modelo no coincide con su manifest registrado."""


def _manifest_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.name + MANIFEST_SUFFIX)


def compute_sha256(path: Path | str) -> str:
    """Hash SHA-256 del archivo, leído en bloques (seguro para artefactos grandes)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(model_path: Path | str) -> Path:
    """Escribe el manifest `<modelo>.sha256` junto a un artefacto recién entrenado.

    Llamado por `src.ml.train.train_model` inmediatamente después de
    `joblib.dump`, para que todo modelo publicado por el pipeline oficial
    quede verificable.
    """
    resolved = Path(model_path)
    manifest_path = _manifest_path(resolved)
    manifest_path.write_text(compute_sha256(resolved) + "\n", encoding="utf-8")
    return manifest_path


def verify_model_integrity(model_path: Path | str) -> None:
    """Verifica un artefacto de modelo contra su manifest antes de cargarlo.

    Levanta `ModelIntegrityError` (nunca se auto-corrige) si el manifest
    existe y el hash no coincide. Si el manifest no existe, solo advierte.
    """
    resolved = Path(model_path)
    manifest_path = _manifest_path(resolved)
    if not manifest_path.is_file():
        logger.warning(
            "No se encontró manifest de integridad (%s) para %s; no se puede "
            "verificar que el artefacto no fue alterado. Vuelve a entrenar con "
            "`python -m src.ml.train` (o `make train`) para generarlo.",
            manifest_path,
            resolved,
        )
        return
    expected = manifest_path.read_text(encoding="utf-8").strip()
    actual = compute_sha256(resolved)
    if actual != expected:
        raise ModelIntegrityError(
            f"El hash de integridad de {resolved} no coincide con su manifest "
            f"({manifest_path}): esperado {expected}, obtenido {actual}. El "
            "artefacto pudo haber sido alterado o corrompido; no se carga."
        )
