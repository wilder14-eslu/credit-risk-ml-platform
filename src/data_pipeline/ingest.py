"""Data ingestion entry point.

Resolves the local directory holding the raw "Give Me Some Credit" CSVs,
downloading it on demand via `kagglehub` if it is not already present. In
this repository the dataset ships committed under `DATA/GiveMeSomeCredit/`
(see `.gitignore` if you decide to stop tracking it), so in practice this
function is a no-op that just returns that path.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_DATA_DIR = Path("DATA/GiveMeSomeCredit")
KAGGLE_COMPETITION = "GiveMeSomeCredit"


def download_give_me_some_credit(destination: Path = DEFAULT_DATA_DIR) -> Path:
    """Return a local directory containing the raw Kaggle CSV files.

    If ``destination`` already has CSV files (the common case for this repo),
    they are reused as-is. Otherwise the Kaggle competition dataset is
    downloaded via `kagglehub`, which requires Kaggle API credentials
    (``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME``/``KAGGLE_KEY``
    environment variables) and that you have accepted the competition rules
    on kaggle.com.
    """
    destination = Path(destination)
    if any(destination.glob("*.csv")):
        return destination

    try:
        import kagglehub
    except ImportError as error:  # pragma: no cover - exercised only when data is missing
        raise RuntimeError(
            f"No se encontraron archivos CSV en {destination} y `kagglehub` no "
            "está instalado. Instala `kagglehub` o coloca manualmente "
            "cs-training.csv / cs-test.csv en esa carpeta."
        ) from error

    downloaded_path = Path(kagglehub.competition_download(KAGGLE_COMPETITION))
    destination.mkdir(parents=True, exist_ok=True)
    for csv_file in downloaded_path.glob("*.csv"):
        target = destination / csv_file.name
        if not target.exists():
            target.write_bytes(csv_file.read_bytes())
    return destination
