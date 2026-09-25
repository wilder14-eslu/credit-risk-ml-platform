"""Fixtures compartidas: muestra real del dataset y modelos entrenados una vez por sesión."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_risk.config import PROJECT_ROOT, target_name
from credit_risk.data.quality import clean, to_canonical
from credit_risk.models.training import fit_model, make_splits

DATA_FILE = PROJECT_ROOT / "DATA" / "GiveMeSomeCredit" / "cs-training.csv"


@pytest.fixture(scope="session")
def raw_sample() -> pd.DataFrame:
    return pd.read_csv(DATA_FILE, nrows=20_000)


@pytest.fixture(scope="session")
def canonical(raw_sample) -> pd.DataFrame:
    data = clean(to_canonical(raw_sample))
    data[target_name()] = data[target_name()].astype(int)
    return data


@pytest.fixture(scope="session")
def splits(canonical):
    return make_splits(canonical)


@pytest.fixture(scope="session")
def lr_model(splits):
    return fit_model("logistic_regression", {}, splits)


@pytest.fixture(scope="session")
def xgb_model(splits):
    return fit_model("xgboost", {"n_estimators": 120}, splits)
