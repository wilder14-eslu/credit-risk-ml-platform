"""End-to-end smoke test: train a small model and score one applicant.

This exercises the real training/inference path (not mocks), which is what
actually catches drift between `src.ml.train`, `src.ml.predict` and the
canonical feature schema.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.ml.predict import predict_default_probability
from src.ml.train import train_model

DATA_FILE = Path("DATA/GiveMeSomeCredit/cs-training.csv")


@pytest.mark.skipif(not DATA_FILE.is_file(), reason="Dataset not available locally")
def test_train_and_predict_roundtrip(tmp_path) -> None:
    raw_data = pd.read_csv(DATA_FILE, index_col=0).sample(n=2000, random_state=42)
    model_path = tmp_path / "model.joblib"

    metrics = train_model(raw_data, model_path=model_path, register=False)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert model_path.is_file()

    applicant = {
        "RevolvingUtilizationOfUnsecuredLines": 0.3,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.25,
        "MonthlyIncome": 6000,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2,
    }
    result = predict_default_probability(applicant, model_path=model_path)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["decision"] in {"aprobar", "rechazar"}
    assert result["risk_band"] in {"bajo", "medio", "alto"}
