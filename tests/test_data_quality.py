import pandas as pd
import pytest

from src.data_pipeline.validate import validate_input_data
from src.feature_store.features import FEATURE_NAMES, build_features


def valid_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": [0.2],
            "age": [40],
            "NumberOfTime30-59DaysPastDueNotWorse": [0],
            "DebtRatio": [0.3],
            "MonthlyIncome": [5000],
            "NumberOfOpenCreditLinesAndLoans": [4],
            "NumberOfTimes90DaysLate": [0],
            "NumberRealEstateLoansOrLines": [1],
            "NumberOfTime60-89DaysPastDueNotWorse": [0],
            "NumberOfDependents": [1],
            "target_default": [0],
        }
    )


def test_validation_accepts_kaggle_schema() -> None:
    assert validate_input_data(valid_data(), require_target=True)
    assert tuple(build_features(valid_data()).columns) == FEATURE_NAMES


def test_validation_rejects_invalid_age() -> None:
    data = valid_data()
    data.loc[0, "age"] = 17
    with pytest.raises(ValueError, match="age"):
        validate_input_data(data)


def test_validation_rejects_missing_column() -> None:
    data = valid_data().drop(columns=["NumberOfTimes90DaysLate"])
    with pytest.raises(ValueError, match="Schema error"):
        validate_input_data(data)
