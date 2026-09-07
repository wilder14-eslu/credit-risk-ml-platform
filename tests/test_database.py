from src.database.models import Base


def test_prediction_and_outcome_models_are_registered() -> None:
    tables = Base.metadata.tables
    assert "predictions" in tables
    assert "outcomes" in tables
    assert {"applicant_id", "probability", "decision"}.issubset(
        column.name for column in tables["predictions"].columns
    )
    assert {"applicant_id", "actual_default"}.issubset(
        column.name for column in tables["outcomes"].columns
    )
