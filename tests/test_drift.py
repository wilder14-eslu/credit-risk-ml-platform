import pandas as pd

from src.monitoring.drift import (
    compute_feature_drift,
    load_reference_distribution,
    population_stability_index,
    save_reference_distribution,
)


def test_psi_is_zero_for_identical_distributions() -> None:
    proportions = [0.1, 0.2, 0.3, 0.2, 0.2]
    assert population_stability_index(proportions, proportions) == 0.0


def test_psi_is_high_for_shifted_distributions() -> None:
    expected = [0.5, 0.5]
    actual = [0.05, 0.95]
    assert population_stability_index(expected, actual) > 0.2


def test_compute_feature_drift_flags_a_shifted_feature(tmp_path) -> None:
    training = pd.DataFrame({"age": list(range(18, 78))})
    reference_path = save_reference_distribution(training, tmp_path / "reference.json")
    reference = load_reference_distribution(reference_path)

    stable_traffic = pd.DataFrame({"age": list(range(18, 78))})
    stable_result = compute_feature_drift(stable_traffic, reference)
    assert not stable_result["drift_detected"]

    shifted_traffic = pd.DataFrame({"age": [80] * 60})
    shifted_result = compute_feature_drift(shifted_traffic, reference)
    assert shifted_result["drift_detected"]
    assert "age" in shifted_result["drifted_features"]


def test_compute_feature_drift_handles_no_traffic_yet() -> None:
    result = compute_feature_drift(pd.DataFrame(), reference={})
    assert result == {
        "psi": {},
        "drifted_features": [],
        "drift_detected": False,
        "threshold": 0.2,
    }
