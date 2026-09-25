import numpy as np
import pytest

from credit_risk.config import base_feature_names, target_name
from credit_risk.data.quality import assert_quality, clean, run_expectations, to_canonical
from credit_risk.data.simulation import apply_scenario


def test_to_canonical_renames_and_adds_stable_id(raw_sample):
    data = to_canonical(raw_sample)
    assert set(base_feature_names()) <= set(data.columns)
    assert target_name() in data.columns
    assert data["applicant_id"].is_unique
    again = to_canonical(raw_sample)
    assert (data["applicant_id"] == again["applicant_id"]).all()


def test_expectations_pass_on_real_data(canonical):
    results = run_expectations(canonical)
    assert_quality(results)  # no lanza
    assert any(r.name == "default_rate_range" and r.passed for r in results)


def test_expectations_block_bad_batches(canonical):
    broken = canonical.copy()
    broken["age"] = np.nan
    with pytest.raises(ValueError, match="calidad"):
        assert_quality(run_expectations(broken))


def test_expectations_detect_duplicates(canonical):
    dup = canonical.copy()
    dup.loc[dup.index[:10], "applicant_id"] = dup["applicant_id"].iloc[0]
    failed = [r for r in run_expectations(dup) if not r.passed]
    assert any(r.name == "unique_applicant_id" for r in failed)


def test_clean_removes_out_of_range(raw_sample):
    data = to_canonical(raw_sample)
    data.loc[data.index[0], "age"] = 0
    cleaned = clean(data)
    assert (cleaned["age"] >= 18).all()
    assert len(cleaned) < len(data)


@pytest.mark.parametrize("scenario", ["none", "covariate", "concept", "prior", "mixed"])
def test_simulation_scenarios(canonical, scenario):
    out = apply_scenario(canonical, scenario, seed=1)
    assert len(out) == len(canonical)
    if scenario == "covariate":
        assert out["monthly_income"].mean() < canonical["monthly_income"].mean()
    if scenario in ("concept", "prior"):
        assert (out[base_feature_names()[0]] == canonical[base_feature_names()[0]]).all()
        assert not out[target_name()].equals(canonical[target_name()])


def test_simulation_rejects_unknown_scenario(canonical):
    with pytest.raises(ValueError):
        apply_scenario(canonical, "tsunami")
