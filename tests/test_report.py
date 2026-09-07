from src.ml.report import build_report_sections, render_markdown, top_feature_importances


def _fake_train_result() -> dict:
    train_metrics = {
        "roc_auc": 0.90,
        "pr_auc": 0.60,
        "gini": 0.80,
        "ks_statistic": 0.55,
        "precision": 0.7,
        "recall": 0.6,
        "f1": 0.65,
        "log_loss": 0.3,
        "brier_score": 0.08,
    }
    test_metrics = {
        "roc_auc": 0.83,
        "pr_auc": 0.52,
        "gini": 0.66,
        "ks_statistic": 0.48,
        "precision": 0.65,
        "recall": 0.55,
        "f1": 0.60,
        "log_loss": 0.35,
        "brier_score": 0.10,
    }
    return {
        "algorithm": "xgboost",
        "run_id": "abc123",
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "diagnosis": {"label": "buen_ajuste", "gap": 0.07, "explanation": "todo bien"},
        "latency_ms_per_row": 0.42,
    }


def test_build_report_sections_uses_the_template_narrative_without_llm() -> None:
    sections = build_report_sections(_fake_train_result())
    assert sections["algorithm"] == "xgboost"
    assert "xgboost" in sections["narrative"]
    assert len(sections["metrics_table"]) == 9


def test_render_markdown_includes_metrics_and_narrative() -> None:
    sections = build_report_sections(_fake_train_result())
    markdown = render_markdown(sections)
    assert "ROC-AUC" in markdown
    assert sections["narrative"] in markdown
    assert "buen_ajuste" in markdown


def test_top_feature_importances_reads_tree_model_attribute() -> None:
    class FakeTreeModel:
        feature_importances_ = [0.1, 0.5, 0.4]

    result = top_feature_importances(FakeTreeModel(), ["age", "income", "debt_ratio"], top_n=2)
    assert result[0]["feature"] == "income"
    assert result[1]["feature"] == "debt_ratio"


def test_top_feature_importances_reads_linear_model_coefficients() -> None:
    class FakeLinearModel:
        coef_ = [[-0.2, 0.9, 0.1]]

    class FakePipeline:
        named_steps = {"scaler": object(), "model": FakeLinearModel()}

    result = top_feature_importances(FakePipeline(), ["age", "income", "debt_ratio"], top_n=1)
    assert result[0]["feature"] == "income"
