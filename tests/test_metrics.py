import numpy as np

from src.ml.metrics import compute_classification_metrics, diagnose_fit, ks_statistic


def _synthetic_labels_and_scores(n: int = 200, separation: float = 3.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    noise = rng.normal(size=n)
    scores = 1 / (1 + np.exp(-(separation * y_true - separation / 2 + noise)))
    return y_true, scores


def test_ks_statistic_is_zero_when_scores_do_not_separate_classes() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert ks_statistic(y_true, scores) == 0.0


def test_ks_statistic_is_high_when_scores_perfectly_separate_classes() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert ks_statistic(y_true, scores) == 1.0


def test_compute_classification_metrics_has_the_full_panel() -> None:
    y_true, scores = _synthetic_labels_and_scores()
    metrics = compute_classification_metrics(y_true, scores)
    for key in (
        "roc_auc",
        "pr_auc",
        "gini",
        "ks_statistic",
        "precision",
        "recall",
        "f1",
        "log_loss",
        "brier_score",
        "true_positives",
        "false_positives",
        "true_negatives",
        "false_negatives",
    ):
        assert key in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["gini"] == 2 * metrics["roc_auc"] - 1


def test_diagnose_fit_flags_overfitting() -> None:
    train_metrics = {"roc_auc": 0.95}
    test_metrics = {"roc_auc": 0.70}
    diagnosis = diagnose_fit(train_metrics, test_metrics)
    assert diagnosis["label"] == "sobreajuste"


def test_diagnose_fit_flags_underfitting() -> None:
    train_metrics = {"roc_auc": 0.55}
    test_metrics = {"roc_auc": 0.52}
    diagnosis = diagnose_fit(train_metrics, test_metrics)
    assert diagnosis["label"] == "subajuste"


def test_diagnose_fit_flags_good_fit() -> None:
    train_metrics = {"roc_auc": 0.82}
    test_metrics = {"roc_auc": 0.80}
    diagnosis = diagnose_fit(train_metrics, test_metrics)
    assert diagnosis["label"] == "buen_ajuste"
