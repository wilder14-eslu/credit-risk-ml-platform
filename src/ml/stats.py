"""Statistical tools for rigorous model evaluation.

Pure functions (NumPy/SciPy only, no plotting, no I/O) so they are cheap to
unit-test and reusable from notebooks, the evaluation job and CI:

- ``delong_roc_test`` / ``delong_auc_ci``: DeLong et al. (1988) test for two
  correlated ROC-AUCs measured on the same observations, using the fast
  O(n log n) algorithm of Sun & Xu (2014).
- ``bootstrap_ci``: percentile bootstrap confidence interval for any metric.
- ``corrected_resampled_ttest``: Nadeau & Bengio (2003) corrected t-test to
  compare two models across (repeated) k-fold cross-validation. The naive
  paired t-test is anti-conservative there because training folds overlap.
- ``t_confidence_interval``: Student-t interval for a mean over folds.
- ``expected_calibration_error`` / ``calibration_slope_intercept``:
  calibration diagnostics (not just ranking).
- ``decile_table``: the gains/lift/KS table used in credit-scoring reports.
- ``threshold_analysis``: precision/recall/F1/expected cost per threshold.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# DeLong
# --------------------------------------------------------------------------


def _fast_delong(predictions_sorted: np.ndarray, n_positives: int) -> tuple[np.ndarray, np.ndarray]:
    """Sun & Xu (2014). ``predictions_sorted`` is (k models, n) with all
    positive observations first. Returns (aucs, covariance matrix)."""
    m = n_positives
    n = predictions_sorted.shape[1] - m
    positives = predictions_sorted[:, :m]
    negatives = predictions_sorted[:, m:]
    k = predictions_sorted.shape[0]

    tx = np.vstack([stats.rankdata(positives[r]) for r in range(k)])
    ty = np.vstack([stats.rankdata(negatives[r]) for r in range(k)])
    tz = np.vstack([stats.rankdata(predictions_sorted[r]) for r in range(k)])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def _sort_positives_first(y_true: Any, *scores: Any) -> tuple[np.ndarray, int]:
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true, kind="stable")
    matrix = np.vstack([np.asarray(score, dtype=float)[order] for score in scores])
    return matrix, int(y_true.sum())


def delong_auc_ci(y_true: Any, scores: Any, alpha: float = 0.05) -> dict[str, float]:
    """ROC-AUC with its DeLong standard error and (1 - alpha) CI."""
    matrix, m = _sort_positives_first(y_true, scores)
    aucs, cov = _fast_delong(matrix, m)
    auc = float(aucs[0])
    se = float(np.sqrt(cov[0, 0]))
    z = stats.norm.ppf(1 - alpha / 2)
    return {"auc": auc, "se": se, "ci_low": auc - z * se, "ci_high": min(auc + z * se, 1.0)}


def delong_roc_test(y_true: Any, scores_a: Any, scores_b: Any) -> dict[str, float]:
    """Two-sided DeLong test of H0: AUC_a == AUC_b (same observations)."""
    matrix, m = _sort_positives_first(y_true, scores_a, scores_b)
    aucs, cov = _fast_delong(matrix, m)
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ cov @ contrast)
    diff = float(aucs[0] - aucs[1])
    if variance <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff,
                "z": 0.0, "p_value": 1.0}
    z = diff / np.sqrt(variance)
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "diff": diff,
        "z": float(z),
        "p_value": float(2 * stats.norm.sf(abs(z))),
    }


# --------------------------------------------------------------------------
# Resampling-based inference
# --------------------------------------------------------------------------


def bootstrap_ci(
    y_true: Any,
    scores: Any,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, float]:
    """Percentile bootstrap CI (resampling observations with replacement).

    Resamples that end up with a single class are skipped (possible only on
    tiny samples).
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_sample = y_true[idx]
        if y_sample.min() == y_sample.max():
            continue
        values.append(metric(y_sample, scores[idx]))
    values_arr = np.asarray(values)
    return {
        "estimate": float(metric(y_true, scores)),
        "ci_low": float(np.quantile(values_arr, alpha / 2)),
        "ci_high": float(np.quantile(values_arr, 1 - alpha / 2)),
        "se": float(values_arr.std(ddof=1)),
    }


def t_confidence_interval(
    values: Any, alpha: float = 0.05, test_train_ratio: float | None = None
) -> dict[str, float]:
    """Mean, sample std and Student-t (1 - alpha) CI across CV folds.

    With ``test_train_ratio`` (n_val / n_fit) the variance of the mean uses
    the Nadeau-Bengio correction ``(1/J + n_val/n_fit) * s^2`` instead of
    ``s^2 / J``: fold scores are positively correlated (shared training
    rows), so the naive interval is too narrow.
    """
    values = np.asarray(values, dtype=float)
    j = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if j > 1 else 0.0
    factor = 1.0 / j if test_train_ratio is None else 1.0 / j + test_train_ratio
    half = float(stats.t.ppf(1 - alpha / 2, j - 1) * std * np.sqrt(factor)) if j > 1 else 0.0
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half}


def corrected_resampled_ttest(
    scores_a: Any, scores_b: Any, n_train: int, n_test: int
) -> dict[str, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test.

    ``scores_a``/``scores_b`` are per-fold metrics for two models evaluated
    on the *same* folds. The variance of the fold differences is inflated by
    ``(1/J + n_test/n_train)`` to account for overlapping training sets.
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    j = len(diff)
    mean = float(diff.mean())
    var = float(diff.var(ddof=1))
    corrected_var = (1.0 / j + n_test / n_train) * var
    if corrected_var <= 0:
        return {"diff_mean": mean, "t": 0.0, "p_value": 1.0, "df": j - 1}
    t = mean / np.sqrt(corrected_var)
    return {
        "diff_mean": mean,
        "t": float(t),
        "p_value": float(2 * stats.t.sf(abs(t), j - 1)),
        "df": j - 1,
    }


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def expected_calibration_error(y_true: Any, probabilities: Any, n_bins: int = 10) -> float:
    """ECE with equal-frequency (quantile) bins: sum_b w_b |obs_b - pred_b|."""
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(probabilities)
    bins = np.array_split(order, n_bins)
    n = len(y_true)
    return float(
        sum(len(b) / n * abs(y_true[b].mean() - probabilities[b].mean()) for b in bins if len(b))
    )


def calibration_slope_intercept(y_true: Any, probabilities: Any) -> dict[str, float]:
    """Logistic recalibration ``logit(P(y=1)) = a + b * logit(p)``.

    Perfect calibration: slope b = 1 and intercept a = 0. b < 1 means
    predictions are too extreme (over-confident), b > 1 too timid.
    ``calibration_in_the_large`` = observed rate - mean predicted rate.
    """
    from sklearn.linear_model import LogisticRegression

    y_true = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y_true)
    return {
        "slope": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
        "calibration_in_the_large": float(y_true.mean() - p.mean()),
    }


# --------------------------------------------------------------------------
# Business views
# --------------------------------------------------------------------------


def decile_table(y_true: Any, probabilities: Any, n_bins: int = 10) -> list[dict[str, float]]:
    """Gains/lift table: rows sorted from highest to lowest predicted risk."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(-probabilities, kind="stable")
    groups = np.array_split(order, n_bins)
    total_bad = y_true.sum()
    total_good = len(y_true) - total_bad
    base_rate = total_bad / len(y_true)

    rows, cum_bad, cum_good, cum_n = [], 0, 0, 0
    for decile, idx in enumerate(groups, start=1):
        bad = int(y_true[idx].sum())
        n = len(idx)
        cum_bad += bad
        cum_good += n - bad
        cum_n += n
        rows.append(
            {
                "decile": decile,
                "n": n,
                "min_score": float(probabilities[idx].min()),
                "max_score": float(probabilities[idx].max()),
                "bad": bad,
                "bad_rate": bad / n,
                "cum_bad_capture": cum_bad / total_bad,
                "lift": (bad / n) / base_rate,
                "cum_lift": (cum_bad / cum_n) / base_rate,
                "ks": abs(cum_bad / total_bad - cum_good / total_good),
            }
        )
    return rows


def threshold_analysis(
    y_true: Any,
    probabilities: Any,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    thresholds: Any = None,
) -> dict[str, Any]:
    """Metrics per threshold and the thresholds that optimise F1, Youden's J
    and expected misclassification cost (``cost_fn`` per missed default,
    ``cost_fp`` per good applicant wrongly rejected).

    For a perfectly calibrated model the Bayes-optimal cost threshold is
    ``cost_fp / (cost_fp + cost_fn)``; the empirical optimum is reported
    alongside so the two can be compared.
    """
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 0.91, 0.005), 3)
    thresholds = np.asarray(thresholds, dtype=float)

    positives = y_true.sum()
    negatives = len(y_true) - positives
    curve = []
    for t in thresholds:
        pred = probabilities >= t
        tp = int((pred & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        fn = int(positives - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        specificity = 1 - fp / negatives if negatives else 0.0
        curve.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "youden_j": recall + specificity - 1,
                "rejection_rate": float(pred.mean()),
                "expected_cost_per_applicant": (cost_fn * fn + cost_fp * fp) / len(y_true),
            }
        )

    def _best(key: str, maximize: bool = True) -> dict[str, float]:
        chooser = max if maximize else min
        return chooser(curve, key=lambda row: row[key])

    return {
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "bayes_threshold": cost_fp / (cost_fp + cost_fn),
        "best_f1": _best("f1"),
        "best_youden": _best("youden_j"),
        "min_cost": _best("expected_cost_per_applicant", maximize=False),
        "curve": curve,
    }


def approval_strategy_curve(
    y_true: Any, probabilities: Any, approval_rates: Any = None
) -> list[dict[str, float]]:
    """Bad rate of the approved portfolio when approving the lowest-risk X%."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if approval_rates is None:
        approval_rates = np.round(np.arange(0.30, 1.0001, 0.01), 2)
    order = np.argsort(probabilities, kind="stable")
    sorted_y = y_true[order]
    cum_bad = np.cumsum(sorted_y)
    n = len(y_true)
    rows = []
    for rate in approval_rates:
        k = max(int(round(rate * n)), 1)
        rows.append(
            {
                "approval_rate": float(rate),
                "bad_rate_approved": float(cum_bad[k - 1] / k),
                "score_cutoff": float(probabilities[order][k - 1]),
            }
        )
    return rows
