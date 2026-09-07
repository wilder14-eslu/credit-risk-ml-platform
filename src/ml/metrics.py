"""Classification quality metrics beyond ROC-AUC, plus over/underfitting
diagnosis from comparing train vs test performance.

Used by `src.ml.train.train_model` (which logs train+test metrics side by
side so every run can be checked for generalization, not just accuracy) and
by `src.ml.report` (to build the human-readable training report) and
`src.ml.benchmark` (to compare candidate algorithms on equal footing).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ks_statistic(y_true: Any, probabilities: Any) -> float:
    """Kolmogorov-Smirnov statistic: the maximum separation between the
    cumulative distributions of scores for the positive and negative
    classes. Standard in credit scoring alongside ROC-AUC/Gini to judge how
    well the model ranks good vs bad payers."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(probabilities)
    sorted_scores = probabilities[order]
    sorted_labels = y_true[order]

    cum_pos = np.cumsum(sorted_labels) / n_pos
    cum_neg = np.cumsum(1 - sorted_labels) / n_neg

    # Only compare cumulative distributions at the boundary of each group of
    # tied scores, so rows sharing an identical score never register a
    # spurious gap purely from how the (arbitrary, stable) sort ordered
    # them internally.
    is_group_boundary = np.append(sorted_scores[:-1] != sorted_scores[1:], True)
    return float(np.max(np.abs(cum_pos - cum_neg)[is_group_boundary]))


def compute_classification_metrics(
    y_true: Any, probabilities: Any, threshold: float = 0.5
) -> dict[str, float]:
    """Full metric panel for one split (train or test): ranking quality
    (ROC-AUC, PR-AUC, Gini, KS), decision quality at ``threshold``
    (precision/recall/F1/confusion matrix) and calibration (log loss,
    Brier score -- how close predicted probabilities are to reality, not
    just how well they rank applicants)."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, probabilities))
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return {
        "roc_auc": roc_auc,
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "gini": 2 * roc_auc - 1,
        "ks_statistic": ks_statistic(y_true, probabilities),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def diagnose_fit(
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    overfit_gap_threshold: float = 0.07,
    underfit_auc_threshold: float = 0.65,
) -> dict[str, Any]:
    """Compare train vs test ROC-AUC to flag over/underfitting.

    - Sobreajuste (overfitting): the model scores much better on data it
      memorized (train) than on unseen data (test) -- a large gap.
    - Subajuste (underfitting): the model scores poorly on *both* splits --
      it hasn't captured the signal in the data at all, regardless of gap.
    - Buen ajuste: neither of the above.
    """
    gap = train_metrics["roc_auc"] - test_metrics["roc_auc"]

    if (
        test_metrics["roc_auc"] < underfit_auc_threshold
        and train_metrics["roc_auc"] < underfit_auc_threshold
    ):
        label = "subajuste"
        explanation = (
            f"ROC-AUC bajo tanto en entrenamiento ({train_metrics['roc_auc']:.3f}) como en "
            f"prueba ({test_metrics['roc_auc']:.3f}): el modelo no está capturando la señal "
            "de los datos. Considera features adicionales, menos regularización o un modelo "
            "más expresivo."
        )
    elif gap > overfit_gap_threshold:
        label = "sobreajuste"
        explanation = (
            f"El ROC-AUC de entrenamiento ({train_metrics['roc_auc']:.3f}) supera al de "
            f"prueba ({test_metrics['roc_auc']:.3f}) por {gap:.3f}: el modelo memoriza el "
            "set de entrenamiento más de lo que generaliza. Considera más regularización, "
            "menos profundidad/estimadores o más datos."
        )
    else:
        label = "buen_ajuste"
        explanation = (
            f"ROC-AUC de entrenamiento ({train_metrics['roc_auc']:.3f}) y de prueba "
            f"({test_metrics['roc_auc']:.3f}) están alineados (brecha de {gap:.3f}): el "
            "modelo generaliza razonablemente bien."
        )

    return {"label": label, "gap": round(gap, 4), "explanation": explanation}
