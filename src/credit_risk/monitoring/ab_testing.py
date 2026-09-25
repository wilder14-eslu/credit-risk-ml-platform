"""A/B testing champion vs challenger sobre tráfico real etiquetado.

- Asignación determinista y "sticky" por hash del `applicant_id` (el mismo
  solicitante siempre ve la misma variante; reproducible en gateway y jobs).
- Decisión con bootstrap pareado de la diferencia de AUC, más test de dos
  proporciones sobre la tasa de default entre los aprobados (impacto negocio).
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from credit_risk.monitoring.concept_drift import two_proportion_ztest


def assign_variant(applicant_id: str, challenger_traffic: float, salt: str = "ab-v1") -> str:
    if challenger_traffic <= 0:
        return "champion"
    digest = hashlib.sha256(f"{salt}:{applicant_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "challenger" if bucket < challenger_traffic else "champion"


def bootstrap_auc_diff(
    y_a: np.ndarray, p_a: np.ndarray, y_b: np.ndarray, p_b: np.ndarray, iterations: int = 500, seed: int = 42
) -> dict[str, float]:
    """IC 95% de AUC(B) - AUC(A) remuestreando cada brazo de forma independiente."""
    rng = np.random.default_rng(seed)
    y_a, p_a, y_b, p_b = map(np.asarray, (y_a, p_a, y_b, p_b))
    diffs = []
    for _ in range(iterations):
        ia = rng.integers(0, len(y_a), len(y_a))
        ib = rng.integers(0, len(y_b), len(y_b))
        if len(np.unique(y_a[ia])) < 2 or len(np.unique(y_b[ib])) < 2:
            continue
        diffs.append(roc_auc_score(y_b[ib], p_b[ib]) - roc_auc_score(y_a[ia], p_a[ia]))
    diffs = np.asarray(diffs)
    point = roc_auc_score(y_b, p_b) - roc_auc_score(y_a, p_a)
    return {
        "auc_diff": float(point),
        "ci_low": float(np.quantile(diffs, 0.025)) if diffs.size else float("nan"),
        "ci_high": float(np.quantile(diffs, 0.975)) if diffs.size else float("nan"),
        "p_value": float(np.mean(diffs <= 0)) if diffs.size else float("nan"),
    }


def evaluate_ab_test(
    champion: dict[str, np.ndarray],
    challenger: dict[str, np.ndarray],
    cfg: dict[str, Any],
    days_running: float = 0.0,
) -> dict[str, Any]:
    """Cada brazo: {'y': etiquetas, 'p': probas, 'approved': bool}. Devuelve la decisión."""
    n_a, n_b = len(champion["y"]), len(challenger["y"])
    result: dict[str, Any] = {"n_champion": n_a, "n_challenger": n_b, "days_running": days_running}
    if min(n_a, n_b) < cfg["min_labeled_per_arm"]:
        result.update(decision="continuar", reason="muestra insuficiente por brazo")
        return result

    auc = bootstrap_auc_diff(
        champion["y"], champion["p"], challenger["y"], challenger["p"], cfg["bootstrap_iterations"]
    )
    result.update(auc)
    result["auc_champion"] = float(roc_auc_score(champion["y"], champion["p"]))
    result["auc_challenger"] = float(roc_auc_score(challenger["y"], challenger["p"]))

    a_ok, b_ok = champion["approved"].astype(bool), challenger["approved"].astype(bool)
    bad_a, bad_b = int(champion["y"][a_ok].sum()), int(challenger["y"][b_ok].sum())
    z, p = two_proportion_ztest(bad_b, int(b_ok.sum()), bad_a, int(a_ok.sum()))
    result.update(
        approval_rate_champion=float(a_ok.mean()),
        approval_rate_challenger=float(b_ok.mean()),
        bad_rate_approved_champion=bad_a / max(int(a_ok.sum()), 1),
        bad_rate_approved_challenger=bad_b / max(int(b_ok.sum()), 1),
        bad_rate_z=z,
        bad_rate_pvalue=p,
    )

    alpha = cfg["alpha"]
    challenger_better = auc["ci_low"] > 0 and auc["p_value"] < alpha
    challenger_worse = auc["ci_high"] < 0
    business_harm = (p < alpha) and (z > 0)  # más morosos aprobados con el challenger
    if challenger_better and not business_harm:
        result.update(decision="promover", reason="challenger superior con significancia")
    elif challenger_worse or business_harm:
        result.update(decision="detener", reason="challenger inferior o dañino para el negocio")
    elif days_running >= cfg["max_test_days"]:
        result.update(decision="detener", reason="sin diferencia significativa al cierre del test")
    else:
        result.update(decision="continuar", reason="aún sin significancia")
    return result
