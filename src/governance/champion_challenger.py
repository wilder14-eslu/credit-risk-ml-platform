"""Marco champion vs challenger sobre el mismo holdout.

Produce una **recomendación para revisión humana**, nunca una promoción:
``auto_promote`` es siempre ``False``. Un challenger solo es "elegible" si su
mejora es estadísticamente significativa *y* supera el margen práctico, sin
empeorar la calibración más allá de una tolerancia.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.inference.effect_sizes import DEFAULT_PRACTICAL_MARGIN
from src.ml import stats as st


def compare_champion_challenger(
    y_true: Any,
    champion_proba: Any,
    challenger_proba: Any,
    champion_name: str = "champion",
    challenger_name: str = "challenger",
    practical_margin: float = DEFAULT_PRACTICAL_MARGIN,
    max_calibration_slope_error_worsening: float = 0.05,
    alpha: float = 0.05,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p_champ = np.asarray(champion_proba, dtype=float)
    p_chal = np.asarray(challenger_proba, dtype=float)
    delong = st.delong_roc_test(y, p_chal, p_champ)  # diff = challenger - champion
    cal_champ = st.calibration_slope_intercept(y, p_champ)
    cal_chal = st.calibration_slope_intercept(y, p_chal)
    ece_champ = st.expected_calibration_error(y, p_champ)
    ece_chal = st.expected_calibration_error(y, p_chal)

    slope_err_champ = abs(cal_champ["slope"] - 1)
    slope_err_chal = abs(cal_chal["slope"] - 1)
    calibration_ok = (slope_err_chal - slope_err_champ) <= max_calibration_slope_error_worsening
    significant = delong["p_value"] < alpha and delong["diff"] > 0
    material = delong["diff"] >= practical_margin
    eligible = bool(significant and material and calibration_ok)

    if eligible:
        recommendation = "Challenger elegible para revisión humana (significativo, material y calibrado)."
    elif significant and not material:
        recommendation = (
            "Mantener el champion: la mejora es estadísticamente significativa pero menor "
            f"que el margen práctico ({practical_margin})."
        )
    elif not significant:
        recommendation = "Mantener el champion: no hay mejora estadísticamente significativa."
    else:
        recommendation = "Mantener el champion: el challenger no pasa el control de calibración."

    return {
        "champion": champion_name,
        "challenger": challenger_name,
        "auc_champion": delong["auc_b"],
        "auc_challenger": delong["auc_a"],
        "auc_difference": delong["diff"],
        "p_value_delong": delong["p_value"],
        "practical_margin": practical_margin,
        "calibration": {
            "slope_champion": cal_champ["slope"], "slope_challenger": cal_chal["slope"],
            "ece_champion": ece_champ, "ece_challenger": ece_chal,
            "calibration_ok": bool(calibration_ok),
        },
        "challenger_eligible_for_review": eligible,
        "recommendation": recommendation,
        "auto_promote": False,
        "requires_human_approval": True,
    }
