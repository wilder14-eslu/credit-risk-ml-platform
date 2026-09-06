"""Model explanation entry point.

Wraps SHAP's TreeExplainer to turn a single applicant's feature vector into
a ranked, human-readable list of "why" the model scored them the way it
did. This is what the API's `top_factors` field and the Streamlit demo's
"factores que más influyeron" chart are built from.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def explain_features(model: Any, features: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    """Return the ``top_n`` features that most influenced a single prediction.

    ``features`` must be a single-row DataFrame with the canonical feature
    columns (see `src.feature_store.features.FEATURE_NAMES`), in the same
    order the model was trained on.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    # Binary XGBoost classifiers return a single (n_samples, n_features)
    # array; some SHAP/model combinations return a list per class instead.
    values = shap_values[1] if isinstance(shap_values, list) else shap_values

    contributions = [
        {
            "feature": column,
            "value": float(features.iloc[0][column]),
            "impact": float(values[0][index]),
        }
        for index, column in enumerate(features.columns)
    ]
    contributions.sort(key=lambda item: abs(item["impact"]), reverse=True)
    return contributions[:top_n]
