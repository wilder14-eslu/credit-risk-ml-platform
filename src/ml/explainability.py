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


# --------------------------------------------------------------------------
# Global, dependence and interaction explanations
# --------------------------------------------------------------------------


def _positive_class(values: Any) -> Any:
    import numpy as np

    if isinstance(values, list):
        return np.asarray(values[1])
    arr = np.asarray(values)
    return arr[..., 1] if arr.ndim == 3 else arr


def shap_sample(
    model: Any, features: pd.DataFrame, n: int = 3000, seed: int = 42
) -> tuple[pd.DataFrame, Any]:
    """SHAP (log-odds) sobre una muestra fija. Devuelve (muestra, matriz de valores)."""
    import shap

    sample = features.sample(n=min(n, len(features)), random_state=seed)
    values = _positive_class(shap.TreeExplainer(model).shap_values(sample))
    return sample, values


def global_importance(values: Any, columns: list[str]) -> list[dict[str, Any]]:
    """Importancia global = media del |SHAP| por variable, con signo de la asociación.

    ``direction`` es la correlación entre el valor de la variable y su SHAP:
    una **asociación** dentro del modelo, no un efecto causal.
    """
    import numpy as np

    values = np.asarray(values)
    rows = [
        {"feature": column, "mean_abs_shap": float(np.abs(values[:, j]).mean())}
        for j, column in enumerate(columns)
    ]
    return sorted(rows, key=lambda r: r["mean_abs_shap"], reverse=True)


def association_direction(sample: pd.DataFrame, values: Any) -> dict[str, float]:
    """Correlación de Spearman entre valor de la variable y su SHAP (asociación)."""
    import numpy as np
    from scipy import stats

    values = np.asarray(values)
    out = {}
    for j, column in enumerate(sample.columns):
        x = sample[column].to_numpy(dtype=float)
        if np.nanstd(x) == 0 or np.std(values[:, j]) == 0:
            out[column] = 0.0
            continue
        rho = stats.spearmanr(x, values[:, j], nan_policy="omit").statistic
        out[column] = float(rho)
    return out


def interaction_strength(
    model: Any, features: pd.DataFrame, n: int = 800, seed: int = 42
) -> pd.DataFrame | None:
    """Matriz de media del |valor de interacción SHAP| (off-diagonal). ``None`` si el
    modelo no soporta interacciones exactas."""
    import numpy as np
    import shap

    sample = features.sample(n=min(n, len(features)), random_state=seed)
    try:
        raw = shap.TreeExplainer(model).shap_interaction_values(sample)
        inter = np.asarray(raw[1] if isinstance(raw, list) else raw)
    except Exception:
        try:
            from catboost import Pool

            inter = np.asarray(
                model.get_feature_importance(Pool(sample), type="ShapInteractionValues")
            )[:, :-1, :-1]
        except Exception:
            return None
    matrix = np.abs(inter).mean(axis=0)
    return pd.DataFrame(matrix, index=features.columns, columns=features.columns)
