"""Pipeline completo y reproducible: del CSV crudo a ``reports/``.

Una sola ejecución (``python -m src.pipelines.full_report``) produce:

* ``reports/metrics.json`` + ``evaluation_report.md``: protocolo de evaluación
  (CV repetida, holdout aislado, DeLong, Nadeau-Bengio, calibración, deciles).
* ``reports/results.json``: todo lo demás (scorecard, comparaciones con Holm,
  costos por ratio, cartera, estabilidad, equidad, stress, monitoreo,
  explicabilidad, champion-challenger, limitaciones).
* ``reports/figures/*.png``: todas las figuras.
* ``reports/MODEL_CARD.md`` y ``reports/DATA_CARD.md``.

La CV anidada con Optuna (lenta) vive en ``python -m src.ml.nested_cv`` y,
si ``reports/nested_cv.json`` existe, se incorpora al reporte.

Reglas: el holdout se usa solo para medir; ningún umbral ni hiperparámetro se
elige con él; nada se inventa (sin LGD/EAD, sin fechas, sin rechazados).
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from src.data_pipeline.validate import clean_out_of_range_rows, validate_input_data
from src.feature_store.features import load_feature_schema
from src.governance.cards import render_data_card, render_model_card
from src.governance.champion_challenger import compare_champion_challenger
from src.governance.provenance import collect_provenance
from src.inference.comparison import pairwise_cv_comparison, pairwise_delong_comparison
from src.ml import stats as st
from src.ml.evaluation import DISPLAY_NAMES, _fold_metrics, run_evaluation
from src.ml.explainability import (
    association_direction,
    explain_features,
    global_importance,
    interaction_strength,
    shap_sample,
)
from src.ml.stability import seed_stability, summarize_cv_folds
from src.monitoring.business import business_drift, score_drift
from src.monitoring.decision import build_retraining_recommendation
from src.monitoring.drift import compute_feature_drift, save_reference_distribution
from src.monitoring.performance import performance_drift, performance_panel
from src.monitoring.quality import data_quality_report
from src.risk import figures as fig
from src.risk.cost_analysis import cost_ratio_table
from src.risk.expected_loss import LgdEadUnavailableError
from src.risk.fairness import age_groups, group_fairness_report
from src.risk.oot import OOT_UNAVAILABLE_MESSAGE, find_temporal_column
from src.risk.portfolio import (
    approval_scenarios,
    gains_curve,
    risk_concentration,
    segment_performance,
)
from src.risk.scorecard import CreditScorecard
from src.risk.stress import apply_scenario, load_scenarios, run_stress_test

logger = logging.getLogger(__name__)

DATASET = Path("DATA/GiveMeSomeCredit/cs-training.csv")
SEED = 42
SENTINEL_FEATURES = {
    "number_of_time_30_59_days_past_due": (96.0, 98.0),
    "number_of_time_60_89_days_past_due": (96.0, 98.0),
    "number_of_times_90_days_late": (96.0, 98.0),
}
NAMES = {**DISPLAY_NAMES, "scorecard": "Scorecard WoE"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def profile_dataset(raw: pd.DataFrame) -> dict[str, Any]:
    """Perfil descriptivo del dataset limpio (todo calculado, nada supuesto)."""
    schema = load_feature_schema()
    features = {}
    for name, definition in schema["features"].items():
        col = definition["source"] if definition["source"] in raw.columns else name
        values = pd.to_numeric(raw[col], errors="coerce")
        features[name] = {
            "source_column": col,
            "missing_rate": float(values.isna().mean()),
            "min": float(values.min()),
            "median": float(values.median()),
            "p99": float(values.quantile(0.99)),
            "max": float(values.max()),
        }
    target = schema.get("target_source", schema["target"])
    sentinels = {}
    for name, codes in SENTINEL_FEATURES.items():
        col = schema["features"][name]["source"]
        mask = raw[col].isin(codes)
        sentinels[name] = {
            "rows": int(mask.sum()),
            "default_rate_in_rows": float(raw.loc[mask, target].mean()) if mask.any() else None,
        }
    return {
        "rows": len(raw),
        "n_features": len(features),
        "default_prevalence": float(raw[target].mean()),
        "defaults": int(raw[target].sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "features": features,
        "sentinel_codes_96_98": sentinels,
        "temporal_variable": find_temporal_column(raw),
        "target_source": target,
    }


def _segments(x: pd.DataFrame) -> dict[str, pd.Series]:
    income = x["monthly_income"]
    quartiles, edges = pd.qcut(income, 4, labels=False, retbins=True, duplicates="drop")
    names = [f"Q{i + 1} (ingreso {edges[i]:,.0f}-{edges[i + 1]:,.0f})" for i in range(len(edges) - 1)]
    income_band = quartiles.map(dict(enumerate(names))).astype(object).where(
        income.notna(), "Ingreso faltante"
    )
    util = pd.cut(
        x["revolving_utilization_unsecured"], [-0.001, 0.3, 0.7, 1.0, np.inf],
        labels=["1: <=30 %", "2: 30-70 %", "3: 70-100 %", "4: >100 %"],
    ).astype(str)
    return {
        "age_group": age_groups(x["age"]).astype(str),
        "income_quartile": income_band,
        "revolving_utilization": util,
    }


def _scorecard_cv(x_dev: pd.DataFrame, y_dev: pd.Series, n_splits: int, n_repeats: int,
                  special: dict[str, tuple[float, ...]]) -> list[dict[str, float]]:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    rows = []
    y = y_dev.to_numpy()
    for tr, va in cv.split(x_dev, y):
        card = CreditScorecard(special_values=special).fit(x_dev.iloc[tr], y[tr])
        rows.append(_fold_metrics(y[va], card.predict_pd(x_dev.iloc[va])))
    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def run(
    data_path: Path | str = DATASET,
    output_dir: Path | str = "reports",
    n_splits: int = 5,
    n_repeats: int = 3,
    n_boot: int = 1000,
    seeds: tuple[int, ...] = tuple(range(10)),
) -> dict[str, Any]:
    out = Path(output_dir)
    figs = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    raw = clean_out_of_range_rows(pd.read_csv(data_path, index_col=0))
    validate_input_data(raw, require_target=True)
    data_profile = profile_dataset(raw)

    # 1. Protocolo de evaluación (CV repetida + holdout aislado) ------------
    logger.info("1/9 Evaluación base (CV %dx%d)...", n_splits, n_repeats)
    result, art = run_evaluation(
        raw, output_dir=out, figures_dir=figs, n_splits=n_splits, n_repeats=n_repeats,
        n_boot=n_boot, return_artifacts=True,
    )
    champion = result["champion"]
    y_test = art["y_test"].to_numpy()
    y_dev = art["y_dev"]
    proba = dict(art["test_proba"])
    threshold = result["threshold"]["selected_threshold"]

    # 2. Scorecard tradicional (WoE/IV) ------------------------------------
    logger.info("2/9 Scorecard WoE...")
    card = CreditScorecard(special_values=SENTINEL_FEATURES).fit(art["x_dev"], y_dev)
    proba["scorecard"] = card.predict_pd(art["x_test"])
    scores_holdout = card.score(art["x_test"])
    card_cv_folds = _scorecard_cv(art["x_dev"], y_dev, n_splits, n_repeats, SENTINEL_FEATURES)
    per_fold = {**art["per_fold"], "scorecard": card_cv_folds}
    scorecard = {
        "parameters": {"pdo": card.pdo, "base_score": card.base_score, "base_odds": card.base_odds,
                       "max_bins": card.max_bins, "min_share": card.min_share},
        "iv_table": card.iv_table(),
        "points_table": card.points_table(),
        "holdout": {
            **_fold_metrics(y_test, proba["scorecard"]),
            "auc_delong": st.delong_auc_ci(y_test, proba["scorecard"]),
            "delong_vs_champion": st.delong_roc_test(y_test, proba[champion], proba["scorecard"]),
            "calibration": st.calibration_slope_intercept(y_test, proba["scorecard"]),
        },
        "cv_roc_auc": {
            "mean": float(np.mean([f["roc_auc"] for f in card_cv_folds])),
            "std": float(np.std([f["roc_auc"] for f in card_cv_folds], ddof=1)),
        },
        "decile_table": st.decile_table(y_test, proba["scorecard"]),
        "score_range_holdout": [float(scores_holdout.min()), float(scores_holdout.max())],
    }
    rates = [d["bad_rate"] for d in scorecard["decile_table"]]
    scorecard["bad_rate_monotone_by_decile"] = bool(
        all(a >= b for a, b in pairwise(rates))
    )

    # 3. Comparaciones pareadas con Holm ------------------------------------
    logger.info("3/9 Comparaciones pareadas (Holm)...")
    all_models = [*art["usable"], "scorecard"]
    cv_auc = {m: [f["roc_auc"] for f in per_fold[m]] for m in all_models}
    comparisons = {
        "cv_nadeau_bengio_holm": pairwise_cv_comparison(cv_auc, art["n_fit"], art["n_val"]),
        "holdout_delong_holm": pairwise_delong_comparison(
            y_test, {m: proba[m] for m in all_models}
        ),
        "practical_margin_auc": 0.005,
    }

    # 4. Umbral por costos (varios FN:FP) -----------------------------------
    logger.info("4/9 Umbrales por costo...")
    cost_ratios = cost_ratio_table(y_dev, art["oof"][champion], y_test, proba[champion])

    # 5. Cartera y segmentos -------------------------------------------------
    logger.info("5/9 Cartera y estabilidad...")
    curves = {m: gains_curve(y_test, proba[m]) for m in all_models}
    portfolio = {
        "champion": champion,
        "concentration": risk_concentration(y_test, proba[champion]),
        "approval_scenarios": approval_scenarios(y_test, proba[champion]),
        "segments": {
            name: segment_performance(y_test, proba[champion], seg, n_boot=300)
            for name, seg in _segments(art["x_test"]).items()
        },
        "note": "Escenarios analíticos de tasa de aprobación; no son recomendaciones de crédito.",
    }

    # 6. Estabilidad ---------------------------------------------------------
    full_x = pd.concat([art["x_dev"], art["x_test"]])
    full_y = pd.concat([y_dev, art["y_test"]])
    stability = {
        "seed_stability": seed_stability(full_x, full_y, art["usable"], seeds=seeds),
        "cv_folds": summarize_cv_folds(art["per_fold"]),
        "bootstrap_holdout": {
            m: {"roc_auc": result["holdout"][m]["auc_delong"],
                **result["holdout"][m]["bootstrap"]} for m in art["usable"]
        },
    }

    # 7. Equidad y stress ----------------------------------------------------
    logger.info("7/9 Equidad y stress...")
    fairness = group_fairness_report(
        y_test, proba[champion], age_groups(art["x_test"]["age"]), threshold
    )
    stress = run_stress_test(art["fitted"][champion], art["x_test_imp"], threshold)

    # 8. Monitoreo (demostración offline) -----------------------------------
    logger.info("8/9 Monitoreo...")
    monitoring = _monitoring_demo(result, art, proba[champion], y_test, threshold, out)

    # 9. Explicabilidad + champion-challenger -------------------------------
    logger.info("9/9 Explicabilidad...")
    explain, shap_payload = _explainability(art, proba[champion], champion)
    default_algorithm = "xgboost"
    challenger = compare_champion_challenger(
        y_test, proba[default_algorithm], proba[champion],
        champion_name=f"{default_algorithm} (por defecto en src/ml/train.py, usado por la demo)",
        challenger_name=f"{champion} (mayor ROC-AUC en CV)",
    ) if champion != default_algorithm else None

    nested_path = out / "nested_cv.json"
    nested = json.loads(nested_path.read_text(encoding="utf-8")) if nested_path.is_file() else None

    limitations = {
        "out_of_time": {"performed": False, "reason": OOT_UNAVAILABLE_MESSAGE},
        "reject_inference": {
            "performed": False,
            "reason": "El dataset solo contiene solicitantes con desempeño observado; "
                      "no existe población rechazada. El marco está en "
                      "src/risk/reject_inference.py y solo se probó con datos sintéticos.",
        },
        "lgd_ead_expected_loss": {
            "performed": False,
            "reason": "El dataset no tiene LGD, EAD, saldos ni recuperaciones. "
                      "src/risk/expected_loss.py es un marco parametrizable que se "
                      "niega a calcular sin insumos explícitos.",
            "error_type": LgdEadUnavailableError.__name__,
        },
        "fairness_scope": "Solo hay edad; no existen género, estado civil, etnia ni "
                          "geografía, así que la equidad solo puede evaluarse por edad.",
    }

    provenance = collect_provenance(
        data_path,
        {"seed": SEED, "cv": {"n_splits": n_splits, "n_repeats": n_repeats},
         "bootstrap": n_boot, "seeds_stability": list(seeds),
         "cost_fn_fp": [result["threshold"]["cost_fn"], result["threshold"]["cost_fp"]],
         "hyperparameters": result["meta"]["hyperparameters"],
         "scorecard": scorecard["parameters"]},
    )

    results = {
        "provenance": provenance,
        "data": data_profile,
        "champion": champion,
        "selected_threshold": threshold,
        "scorecard": scorecard,
        "comparisons": comparisons,
        "cost_ratios": cost_ratios,
        "portfolio": portfolio,
        "stability": stability,
        "fairness": fairness,
        "stress": stress,
        "monitoring": monitoring,
        "explainability": explain,
        "champion_challenger": challenger,
        "nested_cv": nested,
        "limitations": limitations,
    }

    # Figuras nuevas ----------------------------------------------------------
    fig.scorecard_vs_ml(y_test, {m: proba[m] for m in all_models}, figs / "11_scorecard_vs_ml.png")
    fig.information_value(scorecard["iv_table"], figs / "12_scorecard_information_value.png")
    fig.score_distribution(scores_holdout, y_test, figs / "13_scorecard_score_distribution.png")
    fig.lift_and_concentration(curves, champion, figs / "14_lift_and_cumulative_defaults.png")
    fig.cost_ratio_thresholds(cost_ratios, figs / "15_cost_ratio_thresholds.png")
    fig.model_comparison_forest(
        comparisons["cv_nadeau_bengio_holm"],
        "Comparación pareada en CV (Nadeau-Bengio, ajuste de Holm)",
        figs / "16_model_comparison_cv.png",
    )
    fig.model_comparison_forest(
        comparisons["holdout_delong_holm"],
        "Comparación pareada en holdout (DeLong, ajuste de Holm)",
        figs / "17_model_comparison_holdout.png",
    )
    fig.stability_figure(stability["seed_stability"], figs / "18_seed_stability.png")
    fig.stress_figure(stress, figs / "19_stress_scenarios.png")
    fig.fairness_figure(fairness, figs / "20_fairness_by_age.png")
    fig.psi_monitoring(monitoring["figure_inputs"]["feature_psi"],
                       monitoring["figure_inputs"]["score_psi"], figs / "21_psi_monitoring.png")
    fig.score_drift_figure(monitoring["figure_inputs"]["reference_share"],
                           monitoring["figure_inputs"]["current_shares"],
                           figs / "22_prediction_drift.png")
    sample, values = shap_payload["sample"], shap_payload["values"]
    fig.shap_dependence(sample, values, [r["feature"] for r in explain["global_importance"][:4]],
                        figs / "23_shap_dependence.png")
    fig.shap_local(explain["local_cases_plot"], figs / "24_shap_local.png")
    if shap_payload["interaction"] is not None:
        fig.shap_interaction_heatmap(shap_payload["interaction"], figs / "25_shap_interaction.png")
    if nested:
        fig.nested_cv_figure(nested, figs / "26_nested_cv.png")

    results["monitoring"].pop("figure_inputs")
    explain.pop("local_cases_plot")
    (out / "results.json").write_text(
        json.dumps(_jsonable(results), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "MODEL_CARD.md").write_text(render_model_card(result, results), encoding="utf-8")
    (out / "DATA_CARD.md").write_text(render_data_card(results), encoding="utf-8")
    return results


# --------------------------------------------------------------------------
# Monitoring and explainability blocks
# --------------------------------------------------------------------------


def _monitoring_demo(result: dict[str, Any], art: dict[str, Any], p_holdout: np.ndarray,
                     y_test: np.ndarray, threshold: float, out: Path) -> dict[str, Any]:
    """Demostración offline de cada familia de monitoreo.

    * Referencia = predicciones out-of-fold del desarrollo.
    * Ventana A = holdout aislado (misma población: no se espera deriva).
    * Ventana B = stress moderado SIMULADO sobre las variables del holdout.
      Está rotulada como simulación y no se le inventan outcomes.
    """
    champion = result["champion"]
    reference_scores = art["oof"][champion]
    y_dev = art["y_dev"].to_numpy()
    reference_path = save_reference_distribution(
        art["x_dev_imp"], out / "monitoring_reference_distribution.json"
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    ref_missing = {k: float(v) for k, v in art["x_dev"].isna().mean().items()}

    moderate = load_scenarios()["scenarios"]["moderate_stress"]
    shocked = apply_scenario(art["x_test_imp"], moderate, seed=SEED)
    p_shift = art["fitted"][champion].predict_proba(shocked)[:, 1]

    ref_panel = performance_panel(y_dev, reference_scores)
    cur_panel = performance_panel(y_test, p_holdout)

    windows = {}
    specs = (
        ("holdout (sin cambio esperado)", art["x_test_imp"], p_holdout, y_test, art["x_test"]),
        ("cambio SIMULADO (stress moderado)", shocked, p_shift, None, None),
    )
    feature_psi: dict[str, dict[str, float]] = {}
    score_psi: dict[str, float] = {}
    current_shares: dict[str, list[float]] = {}
    for name, feats, scores, outcomes, raw_feats in specs:
        drift = compute_feature_drift(feats, reference)
        sd = score_drift(reference_scores, scores)
        biz = business_drift(reference_scores, scores, threshold, current_outcomes=outcomes)
        perf = None
        if outcomes is not None:
            perf = {"panel": cur_panel, "drift": performance_drift(ref_panel, cur_panel)}
        quality = data_quality_report(raw_feats, ref_missing) if raw_feats is not None else None
        signals = {
            "feature_drift": drift["drift_detected"],
            "prediction_drift": sd["drift_detected"],
            "business_rejection_drift": biz["rejection_drift"],
            "performance": bool(perf and perf["drift"]["drift_detected"]),
        }
        windows[name] = {
            "data_quality": quality,
            "feature_drift": drift,
            "prediction_drift": {k: v for k, v in sd.items() if "share" not in k},
            "performance": perf,
            "business": biz,
            "recommendation": build_retraining_recommendation(signals),
        }
        feature_psi[name] = drift["psi"]
        score_psi[name] = sd["psi"]
        current_shares[name] = sd["current_share"]
    return {
        "reference": {"source": "predicciones out-of-fold del desarrollo", "panel": ref_panel},
        "windows": windows,
        "note": ("Demostración offline. Este repositorio no tiene tráfico de producción; "
                 "la ventana 'SIMULADO' aplica el escenario de stress moderado a las "
                 "variables del holdout y no tiene outcomes."),
        "figure_inputs": {
            "feature_psi": feature_psi, "score_psi": score_psi,
            "reference_share": score_drift(reference_scores, p_holdout)["reference_share"],
            "current_shares": current_shares,
        },
    }


def _explainability(art: dict[str, Any], p: np.ndarray, champion: str
                    ) -> tuple[dict[str, Any], dict[str, Any]]:
    model = art["fitted"][champion]
    x = art["x_test_imp"]
    sample, values = shap_sample(model, x, n=3000)
    importance = global_importance(values, list(x.columns))
    direction = association_direction(sample, values)
    for row in importance:
        row["spearman_value_vs_shap"] = direction[row["feature"]]
    interaction = interaction_strength(model, x, n=800)

    order = np.argsort(p)
    picks = {
        "Solicitante de mayor riesgo": int(order[-1]),
        "Solicitante de riesgo mediano": int(order[len(order) // 2]),
        "Solicitante de menor riesgo": int(order[0]),
    }
    cases, plot_cases = [], []
    for label, pos in picks.items():
        factors = explain_features(model, x.iloc[[pos]], top_n=5)
        cases.append({"label": label, "probability": float(p[pos]), "factors": factors,
                      "observed_default": int(art["y_test"].iloc[pos])})
        plot_cases.append({"label": label, "probability": float(p[pos]), "factors": factors})
    top_pairs = []
    if interaction is not None:
        m = interaction.to_numpy().copy()
        np.fill_diagonal(m, 0)
        cols = list(interaction.columns)
        pairs = [(cols[i], cols[j], float(m[i, j])) for i in range(len(cols))
                 for j in range(i + 1, len(cols))]
        top_pairs = [
            {"feature_a": a, "feature_b": b, "mean_abs_interaction": v}
            for a, b, v in sorted(pairs, key=lambda t: -t[2])[:5]
        ]
    explain = {
        "model": champion,
        "sample_size": len(sample),
        "global_importance": importance,
        "local_cases": cases,
        "local_cases_plot": plot_cases,
        "top_interactions": top_pairs,
        "interpretation_note": (
            "Los valores SHAP describen cómo usa cada variable el modelo ajustado "
            "(asociación). No muestran que cambiar una variable cambie el riesgo real."
        ),
    }
    return explain, {"sample": sample, "values": values, "interaction": interaction}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default=str(DATASET))
    parser.add_argument("--output", default="reports")
    parser.add_argument("--quick", action="store_true",
                        help="3x1 CV, 300 bootstraps, 3 seeds (smoke test, not for reporting)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.quick:
        run(args.data, args.output, n_splits=3, n_repeats=1, n_boot=300, seeds=(0, 1, 2))
    else:
        run(args.data, args.output)
    print(f"Reporte completo en {args.output}/")
