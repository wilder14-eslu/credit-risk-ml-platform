"""Rigorous, reproducible offline evaluation of every candidate model.

What `src.ml.benchmark` does in one train/test split, this module does with
the statistical care a model-validation team expects:

1. **Holdout reservado.** The same stratified 80/20 split used by
   `src.ml.train` (seed 42). The 20 % test set is touched **once**, at the
   end, never for model or threshold selection.
2. **Validación cruzada estratificada repetida** (default 5 folds x 3
   repeats) on the 80 % development set, with median imputation fitted
   *inside* each fold (no leakage). Per-fold ROC-AUC, PR-AUC, Gini, KS,
   Brier, log loss, ECE, train-fold AUC (overfitting gap) and fit time.
3. **Selección del champion** by mean CV ROC-AUC, and Nadeau-Bengio
   corrected t-tests of every candidate against it.
4. **Evaluación en holdout**: every model refit on the whole development
   set; DeLong CI + DeLong tests vs the champion, bootstrap CIs, calibration
   (ECE, slope/intercept), decile/gains table, latency (single-row p50/p95,
   the number that matters for an online API).
5. **Umbral de decisión** chosen on out-of-fold predictions (never on the
   holdout) by F1, Youden's J and expected misclassification cost.
6. **Figuras** (PNG) + `metrics.json` + `evaluation_report.md` under
   ``reports/`` so the README numbers are traceable to a single command.

Run with:
    python -m src.ml.evaluation            # full run (~5-10 min on a laptop)
    python -m src.ml.evaluation --quick    # 3 folds x 1 repeat, 300 bootstraps
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from src.data_pipeline.preprocess import (
    apply_imputation,
    fit_imputation_values,
    split_raw_features,
)
from src.ml import stats as st
from src.ml.metrics import compute_classification_metrics, ks_statistic
from src.ml.train import ALGORITHMS, DEFAULT_PARAMS, _build_model

logger = logging.getLogger(__name__)

DISPLAY_NAMES = {
    "logistic_regression": "Regresión Logística",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}


def _fold_metrics(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y_true, p)
    return {
        "roc_auc": float(auc),
        "pr_auc": float(average_precision_score(y_true, p)),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, p),
        "brier": float(brier_score_loss(y_true, p)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "ece": st.expected_calibration_error(y_true, p),
    }


def _pipeline(algorithm: str) -> Pipeline:
    """Imputation inside the estimator so CV refits it per fold."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", _build_model(algorithm, dict(DEFAULT_PARAMS[algorithm]))),
        ]
    ).set_output(transform="pandas")


def available_algorithms(algorithms: tuple[str, ...] = ALGORITHMS) -> list[str]:
    usable = []
    for algorithm in algorithms:
        try:
            _build_model(algorithm, dict(DEFAULT_PARAMS[algorithm]))
            usable.append(algorithm)
        except ImportError as error:
            logger.warning("Se omite %s: %s", algorithm, error)
    return usable


def cross_validate_models(
    x_dev: pd.DataFrame,
    y_dev: pd.Series,
    algorithms: list[str],
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> tuple[dict[str, list[dict[str, float]]], dict[str, np.ndarray]]:
    """Repeated stratified k-fold. Every model sees exactly the same folds
    (required for the paired/corrected t-test). Out-of-fold probabilities
    are kept from the first repetition (each row predicted exactly once)."""
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    splits = list(cv.split(x_dev, y_dev))
    y_arr = y_dev.to_numpy()

    per_fold: dict[str, list[dict[str, float]]] = {a: [] for a in algorithms}
    oof: dict[str, np.ndarray] = {a: np.full(len(y_dev), np.nan) for a in algorithms}

    for algorithm in algorithms:
        for fold_id, (train_idx, val_idx) in enumerate(splits):
            estimator = clone(_pipeline(algorithm))
            start = time.perf_counter()
            estimator.fit(x_dev.iloc[train_idx], y_arr[train_idx])
            fit_seconds = time.perf_counter() - start

            p_val = estimator.predict_proba(x_dev.iloc[val_idx])[:, 1]
            p_train = estimator.predict_proba(x_dev.iloc[train_idx])[:, 1]
            metrics = _fold_metrics(y_arr[val_idx], p_val)
            metrics["train_roc_auc"] = float(roc_auc_score(y_arr[train_idx], p_train))
            metrics["overfit_gap"] = metrics["train_roc_auc"] - metrics["roc_auc"]
            metrics["fit_seconds"] = fit_seconds
            metrics["repeat"] = fold_id // n_splits
            metrics["fold"] = fold_id % n_splits
            per_fold[algorithm].append(metrics)

            if fold_id < n_splits:
                oof[algorithm][val_idx] = p_val
            logger.info(
                "%-20s fold %2d  AUC=%.4f  gap=%.4f",
                algorithm, fold_id, metrics["roc_auc"], metrics["overfit_gap"],
            )
    return per_fold, oof


def summarize_cv(
    per_fold: dict[str, list[dict[str, float]]], test_train_ratio: float | None = None
) -> dict[str, dict[str, Any]]:
    metric_names = (
        "roc_auc", "pr_auc", "gini", "ks", "brier", "log_loss", "ece",
        "train_roc_auc", "overfit_gap", "fit_seconds",
    )
    summary: dict[str, dict[str, Any]] = {}
    for algorithm, folds in per_fold.items():
        summary[algorithm] = {}
        for name in metric_names:
            values = [fold[name] for fold in folds]
            summary[algorithm][name] = {
                **st.t_confidence_interval(values, test_train_ratio=test_train_ratio),
                "values": values,
            }
    return summary


def measure_latency(model: Any, x_sample: pd.DataFrame, n_calls: int = 300) -> dict[str, float]:
    """Single-applicant latency (what an online API pays per request) and
    batch throughput."""
    rows = [x_sample.iloc[[i % len(x_sample)]] for i in range(n_calls)]
    for row in rows[:20]:  # warm-up
        model.predict_proba(row)
    timings = []
    for row in rows:
        start = time.perf_counter()
        model.predict_proba(row)
        timings.append((time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    model.predict_proba(x_sample)
    batch_seconds = time.perf_counter() - start
    return {
        "single_row_p50_ms": float(np.percentile(timings, 50)),
        "single_row_p95_ms": float(np.percentile(timings, 95)),
        "batch_rows_per_second": float(len(x_sample) / batch_seconds),
    }


def _bootstrap_panel(y: np.ndarray, p: np.ndarray, n_boot: int) -> dict[str, dict[str, float]]:
    return {
        "pr_auc": st.bootstrap_ci(y, p, average_precision_score, n_boot=n_boot),
        "ks": st.bootstrap_ci(y, p, ks_statistic, n_boot=n_boot),
        "brier": st.bootstrap_ci(y, p, brier_score_loss, n_boot=n_boot),
    }


def run_evaluation(
    raw_data: pd.DataFrame,
    output_dir: Path | str = "reports",
    figures_dir: Path | str = "reports/figures",
    n_splits: int = 5,
    n_repeats: int = 3,
    n_boot: int = 1000,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
    algorithms: tuple[str, ...] = ALGORITHMS,
    make_figures: bool = True,
    return_artifacts: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    """Ejecuta el protocolo completo. Con ``return_artifacts=True`` devuelve además
    los modelos ajustados, los splits y las predicciones, para que los análisis
    posteriores (scorecard, stress, equidad, SHAP) usen exactamente los mismos
    objetos."""
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_dev, x_test, y_dev, y_test = split_raw_features(raw_data)
    usable = available_algorithms(algorithms)

    # ---- 1. Cross-validation on the development set -----------------------
    per_fold, oof = cross_validate_models(x_dev, y_dev, usable, n_splits, n_repeats)
    n_val = len(x_dev) // n_splits
    n_fit = len(x_dev) - n_val
    cv_summary = summarize_cv(per_fold, test_train_ratio=n_val / n_fit)
    champion = max(usable, key=lambda a: cv_summary[a]["roc_auc"]["mean"])
    cv_tests = {}
    for algorithm in usable:
        if algorithm == champion:
            continue
        cv_tests[algorithm] = st.corrected_resampled_ttest(
            cv_summary[champion]["roc_auc"]["values"],
            cv_summary[algorithm]["roc_auc"]["values"],
            n_train=n_fit,
            n_test=n_val,
        )

    # ---- 2. Refit on the full development set, score the holdout ----------
    imputation = fit_imputation_values(x_dev)
    x_dev_imp = apply_imputation(x_dev, imputation)
    x_test_imp = apply_imputation(x_test, imputation)
    y_test_arr = y_test.to_numpy()

    fitted: dict[str, Any] = {}
    test_proba: dict[str, np.ndarray] = {}
    holdout: dict[str, dict[str, Any]] = {}
    calibration: dict[str, dict[str, float]] = {}
    latency: dict[str, dict[str, float]] = {}
    for algorithm in usable:
        model = _build_model(algorithm, dict(DEFAULT_PARAMS[algorithm]))
        model.fit(x_dev_imp, y_dev)
        fitted[algorithm] = model
        p_test = model.predict_proba(x_test_imp)[:, 1]
        p_train = model.predict_proba(x_dev_imp)[:, 1]
        test_proba[algorithm] = p_test

        metrics_05 = compute_classification_metrics(y_test_arr, p_test, 0.5)
        holdout[algorithm] = {
            "metrics_at_0_5": metrics_05,
            "train_roc_auc": float(roc_auc_score(y_dev, p_train)),
            "auc_delong": st.delong_auc_ci(y_test_arr, p_test),
            "bootstrap": _bootstrap_panel(y_test_arr, p_test, n_boot),
            "ece": st.expected_calibration_error(y_test_arr, p_test),
        }
        calibration[algorithm] = st.calibration_slope_intercept(y_test_arr, p_test)
        latency[algorithm] = measure_latency(model, x_test_imp.iloc[:2000])

    for algorithm in usable:
        if algorithm != champion:
            holdout[algorithm]["delong_vs_champion"] = st.delong_roc_test(
                y_test_arr, test_proba[champion], test_proba[algorithm]
            )

    # ---- 3. Threshold chosen on OOF predictions (development set) ---------
    oof_champion = oof[champion]
    thresholds = st.threshold_analysis(y_dev.to_numpy(), oof_champion, cost_fn, cost_fp)
    selected_threshold = thresholds["min_cost"]["threshold"]
    holdout_at_selected = compute_classification_metrics(
        y_test_arr, test_proba[champion], selected_threshold
    )
    holdout_rejection = float((test_proba[champion] >= selected_threshold).mean())

    deciles = st.decile_table(y_test_arr, test_proba[champion])
    strategy = {
        algorithm: st.approval_strategy_curve(y_test_arr, test_proba[algorithm])
        for algorithm in usable
    }

    result = {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "rows_after_cleaning": int(len(x_dev) + len(x_test)),
            "n_features": int(x_dev.shape[1]),
            "features": list(x_dev.columns),
            "prevalence": float(pd.concat([y_dev, y_test]).mean()),
            "n_dev": len(x_dev),
            "n_test": len(x_test),
            "test_positives": int(y_test.sum()),
            "cv": {"n_splits": n_splits, "n_repeats": n_repeats, "random_state": 42},
            "n_bootstrap": n_boot,
            "missing_rate": {c: float(v) for c, v in x_dev.isna().mean().items() if v > 0},
            "python": platform.python_version(),
            "hyperparameters": {a: DEFAULT_PARAMS[a] for a in usable},
        },
        "champion": champion,
        "cv": {
            a: {k: {kk: vv for kk, vv in v.items() if kk != "values"} for k, v in s.items()}
            for a, s in cv_summary.items()
        },
        "cv_folds": per_fold,
        "cv_tests_vs_champion": cv_tests,
        "holdout": holdout,
        "calibration": calibration,
        "latency": latency,
        "threshold": {
            "selected_on": "out-of-fold predictions (development set)",
            "cost_fn": cost_fn,
            "cost_fp": cost_fp,
            "bayes_threshold": thresholds["bayes_threshold"],
            "best_f1": thresholds["best_f1"],
            "best_youden": thresholds["best_youden"],
            "min_cost": thresholds["min_cost"],
            "selected_threshold": selected_threshold,
            "holdout_at_selected": holdout_at_selected,
            "holdout_rejection_rate_at_selected": holdout_rejection,
            "holdout_rejection_rate_at_0_5": float((test_proba[champion] >= 0.5).mean()),
        },
        "deciles": deciles,
        "strategy": {a: [r for r in rows if round(r["approval_rate"] * 100) % 5 == 0]
                     for a, rows in strategy.items()},
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "evaluation_report.md").write_text(render_markdown(result), encoding="utf-8")

    if make_figures:
        from src.ml import plots

        plots.render_all(
            figures_dir=figures_dir,
            result=result,
            x_dev=x_dev,
            y_dev=y_dev,
            x_test_imp=x_test_imp,
            y_test=y_test_arr,
            test_proba=test_proba,
            oof_champion=oof_champion,
            threshold_curve=thresholds["curve"],
            strategy_full=strategy,
            fitted=fitted,
            x_dev_imp=x_dev_imp,
        )
    if return_artifacts:
        artifacts = {
            "x_dev": x_dev, "x_test": x_test, "y_dev": y_dev, "y_test": y_test,
            "x_dev_imp": x_dev_imp, "x_test_imp": x_test_imp, "imputation": imputation,
            "fitted": fitted, "test_proba": test_proba, "oof": oof,
            "per_fold": per_fold, "usable": usable, "n_fit": n_fit, "n_val": n_val,
        }
        return result, artifacts
    return result


# --------------------------------------------------------------------------
# Markdown report (auto-generated, traceable)
# --------------------------------------------------------------------------


def _fmt_ci(summary: dict[str, float], digits: int = 4) -> str:
    return (
        f"{summary['mean']:.{digits}f} ± {summary['std']:.{digits}f} "
        f"[{summary['ci_low']:.{digits}f}, {summary['ci_high']:.{digits}f}]"
    )


def render_markdown(result: dict[str, Any]) -> str:
    meta = result["meta"]
    champion = result["champion"]
    order = sorted(result["cv"], key=lambda a: -result["cv"][a]["roc_auc"]["mean"])
    lines = [
        "# Informe de evaluación de modelos (auto-generado)",
        "",
        f"_Generado: {meta['generated_at']} con `python -m src.ml.evaluation`_",
        "",
        (
            f"- Filas: {meta['rows_after_cleaning']:,} | Prevalencia de default: "
            f"{meta['prevalence']:.2%} | Desarrollo: {meta['n_dev']:,} | "
            f"Holdout: {meta['n_test']:,}"
        ),
        (
            f"- CV: {meta['cv']['n_splits']} folds x {meta['cv']['n_repeats']} repeticiones "
            f"(estratificada) | Bootstrap: {meta['n_bootstrap']} remuestras"
        ),
        f"- Champion (por ROC-AUC medio en CV): **{DISPLAY_NAMES[champion]}**",
        "",
        "## Validación cruzada (media ± desv. estándar [IC 95 % t corregido Nadeau-Bengio])",
        "",
        "| Modelo | ROC-AUC | PR-AUC | KS | Brier | ECE | Brecha train-val |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in order:
        cv = result["cv"][a]
        lines.append(
            f"| {DISPLAY_NAMES[a]} | {_fmt_ci(cv['roc_auc'])} | {_fmt_ci(cv['pr_auc'])} | "
            f"{_fmt_ci(cv['ks'])} | {cv['brier']['mean']:.4f} | {cv['ece']['mean']:.4f} | "
            f"{cv['overfit_gap']['mean']:+.4f} |"
        )
    lines += ["", "## Tests vs champion", "",
              "| Modelo | Δ AUC CV | p (Nadeau-Bengio) | Δ AUC holdout | p (DeLong) |",
              "|---|---|---|---|---|"]
    for a in order:
        if a == champion:
            continue
        t = result["cv_tests_vs_champion"][a]
        d = result["holdout"][a]["delong_vs_champion"]
        lines.append(
            f"| {DISPLAY_NAMES[a]} | {t['diff_mean']:+.4f} | {t['p_value']:.3g} | "
            f"{d['diff']:+.4f} | {d['p_value']:.3g} |"
        )
    lines += ["", "## Holdout (20 %, evaluado una sola vez)", "",
              (
                  "| Modelo | ROC-AUC [IC 95 % DeLong] | PR-AUC [IC boot] | KS [IC boot] | "
                  "Brier | ECE | Pendiente calib. | Latencia p50 (ms) |"
              ),
              "|---|---|---|---|---|---|---|---|"]
    for a in order:
        h = result["holdout"][a]
        auc = h["auc_delong"]
        b = h["bootstrap"]
        lines.append(
            f"| {DISPLAY_NAMES[a]} | {auc['auc']:.4f} [{auc['ci_low']:.4f}, {auc['ci_high']:.4f}] | "
            f"{b['pr_auc']['estimate']:.4f} [{b['pr_auc']['ci_low']:.4f}, {b['pr_auc']['ci_high']:.4f}] | "
            f"{b['ks']['estimate']:.4f} [{b['ks']['ci_low']:.4f}, {b['ks']['ci_high']:.4f}] | "
            f"{b['brier']['estimate']:.4f} | {h['ece']:.4f} | "
            f"{result['calibration'][a]['slope']:.3f} | "
            f"{result['latency'][a]['single_row_p50_ms']:.2f} |"
        )
    th = result["threshold"]
    sel = th["holdout_at_selected"]
    lines += [
        "",
        "## Umbral de decisión",
        "",
        (
            "- Elegido en predicciones out-of-fold "
            f"(costo FN:FP = {th['cost_fn']:g}:{th['cost_fp']:g}): "
            f"**{th['selected_threshold']:.3f}** "
            f"(umbral bayesiano teórico {th['bayes_threshold']:.3f})"
        ),
        (
            f"- Holdout con ese umbral: precisión {sel['precision']:.3f}, "
            f"recall {sel['recall']:.3f}, F1 {sel['f1']:.3f}, "
            f"tasa de rechazo {th['holdout_rejection_rate_at_selected']:.1%}"
        ),
        "",
        "## Tabla de deciles (holdout, champion)",
        "",
        "| Decil | Score mín-máx | Tasa default | Lift | % defaults capturados (acum.) | KS |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["deciles"]:
        lines.append(
            f"| {row['decile']} | {row['min_score']:.3f}-{row['max_score']:.3f} | "
            f"{row['bad_rate']:.2%} | {row['lift']:.2f} | {row['cum_bad_capture']:.1%} | "
            f"{row['ks']:.3f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import sys

    from src.data_pipeline.ingest import download_give_me_some_credit
    from src.data_pipeline.validate import clean_out_of_range_rows, validate_input_data

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--cost-fn", type=float, default=5.0)
    parser.add_argument("--cost-fp", type=float, default=1.0)
    parser.add_argument("--quick", action="store_true", help="3x1 CV, 300 bootstraps")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.folds, args.repeats, args.bootstrap = 3, 1, 300

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    data_dir = download_give_me_some_credit()
    csv_files = sorted(Path(data_dir).glob("cs-training.csv"))
    if not csv_files:
        sys.exit(f"No se encontró cs-training.csv en {data_dir}")
    data = clean_out_of_range_rows(pd.read_csv(csv_files[0], index_col=0))
    validate_input_data(data, require_target=True)

    output = run_evaluation(
        data,
        n_splits=args.folds,
        n_repeats=args.repeats,
        n_boot=args.bootstrap,
        cost_fn=args.cost_fn,
        cost_fp=args.cost_fp,
        make_figures=not args.no_figures,
    )
    print(render_markdown(output))
