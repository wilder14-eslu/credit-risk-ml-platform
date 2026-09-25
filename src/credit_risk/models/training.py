"""Split, benchmark de candidatos, tuning con Optuna, quality gates e informe."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.config import base_feature_names, platform_config, target_name
from credit_risk.features.engineering import MedianImputer, build_features
from credit_risk.models import metrics as M
from credit_risk.models.candidates import available_candidates, build_estimator, suggest_params
from credit_risk.models.credit_model import CreditRiskModel

logger = logging.getLogger(__name__)


@dataclass
class Splits:
    x_train: pd.DataFrame  # columnas base crudas (sin features derivadas)
    y_train: pd.Series
    x_val: pd.DataFrame
    y_val: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series


def make_splits(data: pd.DataFrame, cfg: dict | None = None) -> Splits:
    cfg = cfg or platform_config()["data"]
    x = data[list(base_feature_names())]
    y = data[target_name()].astype(int)
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        x, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["random_state"]
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp,
        y_tmp,
        test_size=cfg["validation_size"],
        stratify=y_tmp,
        random_state=cfg["random_state"],
    )
    return Splits(x_train, y_train, x_val, y_val, x_test, y_test)


def fit_model(name: str, params: dict[str, Any], splits: Splits, seed: int = 42) -> CreditRiskModel:
    imputer = MedianImputer()
    x_train = imputer.fit_transform(build_features(splits.x_train))
    estimator = build_estimator(name, params, seed=seed)
    estimator.fit(x_train, splits.y_train)
    model = CreditRiskModel(name, estimator, imputer, metadata={"params": params})
    tcfg = platform_config()["training"]
    val_proba = model.predict_proba(splits.x_val)
    model.threshold = M.optimal_threshold(
        splits.y_val, val_proba, tcfg["cost_false_negative"], tcfg["cost_false_positive"]
    )
    return model


def evaluate(model: CreditRiskModel, splits: Splits) -> dict[str, float]:
    out: dict[str, Any] = {"threshold": model.threshold}
    for prefix, x, y in (
        ("train", splits.x_train, splits.y_train),
        ("test", splits.x_test, splits.y_test),
    ):
        for k, v in M.classification_metrics(y, model.predict_proba(x), model.threshold).items():
            out[f"{prefix}_{k}"] = v
    out["latency_ms"] = M.inference_latency_ms(model.predict_proba, splits.x_test)
    out.update(M.diagnose_fit(out["train_roc_auc"], out["test_roc_auc"]))
    return out


def tune(name: str, splits: Splits, n_trials: int, timeout: int, seed: int = 42) -> dict[str, Any]:
    """Optuna TPE maximizando AUC en validación (nunca toca el test)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    imputer = MedianImputer()
    x_train = imputer.fit_transform(build_features(splits.x_train))
    x_val = imputer.transform(build_features(splits.x_val))

    def objective(trial) -> float:
        est = build_estimator(name, suggest_params(trial, name), seed=seed)
        est.fit(x_train, splits.y_train)
        return M.classification_metrics(splits.y_val, est.predict_proba(x_val)[:, 1])["roc_auc"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    logger.info("Optuna %s: mejor AUC val=%.4f", name, study.best_value)
    return dict(study.best_params)


def run_benchmark(
    splits: Splits,
    candidates: list[str] | None = None,
    n_trials: int = 0,
    timeout: int = 600,
) -> tuple[pd.DataFrame, dict[str, CreditRiskModel]]:
    """Entrena todos los candidatos sobre el MISMO split y devuelve la tabla comparativa."""
    tcfg = platform_config()["training"]
    names = available_candidates(candidates or tcfg["candidates"])
    rows, models = [], {}
    for name in names:
        params = tune(name, splits, n_trials, timeout) if n_trials > 0 else {}
        model = fit_model(name, params, splits)
        result = evaluate(model, splits)
        result["algorithm"] = name
        result["params"] = params
        rows.append(result)
        models[name] = model
        logger.info("%s -> test AUC %.4f", name, result["test_roc_auc"])
    table = pd.DataFrame(rows).sort_values(tcfg["selection_metric"], ascending=False)
    return table.reset_index(drop=True), models


def select_best(table: pd.DataFrame, metric: str | None = None) -> str:
    metric = metric or platform_config()["training"]["selection_metric"]
    gates = platform_config()["quality_gates"]
    eligible = table[table["latency_ms"] <= gates["max_latency_ms"]]
    source = eligible if not eligible.empty else table
    return str(source.sort_values(metric, ascending=False).iloc[0]["algorithm"])


def check_quality_gates(result: dict[str, Any], gates: dict | None = None) -> tuple[bool, list[str]]:
    """Gate absoluto: mínimos que cualquier modelo debe cumplir para registrarse."""
    gates = gates or platform_config()["quality_gates"]
    reasons = []
    if result["test_roc_auc"] < gates["min_test_roc_auc"]:
        reasons.append(f"AUC test {result['test_roc_auc']:.4f} < {gates['min_test_roc_auc']}")
    if result["auc_gap"] > gates["max_overfit_gap"]:
        reasons.append(f"brecha train-test {result['auc_gap']:.4f} > {gates['max_overfit_gap']}")
    if result["test_brier"] > gates["max_brier"]:
        reasons.append(f"Brier {result['test_brier']:.4f} > {gates['max_brier']}")
    if result["latency_ms"] > gates["max_latency_ms"]:
        reasons.append(f"latencia {result['latency_ms']:.2f}ms > {gates['max_latency_ms']}")
    return (not reasons, reasons)


def champion_vs_challenger(
    challenger_auc: float, champion_auc: float | None, min_improvement: float | None = None
) -> tuple[bool, str]:
    """Gate relativo: el challenger debe superar al champion en el mismo test."""
    min_improvement = (
        platform_config()["quality_gates"]["min_auc_improvement"]
        if min_improvement is None
        else min_improvement
    )
    if champion_auc is None or np.isnan(champion_auc):
        return True, "no existe champion: el primer modelo válido se promueve"
    delta = challenger_auc - champion_auc
    if delta >= min_improvement:
        return True, f"challenger mejora AUC en {delta:+.4f}"
    return False, f"mejora insuficiente ({delta:+.4f} < {min_improvement})"


def training_report(table: pd.DataFrame, best: str, result: dict[str, Any], importance: pd.Series) -> str:
    """Informe Markdown que se guarda como artefacto de MLflow y en el volumen UC."""
    cols = [
        "algorithm",
        "test_roc_auc",
        "test_pr_auc",
        "test_ks",
        "test_brier",
        "test_f1",
        "latency_ms",
        "fit_diagnosis",
    ]
    lines = [
        "# Informe de entrenamiento - Credit Risk ML Platform",
        "",
        f"**Modelo ganador:** `{best}`  ",
        f"**Umbral de decisión (costo FN/FP):** {result['threshold']:.2f}  ",
        f"**Diagnóstico de ajuste:** {result['fit_diagnosis']} "
        f"(brecha AUC train-test {result['auc_gap']:+.4f})",
        "",
        "## Benchmark de candidatos (mismo split)",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for _, row in table.iterrows():
        cells = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Train vs test", "", "| métrica | train | test |", "|---|---|---|"]
    for m in ("roc_auc", "gini", "pr_auc", "ks", "brier", "log_loss", "ece", "recall", "precision"):
        lines.append(f"| {m} | {result[f'train_{m}']:.4f} | {result[f'test_{m}']:.4f} |")
    lines += ["", "## Factores más influyentes (|SHAP| medio)", ""]
    for name, value in importance.head(10).items():
        lines.append(f"- `{name}`: {value:.4f}")
    return "\n".join(lines) + "\n"
