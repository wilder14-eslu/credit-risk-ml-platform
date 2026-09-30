"""Model Card y Data Card generadas a partir de resultados ejecutados.

Todo número de las cards proviene de ``reports/metrics.json`` y
``reports/results.json``; nada se escribe a mano. Las secciones de uso
previsto, fuera de alcance y política de reentrenamiento son declaraciones de
gobierno (texto fijo), no resultados.
"""

from __future__ import annotations

from typing import Any

NAMES = {
    "logistic_regression": "Regresión Logística",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "scorecard": "Scorecard WoE",
}


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/d" if value is None else f"{value * 100:.{digits}f} %"


def _commit(prov: dict[str, Any]) -> str:
    git = prov["git"]
    dirty = " (con cambios sin commit al generar)" if git.get("dirty") else ""
    return f"`{git['commit']}`{dirty}"


def render_model_card(metrics: dict[str, Any], results: dict[str, Any]) -> str:
    prov = results["provenance"]
    champion = metrics["champion"]
    order = sorted(metrics["cv"], key=lambda a: -metrics["cv"][a]["roc_auc"]["mean"])
    th = metrics["threshold"]
    sel = th["holdout_at_selected"]
    card = results["scorecard"]["holdout"]

    lines = [
        "# Model Card: modelo de probabilidad de default (PD)",
        "",
        "_Generada automáticamente por `python -m src.pipelines.full_report`._",
        "",
        "## 1. Detalles del modelo",
        "",
        "| Campo | Valor |",
        "|---|---|",
        f"| Champion estadístico | {NAMES[champion]} (mayor ROC-AUC medio en CV) |",
        "| Modelo servido por la demo/API | XGBoost (valor por defecto de `src/ml/train.py`) |",
        "| Benchmark interpretable | Scorecard WoE + regresión logística (`src/risk/scorecard.py`) |",
        "| Salida | Probabilidad de default a 2 años (`SeriousDlqin2yrs`) |",
        (f"| Umbral de decisión | {th['selected_threshold']:.3f} (mínimo costo, FN:FP = "
        f"{th['cost_fn']:g}:{th['cost_fp']:g}, elegido con predicciones out-of-fold) |"),
        f"| Versión de código | {_commit(prov)} |",
        f"| Dataset | `{prov['dataset']['path']}` sha256 `{prov['dataset']['sha256'][:16]}...` |",
        f"| Generada | {prov['generated_at']} · Python {prov['python']} |",
        "| Semilla | 42 (split, folds, bootstrap y modelos) |",
        "",
        "## 2. Uso previsto",
        "",
        ("- Proyecto de portafolio que demuestra metodología de riesgo de crédito, validación "
        "estadística y MLOps sobre un dataset público."),
        "- Ordenar solicitantes por riesgo y estimar una PD calibrada para análisis de escenarios.",
        "",
        "## 3. Fuera de alcance",
        "",
        ("- Decisiones de crédito reales sobre personas: el modelo no fue validado por un área de "
        "riesgo de modelos ni aprobado por un regulador."),
        ("- Cálculo de provisiones o capital regulatorio (no hay LGD, EAD ni validación fuera de "
        "tiempo)."),
        "- Poblaciones distintas a la del dataset (otro país, producto o período).",
        "",
        "## 4. Desempeño (holdout 20 %, evaluado una sola vez)",
        "",
        "| Modelo | ROC-AUC [IC 95 % DeLong] | PR-AUC | KS | Brier | ECE | Pendiente calib. |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in order:
        h = metrics["holdout"][a]
        auc = h["auc_delong"]
        b = h["bootstrap"]
        lines.append(
            f"| {NAMES[a]} | {auc['auc']:.4f} [{auc['ci_low']:.4f}, {auc['ci_high']:.4f}] | "
            f"{b['pr_auc']['estimate']:.4f} | {b['ks']['estimate']:.4f} | "
            f"{b['brier']['estimate']:.4f} | {h['ece']:.4f} | "
            f"{metrics['calibration'][a]['slope']:.3f} |"
        )
    sauc = card["auc_delong"]
    lines += [
        (f"| Scorecard WoE | {sauc['auc']:.4f} [{sauc['ci_low']:.4f}, {sauc['ci_high']:.4f}] | "
        f"{card['pr_auc']:.4f} | {card['ks']:.4f} | {card['brier']:.4f} | {card['ece']:.4f} | "
        f"{card['calibration']['slope']:.3f} |"),
        "",
        (f"Con el umbral elegido, el champion detecta {_pct(sel['recall'])} de los defaults con "
        f"precisión {_pct(sel['precision'])} y rechaza {_pct(th['holdout_rejection_rate_at_selected'])} "
        "de las solicitudes del holdout."),
        "",
        "## 5. Equidad (solo edad disponible)",
        "",
        "| Grupo | n | Tasa de rechazo [IC 95 %] | TPR | FPR | Default observado | PD media |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in results["fairness"]["groups"]:
        rr = g["rejection_rate"]
        tpr = g["tpr_defaults_flagged"]
        fpr = g["fpr_goods_rejected"]
        flag = " (muestra pequeña)" if g["small_sample"] else ""
        lines.append(
            f"| {g['group']}{flag} | {g['n']:,} | {_pct(rr['value'])} "
            f"[{_pct(rr['ci_low'])}, {_pct(rr['ci_high'])}] | "
            f"{_pct(tpr['value'] if tpr else None)} | {_pct(fpr['value'] if fpr else None)} | "
            f"{_pct(g['observed_default_rate']['value'])} | {_pct(g['mean_predicted_pd'])} |"
        )
    lines += [
        "",
        f"> {results['fairness']['caveat']}",
        "",
        "## 6. Explicabilidad",
        "",
        "Importancia global (media de |SHAP| en log-odds) del champion:",
        "",
    ]
    for row in results["explainability"]["global_importance"][:5]:
        lines.append(f"- `{row['feature']}`: {row['mean_abs_shap']:.3f} "
                     f"(Spearman valor-SHAP {row['spearman_value_vs_shap']:+.2f})")
    lines += [
        "",
        f"> {results['explainability']['interpretation_note']}",
        "",
        "## 7. Stress testing (análisis de escenarios, no pronóstico)",
        "",
        "| Escenario | PD media | Cambio vs base | Tasa de rechazo |",
        "|---|---|---|---|",
    ]
    for r in results["stress"]:
        lines.append(
            f"| {r['scenario']} | {_pct(r['mean_pd'], 2)} | {r['mean_pd_change_vs_base']:+.1%} | "
            f"{_pct(r['rejection_rate'])} |"
        )
    cc = results.get("champion_challenger")
    lines += ["", "## 8. Champion vs challenger", ""]
    if cc:
        lines += [
            f"- Champion vigente: {cc['champion']} · Challenger: {cc['challenger']}",
            (f"- Δ AUC (challenger - champion) = {cc['auc_difference']:+.4f}, "
            f"p DeLong = {cc['p_value_delong']:.3g}, margen práctico = {cc['practical_margin']}"),
            f"- Recomendación: **{cc['recommendation']}**",
            "- `auto_promote = False`: ninguna promoción ocurre sin aprobación humana.",
        ]
    lines += [
        "",
        "## 9. Monitoreo y política de reentrenamiento",
        "",
        ("- Calidad de datos (faltantes, esquema, valores inválidos), deriva de variables (PSI), "
        "deriva de predicciones (PSI del score), deriva de desempeño (AUC, PR-AUC, KS, Brier, "
        "calibración) y deriva de negocio (aprobación, composición de cartera, default observado)."),
        ("- Las señales solo abren un **candidato de reentrenamiento**. La deriva nunca despliega "
        "un modelo: la promoción requiere `approved_by` (ver "
        "`docs/governance/RETRAINING_POLICY.md`)."),
        "",
        "## 10. Limitaciones",
        "",
    ]
    for key, item in results["limitations"].items():
        text = item["reason"] if isinstance(item, dict) else item
        lines.append(f"- **{key}**: {text}")
    lines += [
        "- Split aleatorio (no temporal): el desempeño en producción suele ser menor.",
        "- Costos FN:FP ilustrativos; deben reemplazarse por pérdidas reales.",
        "",
        "## 11. Versiones de librerías",
        "",
        ", ".join(f"{k} {v}" for k, v in prov["libraries"].items()),
        "",
    ]
    return "\n".join(lines)


def render_data_card(results: dict[str, Any]) -> str:
    data = results["data"]
    prov = results["provenance"]
    lines = [
        "# Data Card: Give Me Some Credit",
        "",
        "_Generada automáticamente por `python -m src.pipelines.full_report`._",
        "",
        "| Campo | Valor |",
        "|---|---|",
        "| Fuente | Kaggle, competencia \"Give Me Some Credit\" (`cs-training.csv`) |",
        f"| Archivo / sha256 | `{prov['dataset']['path']}` / `{prov['dataset']['sha256']}` |",
        f"| Filas tras limpieza | {data['rows']:,} (se descarta 1 fila con edad fuera de rango) |",
        f"| Variables | {data['n_features']} numéricas |",
        f"| Objetivo | `{data['target_source']}`: atraso de 90+ días en los 2 años siguientes |",
        (f"| Prevalencia de default | {_pct(data['default_prevalence'], 2)} ({data['defaults']:,} "
        "defaults) |"),
        f"| Filas duplicadas | {data['duplicate_rows']:,} (se conservan) |",
        f"| Variable temporal | {data['temporal_variable'] or 'ninguna (no es posible validar fuera de tiempo)'} |",
        "| Población rechazada | no incluida (no es posible inferencia de rechazados) |",
        "| LGD / EAD | no incluidos |",
        "",
        "## Variables",
        "",
        "| Variable | Columna original | % faltante | Mín | Mediana | P99 | Máx |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, f in data["features"].items():
        lines.append(
            f"| `{name}` | `{f['source_column']}` | {_pct(f['missing_rate'])} | {f['min']:.4g} | "
            f"{f['median']:.4g} | {f['p99']:.4g} | {f['max']:.4g} |"
        )
    lines += [
        "",
        "## Códigos centinela 96/98 en conteos de atraso",
        "",
        "| Variable | Filas | Tasa de default en esas filas |",
        "|---|---|---|",
    ]
    for name, s in data["sentinel_codes_96_98"].items():
        lines.append(f"| `{name}` | {s['rows']:,} | {_pct(s['default_rate_in_rows'])} |")
    lines += [
        "",
        "## Consideraciones",
        "",
        ("- `age` es un atributo potencialmente protegido en regulación de crédito; se usa como "
        "variable del modelo y como único eje de diagnóstico de equidad."),
        ("- `MonthlyIncome` y `NumberOfDependents` tienen faltantes: se imputan con medianas "
        "ajustadas solo en entrenamiento (modelos ML) o forman un bin propio (scorecard)."),
        ("- Colas extremas en `DebtRatio` y `RevolvingUtilizationOfUnsecuredLines`: robustas en "
        "árboles; la regresión logística usa transformación por cuantiles y el scorecard bins."),
        "- Licencia: términos de la competencia de Kaggle; revisar antes de redistribuir.",
        "",
    ]
    return "\n".join(lines)
