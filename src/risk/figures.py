"""Figuras de riesgo, inferencia, estabilidad, equidad y monitoreo.

Reutilizan la paleta y los rcParams de ``src.ml.plots`` para que todas las
figuras del reporte formen un solo sistema: el color sigue al modelo (nunca a
su ranking), marcas finas, grilla discreta y un solo eje Y por gráfico.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.ml.plots import BAD, COLORS, GOOD, GRID, INK, MUTED, NAMES, _save

SCORECARD_COLOR = "#4a3aa7"
COLORS_EXT = {**COLORS, "scorecard": SCORECARD_COLOR}
NAMES_EXT = {**NAMES, "scorecard": "Scorecard WoE"}


def scorecard_vs_ml(y: np.ndarray, proba: dict[str, np.ndarray], path: Path) -> None:
    """Curvas ROC del scorecard tradicional frente a los modelos de ML + barras de KS."""
    from src.ml.metrics import ks_statistic

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    names = list(proba)
    for a in names:
        fpr, tpr, _ = roc_curve(y, proba[a])
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, color=COLORS_EXT[a], linestyle="--" if a == "scorecard" else "-",
                label=f"{NAMES_EXT[a]}  AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Scorecard tradicional vs modelos de ML (holdout)")
    ax.legend(loc="lower right")

    ks = [ks_statistic(y, proba[a]) for a in names]
    positions = np.arange(len(names))
    bx.barh(positions, ks, color=[COLORS_EXT[a] for a in names], height=0.55)
    for pos, value in zip(positions, ks, strict=True):
        bx.annotate(f"{value:.3f}", (value, pos), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    bx.set_yticks(positions, [NAMES_EXT[a] for a in names])
    bx.invert_yaxis()
    bx.set_xlabel("Estadístico KS")
    bx.set_title("Separación (KS) en holdout")
    bx.grid(axis="y", visible=False)
    fig.subplots_adjust(wspace=0.45)
    _save(fig, path)


def information_value(iv_table: list[dict[str, Any]], path: Path) -> None:
    rows = sorted(iv_table, key=lambda r: r["iv"])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.barh([r["feature"] for r in rows], [r["iv"] for r in rows], color=SCORECARD_COLOR,
            height=0.55)
    for i, r in enumerate(rows):
        ax.annotate(f"{r['iv']:.3f} ({r['strength'].split(' (')[0]})", (r["iv"], i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK)
    for x, label in ((0.1, "débil"), (0.3, "medio"), (0.5, "fuerte")):
        ax.axvline(x, color=GRID, linewidth=1, linestyle=":")
        ax.annotate(label, (x, len(rows) - 0.4), fontsize=8, color=MUTED, ha="left")
    ax.set_xlabel("Information Value (datos de desarrollo)")
    ax.set_title("Scorecard: Information Value por variable")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def score_distribution(scores: np.ndarray, y: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bins = np.linspace(np.percentile(scores, 0.5), np.percentile(scores, 99.5), 45)
    ax.hist(scores[y == 0], bins=bins, density=True, alpha=0.55, color=GOOD, label="Buenos")
    ax.hist(scores[y == 1], bins=bins, density=True, alpha=0.55, color=BAD, label="Malos (default)")
    ax.set_xlabel("Puntaje (mayor = menor riesgo; 600 = odds 50:1, +20 pts duplican los odds)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución del puntaje del scorecard (holdout)")
    ax.legend()
    _save(fig, path)


def lift_and_concentration(curves: dict[str, dict[str, np.ndarray]], champion: str,
                           path: Path) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.6))
    for a, c in curves.items():
        keep = c["population_share"] >= 0.01  # el lift del primer 0-1 % es muy ruidoso
        ax.plot(c["population_share"][keep] * 100, c["lift"][keep], color=COLORS_EXT[a],
                linestyle="--" if a in ("logistic_regression", "scorecard") else "-",
                label=NAMES_EXT[a])
    ax.axhline(1.0, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xlim(0, 100)
    ax.set_xlabel("% de solicitantes revisados (de mayor a menor riesgo)")
    ax.set_ylabel("Lift (captura acumulada / % revisado)")
    ax.set_title("Lift acumulado (holdout)")
    ax.legend()
    c = curves[champion]
    bx.plot(c["population_share"] * 100, c["captured_defaults"] * 100, color=COLORS_EXT[champion])
    bx.plot([0, 100], [0, 100], color=MUTED, linewidth=1, linestyle=":")
    bx.set_xlabel("% de solicitantes revisados")
    bx.set_ylabel("% de defaults capturados")
    bx.set_title(f"Defaults acumulados, {NAMES_EXT[champion]}")
    _save(fig, path)


def cost_ratio_thresholds(rows: list[dict[str, float]], path: Path) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.6))
    ratio = [r["fn_fp_ratio"] for r in rows]
    x = np.arange(len(rows))
    ax.plot(x, [r["threshold"] for r in rows], color=COLORS["catboost"], marker="o",
            markersize=6, label="Umbral elegido (OOF)")
    ax.plot(x, [r["bayes_threshold"] for r in rows], color=MUTED, linestyle="--",
            label="Teórico 1/(1+ratio)")
    for xi, r in zip(x, rows, strict=True):
        ax.annotate(f"{r['threshold']:.3f}", (xi, r["threshold"]), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [f"{r:g}:1" for r in ratio])
    ax.set_xlabel("Relación de costos FN : FP")
    ax.set_ylabel("Umbral de probabilidad")
    ax.set_title("Umbral de mínimo costo según la relación FN:FP")
    ax.legend()

    bx.plot(x, [r["recall"] for r in rows], color="#eb6834", marker="o", markersize=6,
            label="Recall (defaults detectados)")
    bx.plot(x, [r["rejection_rate"] for r in rows], color=MUTED, marker="o", markersize=6,
            linestyle="--", label="Tasa de rechazo")
    bx.plot(x, [r["precision"] for r in rows], color="#4a3aa7", marker="o", markersize=6,
            label="Precisión")
    bx.set_xticks(x, [f"{r:g}:1" for r in ratio])
    bx.set_ylim(0, 1)
    bx.set_xlabel("Relación de costos FN : FP")
    bx.set_ylabel("Valor en holdout")
    bx.set_title("Perfil de decisión en holdout")
    bx.legend()
    _save(fig, path)


def nested_cv_figure(nested: dict[str, Any], path: Path) -> None:
    algos = [a for a in nested["nested"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    rng = np.random.default_rng(0)
    for i, a in enumerate(algos):
        folds = nested["nested"][a]["folds"]
        tuned = np.array([f["outer_tuned"]["roc_auc"] for f in folds])
        default = np.array([f["outer_default"]["roc_auc"] for f in folds])
        ax.scatter(i - 0.12 + rng.uniform(-0.04, 0.04, len(default)), default, s=30,
                   color="white", edgecolor=COLORS_EXT[a], linewidth=1.5, zorder=3,
                   label="Parámetros por defecto" if i == 0 else None)
        ax.scatter(i + 0.12 + rng.uniform(-0.04, 0.04, len(tuned)), tuned, s=30,
                   color=COLORS_EXT[a], zorder=3, label="Ajustado con Optuna (anidado)" if i == 0 else None)
        ax.annotate(f"{tuned.mean():.4f}", (i + 0.12, tuned.max()), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(algos)), [NAMES_EXT[a] for a in algos])
    ax.set_ylabel("ROC-AUC en fold externo")
    cfg = nested["config"]
    ax.set_title(f"CV anidada: {cfg['outer_splits']} folds externos x {cfg['inner_splits']} "
                 f"internos, {cfg['n_trials']} trials de Optuna\najustado vs parámetros por defecto "
                 "en los mismos folds externos")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    _save(fig, path)


def model_comparison_forest(rows: list[dict[str, Any]], title: str, path: Path,
                            margin: float = 0.005) -> None:
    rows = sorted(rows, key=lambda r: r["delta"])
    fig, ax = plt.subplots(figsize=(9.5, max(3.2, 0.42 * len(rows) + 1.6)))
    for i, r in enumerate(rows):
        significant = r["p_adjusted_holm"] < 0.05
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=MUTED, linewidth=1.6, zorder=2)
        ax.scatter([r["delta"]], [i], s=46, zorder=3,
                   color=GOOD if significant else "white", edgecolor=GOOD, linewidth=1.6)
        ax.annotate(f"p Holm {r['p_adjusted_holm']:.3g}", (r["ci_high"], i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.axvline(0, color=INK, linewidth=1)
    ax.axvspan(-margin, margin, color=GRID, alpha=0.6, zorder=0)
    ax.set_yticks(range(len(rows)),
                  [f"{NAMES_EXT.get(r['model_a'], r['model_a'])} - "
                   f"{NAMES_EXT.get(r['model_b'], r['model_b'])}" for r in rows])
    ax.set_xlabel(f"Diferencia de AUC con IC 95 % (banda gris = margen práctico ±{margin})")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def stability_figure(seed_result: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    rng = np.random.default_rng(1)
    algos = list(seed_result["per_seed"])
    for i, a in enumerate(algos):
        values = np.array([r["roc_auc"] for r in seed_result["per_seed"][a]])
        ax.boxplot(values, positions=[i], widths=0.45, showfliers=False,
                   medianprops={"color": INK}, boxprops={"color": MUTED},
                   whiskerprops={"color": MUTED}, capprops={"color": MUTED})
        ax.scatter(i + rng.uniform(-0.1, 0.1, len(values)), values, s=24, color=COLORS_EXT[a],
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.annotate(f"sd {values.std(ddof=1):.4f}", (i, values.max()), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(algos)), [NAMES_EXT[a] for a in algos])
    ax.set_ylabel("ROC-AUC en test")
    ax.set_title(f"Estabilidad con {len(seed_result['seeds'])} semillas aleatorias "
                 "(split + semilla del modelo)")
    _save(fig, path)


def stress_figure(rows: list[dict[str, Any]], path: Path) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.4))
    display = {"base": "base", "moderate_stress": "stress moderado", "severe_stress": "stress severo"}
    labels = [display.get(r["scenario"], r["scenario"]) for r in rows]
    colors = [MUTED, "#eb6834", BAD]
    for axis, key, title, fmt in (
        (ax, "mean_pd", "PD media predicha", "{:.2%}"),
        (bx, "rejection_rate", "Tasa de rechazo con el umbral de decisión", "{:.1%}"),
    ):
        values = [r[key] for r in rows]
        axis.bar(labels, values, color=colors[: len(rows)], width=0.55)
        for i, v in enumerate(values):
            axis.annotate(fmt.format(v), (i, v), xytext=(0, 4), textcoords="offset points",
                          ha="center", fontsize=9, color=INK)
        axis.set_title(title)
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        axis.grid(axis="x", visible=False)
    fig.suptitle("Análisis de escenarios (choques ilustrativos, NO es un pronóstico)", fontsize=11,
                 color=MUTED, y=1.02)
    _save(fig, path)


def fairness_figure(report: dict[str, Any], path: Path) -> None:
    groups = report["groups"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    specs = (
        ("rejection_rate", "Tasa de rechazo"),
        ("tpr_defaults_flagged", "TPR (defaults detectados)"),
        ("fpr_goods_rejected", "FPR (buenos rechazados)"),
    )
    for ax, (key, title) in zip(axes, specs, strict=True):
        for i, g in enumerate(groups):
            r = g[key]
            if r is None:
                continue
            color = MUTED if g["small_sample"] else GOOD
            ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=color, linewidth=1.8, zorder=2)
            ax.scatter([r["value"]], [i], s=40, color=color, zorder=3, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("Tasa con IC 95 % de Wilson")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(groups)), [f"{g['group']} (n={g['n']:,})" for g in groups])
    axes[0].invert_yaxis()
    fig.suptitle("Diagnóstico descriptivo por grupo de edad con el umbral elegido "
                 "(sin conclusiones causales; la edad es la única variable sensible disponible)",
                 fontsize=10.5, color=MUTED, y=1.03)
    _save(fig, path)


def psi_monitoring(feature_psi: dict[str, dict[str, float]], score_psi: dict[str, float],
                   path: Path, threshold: float = 0.2) -> None:
    """PSI por variable en una ventana estable (holdout) y en una ventana SIMULADA con stress."""
    features = list(next(iter(feature_psi.values())))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    width = 0.38
    x = np.arange(len(features))
    styles = {"holdout (sin cambio esperado)": GOOD, "cambio SIMULADO (stress moderado)": "#eb6834"}
    for k, (name, psi) in enumerate(feature_psi.items()):
        ax.bar(x + (k - 0.5) * width, [psi[f] for f in features], width * 0.92,
               color=styles.get(name, MUTED), label=f"{name} | PSI del score {score_psi[name]:.3f}")
    ax.axhline(0.1, color=MUTED, linewidth=1, linestyle=":")
    ax.axhline(threshold, color=BAD, linewidth=1.2, linestyle="--")
    ax.annotate("0.1 moderado", (len(features) - 0.5, 0.1), fontsize=8, color=MUTED,
                ha="right", va="bottom")
    ax.annotate(f"{threshold} significativo", (len(features) - 0.5, threshold), fontsize=8,
                color=BAD, ha="right", va="bottom")
    ax.set_xticks(x, features, rotation=28, ha="right")
    ax.set_ylabel("Population Stability Index")
    ax.set_title("Deriva de variables (PSI vs distribución de desarrollo)")
    ax.set_ylim(0, max(0.22, ax.get_ylim()[1]))
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.72))
    _save(fig, path)


def score_drift_figure(reference: list[float], current: dict[str, list[float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    n = len(reference)
    x = np.arange(1, n + 1)
    ax.plot(x, np.array(reference) * 100, color=MUTED, marker="o", markersize=5, linestyle="--",
            label="Referencia (OOF de desarrollo)")
    palette = [GOOD, "#eb6834"]
    for (name, shares), color in zip(current.items(), palette, strict=False):
        ax.plot(x, np.array(shares) * 100, color=color, marker="o", markersize=5, label=name)
    ax.set_xticks(x)
    ax.set_xlabel("Decil de PD de referencia (1 = menor PD predicha)")
    ax.set_ylabel("% de la población de la ventana")
    ax.set_title("Deriva de predicciones: composición por decil de PD de referencia")
    ax.legend()
    _save(fig, path)


def shap_dependence(sample: pd.DataFrame, values: np.ndarray, features: list[str],
                    path: Path) -> None:
    n = len(features)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.2 * rows))
    for ax, feature in zip(np.atleast_1d(axes).ravel(), features, strict=False):
        j = list(sample.columns).index(feature)
        x = sample[feature].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(x, [1, 99])
        ax.scatter(np.clip(x, lo, hi), values[:, j], s=8, alpha=0.35, color=GOOD, linewidth=0)
        ax.axhline(0, color=MUTED, linewidth=1, linestyle=":")
        ax.set_xlabel("valor de la variable (recortado a percentiles 1-99)")
        ax.set_ylabel("SHAP (log-odds)")
        ax.set_title(feature, fontsize=11)
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Dependencia SHAP = asociación aprendida por el modelo, no causalidad",
                 fontsize=11, color=MUTED)
    fig.tight_layout()
    _save(fig, path)


def shap_local(cases: list[dict[str, Any]], path: Path) -> None:
    """Barras tipo cascada con los factores principales de algunos solicitantes."""
    fig, axes = plt.subplots(len(cases), 1, figsize=(9, 3.3 * len(cases)))
    for ax, case in zip(np.atleast_1d(axes), cases, strict=True):
        factors = sorted(case["factors"], key=lambda f: abs(f["impact"]))
        colors = [BAD if f["impact"] > 0 else GOOD for f in factors]
        ax.barh([f"{f['feature']} = {f['value']:.3g}" for f in factors],
                [f["impact"] for f in factors], color=colors, height=0.55)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_title(f"{case['label']}\nPD {case['probability']:.1%}", fontsize=10)
        ax.set_xlabel("SHAP (log-odds; rojo sube el riesgo)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Explicaciones locales de tres solicitantes del holdout", fontsize=11, color=MUTED,
                 y=1.01)
    fig.tight_layout()
    _save(fig, path)


def shap_interaction_heatmap(matrix: pd.DataFrame, path: Path) -> None:
    order = matrix.sum().sort_values(ascending=False).index.tolist()
    m = matrix.loc[order, order].to_numpy().copy()
    off = m.copy()
    np.fill_diagonal(off, np.nan)
    fig, ax = plt.subplots(figsize=(8.6, 7))
    im = ax.imshow(off, cmap="Blues")
    ax.set_xticks(range(len(order)), order, rotation=40, ha="right")
    ax.set_yticks(range(len(order)), order)
    vmax = np.nanmax(off)
    for i in range(len(order)):
        for j in range(len(order)):
            if i != j:
                ax.text(j, i, f"{off[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if off[i, j] > 0.6 * vmax else INK)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="media |valor de interacción SHAP|")
    ax.set_title("Fuerza de interacción SHAP por pares (fuera de la diagonal)")
    _save(fig, path)
