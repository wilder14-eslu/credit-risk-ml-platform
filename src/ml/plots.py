"""Static figures (PNG) for the evaluation report and the README.

Colour follows the *model*, never its rank, and is identical in every
figure: the Logistic Regression baseline is a neutral dashed grey, the
three boosting libraries take the first three slots of a CVD-validated
categorical palette. Called by `src.ml.evaluation.run_evaluation`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

NAMES = {
    "logistic_regression": "Regresión Logística",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}
COLORS = {
    "xgboost": "#2a78d6",
    "lightgbm": "#eb6834",
    "catboost": "#1baf7a",
    "logistic_regression": "#6b6a66",
}
STYLES = {"logistic_regression": "--"}
INK = "#1f1f1e"
MUTED = "#6b6a66"
GRID = "#e6e5e1"
GOOD = "#2a78d6"
BAD = "#e34948"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": "DejaVu Sans",
        "lines.linewidth": 2,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _order(result: dict[str, Any]) -> list[str]:
    return sorted(result["cv"], key=lambda a: -result["cv"][a]["roc_auc"]["mean"])


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def cv_distribution(result: dict[str, Any], path: Path) -> None:
    order = _order(result)
    folds = result["cv_folds"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    rng = np.random.default_rng(0)
    for ax, metric, title in (
        (axes[0], "roc_auc", "ROC-AUC por fold (validación)"),
        (axes[1], "pr_auc", "PR-AUC por fold (validación)"),
    ):
        for i, a in enumerate(order):
            values = np.array([f[metric] for f in folds[a]])
            ax.boxplot(values, positions=[i], widths=0.5, showfliers=False,
                       medianprops={"color": INK}, boxprops={"color": MUTED},
                       whiskerprops={"color": MUTED}, capprops={"color": MUTED})
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(values)), values, s=22,
                       color=COLORS[a], alpha=0.85, zorder=3, edgecolor="white", linewidth=0.6)
            ax.annotate(f"{values.mean():.4f}", (i, values.max()), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(order)), [NAMES[a] for a in order], rotation=12)
        ax.set_title(title)

    ax = axes[2]
    for i, a in enumerate(order):
        tr = result["cv"][a]["train_roc_auc"]["mean"]
        va = result["cv"][a]["roc_auc"]["mean"]
        ax.plot([i, i], [va, tr], color=GRID, linewidth=3, zorder=1)
        ax.scatter([i], [tr], s=60, color="white", edgecolor=COLORS[a], linewidth=2, zorder=3,
                   label="Train (fold de ajuste)" if i == 0 else None)
        ax.scatter([i], [va], s=60, color=COLORS[a], zorder=3,
                   label="Validación (fold fuera)" if i == 0 else None)
        ax.annotate(f"brecha {tr - va:+.3f}", (i, tr), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(order)), [NAMES[a] for a in order], rotation=12)
    ax.set_title("Sobreajuste: AUC train vs validación")
    ax.legend(loc="lower left")
    meta = result["meta"]["cv"]
    fig.suptitle(
        f"Validación cruzada estratificada {meta['n_splits']} folds x {meta['n_repeats']} "
        "repeticiones (mismos folds para todos los modelos)",
        fontsize=11, color=MUTED, y=1.02,
    )
    _save(fig, path)


def roc_pr_curves(result: dict[str, Any], y: np.ndarray, proba: dict[str, np.ndarray],
                  path: Path) -> None:
    order = _order(result)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axes[0]
    for a in order:
        fpr, tpr, _ = roc_curve(y, proba[a])
        d = result["holdout"][a]["auc_delong"]
        ax.plot(fpr, tpr, color=COLORS[a], linestyle=STYLES.get(a, "-"),
                label=f"{NAMES[a]}  AUC {d['auc']:.4f} [{d['ci_low']:.3f}-{d['ci_high']:.3f}]")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":", label="Azar (AUC 0.5)")
    ax.set_xlabel("Tasa de falsos positivos (buenos rechazados)")
    ax.set_ylabel("Tasa de verdaderos positivos (defaults detectados)")
    ax.set_title("Curva ROC (holdout) con IC 95 % DeLong")
    ax.legend(loc="lower right")

    ax = axes[1]
    prevalence = y.mean()
    for a in order:
        precision, recall, _ = precision_recall_curve(y, proba[a])
        b = result["holdout"][a]["bootstrap"]["pr_auc"]
        ax.plot(recall, precision, color=COLORS[a], linestyle=STYLES.get(a, "-"),
                label=f"{NAMES[a]}  AP {b['estimate']:.4f} [{b['ci_low']:.3f}-{b['ci_high']:.3f}]")
    ax.axhline(prevalence, color=MUTED, linewidth=1, linestyle=":",
               label=f"Azar (prevalencia {prevalence:.1%})")
    ax.set_xlabel("Recall (defaults detectados)")
    ax.set_ylabel("Precisión")
    ax.set_ylim(0, 1)
    ax.set_title("Curva Precision-Recall (holdout) con IC 95 % bootstrap")
    ax.legend(loc="upper right")
    _save(fig, path)


def calibration(result: dict[str, Any], y: np.ndarray, proba: dict[str, np.ndarray],
                path: Path, n_bins: int = 20) -> None:
    order = _order(result)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    top = 0.0
    for a in order:
        idx = np.array_split(np.argsort(proba[a]), n_bins)
        pred = [proba[a][b].mean() for b in idx]
        obs = [y[b].mean() for b in idx]
        top = max(top, max(pred), max(obs))
        c = result["calibration"][a]
        ax.plot(pred, obs, marker="o", markersize=5, color=COLORS[a],
                linestyle=STYLES.get(a, "-"),
                label=f"{NAMES[a]}  ECE {result['holdout'][a]['ece']:.4f}, pendiente {c['slope']:.2f}")
    lim = min(1.0, top * 1.08)
    ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1, linestyle=":", label="Calibración perfecta")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Probabilidad media predicha (por cuantil)")
    ax.set_ylabel("Tasa de default observada")
    ax.set_title(f"Diagrama de fiabilidad (holdout, {n_bins} cuantiles)")
    ax.legend(loc="upper left")

    ax = axes[1]
    bins = np.linspace(0, 1, 41)
    for a in order:
        ax.hist(proba[a], bins=bins, histtype="step", linewidth=1.8, color=COLORS[a],
                linestyle=STYLES.get(a, "-"), label=NAMES[a])
    ax.set_yscale("log")
    ax.set_xlabel("Probabilidad de default predicha")
    ax.set_ylabel("Solicitantes (escala log)")
    ax.set_title("Distribución de probabilidades predichas")
    ax.legend()
    _save(fig, path)


def separation(champion: str, y: np.ndarray, p: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    bins = np.linspace(0, 1, 51)
    ax.hist(p[y == 0], bins=bins, density=True, alpha=0.55, color=GOOD, label="No default (buenos)")
    ax.hist(p[y == 1], bins=bins, density=True, alpha=0.55, color=BAD, label="Default (malos)")
    ax.set_yscale("log")
    ax.set_xlabel("Probabilidad de default predicha")
    ax.set_ylabel("Densidad (escala log)")
    ax.set_title(f"Separación de clases, {NAMES[champion]} (holdout)")
    ax.legend()

    ax = axes[1]
    grid = np.linspace(0, 1, 1001)
    cdf_good = np.searchsorted(np.sort(p[y == 0]), grid, side="right") / (y == 0).sum()
    cdf_bad = np.searchsorted(np.sort(p[y == 1]), grid, side="right") / (y == 1).sum()
    k = int(np.argmax(np.abs(cdf_good - cdf_bad)))
    ax.plot(grid, cdf_good, color=GOOD, label="CDF buenos")
    ax.plot(grid, cdf_bad, color=BAD, label="CDF malos")
    ax.vlines(grid[k], cdf_bad[k], cdf_good[k], color=INK, linewidth=1.5, linestyle="--")
    ax.annotate(f"KS = {abs(cdf_good[k] - cdf_bad[k]):.3f}\nen p = {grid[k]:.3f}",
                (grid[k], (cdf_good[k] + cdf_bad[k]) / 2), xytext=(12, 0),
                textcoords="offset points", fontsize=9, color=INK, va="center")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel("Probabilidad de default predicha (escala symlog)")
    ax.set_ylabel("Proporción acumulada")
    ax.set_title("Estadístico Kolmogorov-Smirnov")
    ax.legend(loc="lower right")
    _save(fig, path)


def threshold(result: dict[str, Any], curve: list[dict[str, float]], path: Path) -> None:
    th = result["threshold"]
    df = pd.DataFrame(curve)
    df = df[df["threshold"] <= 0.7]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.plot(df["threshold"], df["precision"], color="#4a3aa7", label="Precisión")
    ax.plot(df["threshold"], df["recall"], color="#eb6834", label="Recall")
    ax.plot(df["threshold"], df["f1"], color="#1baf7a", label="F1")
    ax.plot(df["threshold"], df["rejection_rate"], color=MUTED, linestyle="--",
            linewidth=1.5, label="Tasa de rechazo")
    for value, text in ((0.5, "0.50 (default)"), (th["selected_threshold"], "elegido")):
        ax.axvline(value, color=INK, linewidth=1, linestyle=":")
        ax.annotate(text, (value, 0.97), xytext=(4, 0), textcoords="offset points",
                    fontsize=8.5, color=INK, va="top")
    ax.set_xlabel("Umbral de probabilidad")
    ax.set_ylabel("Valor")
    ax.set_ylim(0, 1)
    ax.set_title("Métricas vs umbral (predicciones out-of-fold)")
    ax.legend(loc="center right")

    ax = axes[1]
    ax.plot(df["threshold"], df["expected_cost_per_applicant"], color="#2a78d6")
    best = th["min_cost"]
    ax.scatter([best["threshold"]], [best["expected_cost_per_applicant"]], s=60,
               color="#2a78d6", zorder=3, edgecolor="white", linewidth=1.5)
    ax.annotate(f"mínimo en {best['threshold']:.3f}\n(bayesiano teórico {th['bayes_threshold']:.3f})",
                (best["threshold"], best["expected_cost_per_applicant"]), xytext=(14, 18),
                textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("Umbral de probabilidad")
    ax.set_ylabel("Costo esperado por solicitante")
    ax.set_title(f"Costo esperado (costo FN : FP = {th['cost_fn']:g} : {th['cost_fp']:g})")
    _save(fig, path)


def confusion(result: dict[str, Any], y: np.ndarray, p: np.ndarray, path: Path) -> None:
    th = result["threshold"]["selected_threshold"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, t, title in ((axes[0], 0.5, "Umbral 0.50"), (axes[1], th, f"Umbral elegido {th:.3f}")):
        cm = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1])
        rates = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({rates[i, j]:.1%} de la fila)", ha="center",
                        va="center", fontsize=10,
                        color="white" if rates[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1], ["Pred. aprobar", "Pred. rechazar"])
        ax.set_yticks([0, 1], ["Real: pagó", "Real: default"])
        ax.set_title(title)
    fig.suptitle("Matriz de confusión del champion (holdout)", fontsize=12, y=1.02)
    _save(fig, path)


def gains(result: dict[str, Any], y: np.ndarray, proba: dict[str, np.ndarray], path: Path) -> None:
    order = _order(result)
    champion = result["champion"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    x = np.linspace(0, 1, 201)
    for a in order:
        idx = np.argsort(-proba[a])
        cum = np.cumsum(y[idx]) / y.sum()
        pos = np.arange(1, len(y) + 1) / len(y)
        ax.plot(x, np.interp(x, pos, cum), color=COLORS[a], linestyle=STYLES.get(a, "-"),
                label=NAMES[a])
    prevalence = y.mean()
    ax.plot(x, np.minimum(x / prevalence, 1), color=INK, linewidth=1, linestyle=":",
            label="Modelo perfecto")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":", label="Azar")
    ax.set_xlabel("% de solicitantes revisados (de mayor a menor riesgo)")
    ax.set_ylabel("% de defaults capturados")
    ax.set_title("Curva de ganancia acumulada (holdout)")
    ax.legend(loc="lower right")

    ax = axes[1]
    deciles = result["deciles"]
    rates = [d["bad_rate"] for d in deciles]
    ax.bar([d["decile"] for d in deciles], rates, color=COLORS[champion], width=0.7)
    ax.axhline(prevalence, color=MUTED, linestyle=":", linewidth=1,
               label=f"Tasa media {prevalence:.1%}")
    for d in deciles[:3]:
        ax.annotate(f"{d['bad_rate']:.1%}\nlift {d['lift']:.1f}x", (d["decile"], d["bad_rate"]),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Decil de riesgo (1 = mayor probabilidad predicha)")
    ax.set_ylabel("Tasa de default observada")
    ax.set_title(f"Tasa de default por decil, {NAMES[champion]}")
    ax.legend()
    _save(fig, path)


def strategy(result: dict[str, Any], strategy_full: dict[str, list[dict[str, float]]],
             path: Path) -> None:
    order = _order(result)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for a in order:
        rows = strategy_full[a]
        ax.plot([r["approval_rate"] for r in rows], [r["bad_rate_approved"] for r in rows],
                color=COLORS[a], linestyle=STYLES.get(a, "-"), label=NAMES[a])
    prevalence = result["meta"]["prevalence"]
    ax.axhline(prevalence, color=MUTED, linestyle=":", linewidth=1,
               label=f"Sin modelo (aprobar a todos): {prevalence:.1%}")
    champion_rows = {round(r["approval_rate"], 2): r for r in strategy_full[result["champion"]]}
    for rate in (0.80, 0.90):
        r = champion_rows[rate]
        ax.scatter([rate], [r["bad_rate_approved"]], s=50, color=COLORS[result["champion"]],
                   zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(f"aprobando {rate:.0%}: {r['bad_rate_approved']:.2%} de default",
                    (rate, r["bad_rate_approved"]), xytext=(-150, 14),
                    textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xlabel("Tasa de aprobación (se aprueba el X % de menor riesgo)")
    ax.set_ylabel("Tasa de default de la cartera aprobada")
    ax.set_title("Curva de estrategia: aprobación vs riesgo de cartera (holdout)")
    ax.set_ylim(0, prevalence * 1.25)
    ax.legend(loc="upper left", ncol=2)
    _save(fig, path)


def shap_summary(champion: str, model: Any, x: pd.DataFrame, path: Path,
                 n: int = 3000) -> None:
    import shap

    sample = x.sample(n=min(n, len(x)), random_state=42)
    if champion == "logistic_regression":
        return
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    values = values[1] if isinstance(values, list) else values
    plt.figure(figsize=(9, 5.5))
    shap.summary_plot(values, sample, show=False, plot_size=None, max_display=10)
    plt.title(f"SHAP: impacto de cada feature en el log-odds de default ({NAMES[champion]})",
              fontsize=11)
    plt.gcf().savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")


def learning_curve_plot(champion: str, x: pd.DataFrame, y: pd.Series, path: Path) -> None:
    from sklearn.model_selection import StratifiedKFold, learning_curve

    from src.ml.evaluation import _pipeline

    sizes, train_scores, val_scores = learning_curve(
        _pipeline(champion), x, y, cv=StratifiedKFold(3, shuffle=True, random_state=42),
        train_sizes=[0.05, 0.1, 0.2, 0.4, 0.7, 1.0], scoring="roc_auc", n_jobs=1,
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for scores, label, style in ((train_scores, "Train", "--"), (val_scores, "Validación", "-")):
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, mean, color=COLORS[champion], linestyle=style, marker="o", markersize=5,
                label=label)
        ax.fill_between(sizes, mean - std, mean + std, color=COLORS[champion], alpha=0.12)
    ax.set_xlabel("Tamaño del set de entrenamiento")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"Curva de aprendizaje, {NAMES[champion]} (CV 3 folds, ±1 d.e.)")
    ax.legend()
    _save(fig, path)


def render_all(figures_dir: Path | str, result: dict[str, Any], x_dev: pd.DataFrame,
               y_dev: pd.Series, x_test_imp: pd.DataFrame, y_test: np.ndarray,
               test_proba: dict[str, np.ndarray], oof_champion: np.ndarray,
               threshold_curve: list[dict[str, float]],
               strategy_full: dict[str, list[dict[str, float]]], fitted: dict[str, Any],
               x_dev_imp: pd.DataFrame) -> None:
    out = Path(figures_dir)
    champion = result["champion"]
    cv_distribution(result, out / "01_cv_distribucion.png")
    roc_pr_curves(result, y_test, test_proba, out / "02_roc_pr.png")
    calibration(result, y_test, test_proba, out / "03_calibracion.png")
    separation(champion, y_test, test_proba[champion], out / "04_separacion_ks.png")
    threshold(result, threshold_curve, out / "05_umbral.png")
    confusion(result, y_test, test_proba[champion], out / "06_confusion.png")
    gains(result, y_test, test_proba, out / "07_ganancia_deciles.png")
    strategy(result, strategy_full, out / "08_estrategia.png")
    shap_summary(champion, fitted[champion], x_test_imp, out / "09_shap.png")
    learning_curve_plot(champion, x_dev, y_dev, out / "10_curva_aprendizaje.png")
