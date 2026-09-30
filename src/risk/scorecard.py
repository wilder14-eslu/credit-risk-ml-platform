"""Scorecard tradicional: binning, WoE, Information Value, regresión
logística sobre WoE y escalado a puntos.

Es el benchmark interpretable que se compara con los modelos de ML. Todo se
ajusta **solo con datos de desarrollo**; el holdout únicamente se transforma.

Convenciones (Siddiqi, 2017):

* ``WoE = ln(%buenos / %malos)`` por bin; positivo = menor riesgo.
* ``IV = sum((%buenos - %malos) * WoE)``.
* Escalado: ``score = offset + factor * ln(odds_buenos/malos)`` con
  ``factor = PDO / ln 2`` y ``offset = base_score - factor * ln(base_odds)``.
  Por defecto 600 puntos equivalen a odds 50:1 y cada 20 puntos duplican los
  odds (parámetros, no resultados).
* Cada bin debe tener al menos ``min_share`` (1 %) de las observaciones. Un
  mínimo de 5 % (habitual en carteras balanceadas) fusiona aquí los atrasos
  1, 2, 3+ con el 0, porque cada uno pesa menos del 5 %, y destruye la señal
  más predictiva del dataset. Con 120 mil filas, 1 % son unas 1,200.
* Los valores faltantes y los códigos centinela (p. ej. 96/98 en los
  conteos de atraso) forman bins propios, así que no hace falta imputar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

MISSING_CODE = 9999
SPECIAL_BASE_CODE = 1000  # 1000 + índice del valor especial
_SMOOTHING = 0.5


@dataclass
class FeatureBinning:
    """Bins de una variable: intervalos ``[e_i, e_{i+1})`` + faltante + especiales."""

    name: str
    edges: list[float]
    special_values: list[float] = field(default_factory=list)
    woe: dict[int, float] = field(default_factory=dict)
    iv: float = 0.0
    table: list[dict[str, Any]] = field(default_factory=list)

    def assign(self, values: pd.Series) -> np.ndarray:
        """Código de bin por fila: intervalo 0..k-1, especial 1000+i, faltante 9999."""
        v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        codes = np.full(len(v), MISSING_CODE, dtype=int)
        valid = ~np.isnan(v)
        if self.edges:
            codes[valid] = np.digitize(v[valid], self.edges)
        else:
            codes[valid] = 0
        for i, special in enumerate(self.special_values):
            codes[valid & (v == special)] = SPECIAL_BASE_CODE + i
        return codes


def _candidate_edges(values: np.ndarray, max_bins: int, max_discrete: int = 30) -> list[float]:
    """Fronteras candidatas, siempre entre dos valores observados distintos.

    * Variables discretas (<= ``max_discrete`` valores distintos, p. ej. conteos
      de atraso): una frontera entre cada par de valores consecutivos; luego
      ``_merge_small_bins`` agrupa las colas poco pobladas (3+, 4+, ...).
    * Continuas: cuantiles, desplazados al punto medio con el siguiente valor
      observado. Así los empates masivos (94 % de ceros) no generan bins vacíos.
    """
    uniques = np.unique(values)
    if len(uniques) <= 1:
        return []
    if len(uniques) <= max_discrete:
        return [float(x) for x in (uniques[:-1] + uniques[1:]) / 2.0]
    edges = set()
    for q in np.quantile(values, np.linspace(0, 1, max_bins + 1)[1:-1]):
        above = uniques[uniques > q]
        if len(above):
            edges.add(float((q + above[0]) / 2.0))
    return sorted(edges)


def _merge_small_bins(
    values: np.ndarray, edges: list[float], min_share: float
) -> list[float]:
    """Une bins adyacentes con menos de ``min_share`` de las observaciones."""
    edges = list(edges)
    n = len(values)
    while edges:
        codes = np.digitize(values, edges)
        counts = np.bincount(codes, minlength=len(edges) + 1)
        smallest = int(np.argmin(counts))
        if counts[smallest] / n >= min_share:
            break
        # Se elimina la frontera con el vecino más pequeño.
        if smallest == 0:
            drop = 0
        elif smallest == len(counts) - 1:
            drop = len(edges) - 1
        else:
            drop = smallest - 1 if counts[smallest - 1] <= counts[smallest + 1] else smallest
        edges.pop(drop)
    return edges


def fit_binning(
    name: str,
    values: pd.Series,
    target: np.ndarray,
    max_bins: int = 10,
    min_share: float = 0.01,
    special_values: tuple[float, ...] = (),
) -> FeatureBinning:
    """Bins por cuantiles (o por valor si hay pocos distintos) + WoE e IV."""
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    regular_mask = ~np.isnan(v)
    for special in special_values:
        regular_mask &= v != special
    regular = v[regular_mask]

    edges = _candidate_edges(regular, max_bins) if len(regular) else []
    edges = _merge_small_bins(regular, edges, min_share) if edges else []
    binning = FeatureBinning(name=name, edges=edges, special_values=list(special_values))

    codes = binning.assign(values)
    y = np.asarray(target, dtype=int)
    total_good = float((y == 0).sum())
    total_bad = float((y == 1).sum())
    present = np.unique(codes)
    iv_total = 0.0
    for code in present:
        mask = codes == code
        good = float(((y == 0) & mask).sum())
        bad = float(((y == 1) & mask).sum())
        dist_good = (good + _SMOOTHING) / (total_good + _SMOOTHING * len(present))
        dist_bad = (bad + _SMOOTHING) / (total_bad + _SMOOTHING * len(present))
        woe = float(np.log(dist_good / dist_bad))
        iv = float((dist_good - dist_bad) * woe)
        iv_total += iv
        binning.woe[int(code)] = woe
        binning.table.append(
            {
                "feature": name,
                "bin": _describe_bin(binning, int(code)),
                "code": int(code),
                "count": int(mask.sum()),
                "share": float(mask.mean()),
                "bad_rate": float(bad / max(good + bad, 1.0)),
                "woe": woe,
                "iv_contribution": iv,
            }
        )
    binning.iv = iv_total
    binning.table.sort(key=lambda row: (row["code"] >= SPECIAL_BASE_CODE, row["code"]))
    return binning


def _describe_bin(binning: FeatureBinning, code: int) -> str:
    if code == MISSING_CODE:
        return "faltante"
    if code >= SPECIAL_BASE_CODE:
        return f"especial = {binning.special_values[code - SPECIAL_BASE_CODE]:g}"
    edges = binning.edges
    if not edges:
        return "todos"
    if code == 0:
        return f"< {edges[0]:.4g}"
    if code == len(edges):
        return f">= {edges[-1]:.4g}"
    return f"[{edges[code - 1]:.4g}, {edges[code]:.4g})"


def information_value_label(iv: float) -> str:
    """Regla empírica habitual (Siddiqi): poder predictivo según el IV."""
    if iv < 0.02:
        return "inútil"
    if iv < 0.1:
        return "débil"
    if iv < 0.3:
        return "medio"
    if iv < 0.5:
        return "fuerte"
    return "muy alto (> 0.5: verificar posible fuga)"


class CreditScorecard:
    """Scorecard WoE + regresión logística con escalado a puntos."""

    def __init__(
        self,
        pdo: float = 20.0,
        base_score: float = 600.0,
        base_odds: float = 50.0,
        max_bins: int = 10,
        min_share: float = 0.01,
        regularization_c: float = 1.0,
        special_values: dict[str, tuple[float, ...]] | None = None,
        random_state: int = 42,
    ) -> None:
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.max_bins = max_bins
        self.min_share = min_share
        self.regularization_c = regularization_c
        self.special_values = special_values or {}
        self.random_state = random_state
        self.factor = pdo / np.log(2.0)
        self.offset = base_score - self.factor * np.log(base_odds)
        self.binnings: dict[str, FeatureBinning] = {}
        self.model: LogisticRegression | None = None
        self.features: list[str] = []

    # -- ajuste --------------------------------------------------------
    def fit(self, x: pd.DataFrame, y: Any) -> CreditScorecard:
        target = np.asarray(y, dtype=int)
        self.features = list(x.columns)
        self.binnings = {
            name: fit_binning(
                name,
                x[name],
                target,
                self.max_bins,
                self.min_share,
                tuple(self.special_values.get(name, ())),
            )
            for name in self.features
        }
        woe = self._woe_matrix(x)
        self.model = LogisticRegression(
            C=self.regularization_c, max_iter=1000, random_state=self.random_state
        )
        self.model.fit(woe, target)
        return self

    def _woe_matrix(self, x: pd.DataFrame) -> np.ndarray:
        columns = []
        for name in self.features:
            binning = self.binnings[name]
            codes = binning.assign(x[name])
            columns.append(np.array([binning.woe.get(int(c), 0.0) for c in codes]))
        return np.column_stack(columns)

    # -- predicción ----------------------------------------------------
    def _check_fitted(self) -> LogisticRegression:
        if self.model is None:
            raise RuntimeError("El scorecard no está ajustado; llama a fit().")
        return self.model

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Matriz (n, 2) compatible con la API de scikit-learn."""
        return self._check_fitted().predict_proba(self._woe_matrix(x))

    def predict_pd(self, x: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(x)[:, 1]

    def score(self, x: pd.DataFrame) -> np.ndarray:
        """Puntaje: mayor = menor riesgo."""
        return self.pd_to_score(self.predict_pd(x))

    def pd_to_score(self, pd_values: Any) -> np.ndarray:
        p = np.clip(np.asarray(pd_values, dtype=float), 1e-9, 1 - 1e-9)
        return self.offset + self.factor * np.log((1.0 - p) / p)

    def score_to_pd(self, scores: Any) -> np.ndarray:
        s = np.asarray(scores, dtype=float)
        return 1.0 / (1.0 + np.exp((s - self.offset) / self.factor))

    # -- reportes ------------------------------------------------------
    def iv_table(self) -> list[dict[str, Any]]:
        rows = [
            {
                "feature": name,
                "iv": binning.iv,
                "strength": information_value_label(binning.iv),
                "n_bins": len(binning.table),
            }
            for name, binning in self.binnings.items()
        ]
        return sorted(rows, key=lambda row: row["iv"], reverse=True)

    def points_table(self) -> list[dict[str, Any]]:
        """Puntos por bin: ``-(coef * WoE + intercept/n) * factor + offset/n``."""
        model = self._check_fitted()
        n = len(self.features)
        intercept = float(model.intercept_[0])
        rows = []
        for j, name in enumerate(self.features):
            coef = float(model.coef_[0][j])
            for entry in self.binnings[name].table:
                points = -(coef * entry["woe"] + intercept / n) * self.factor + self.offset / n
                rows.append({**entry, "coefficient": coef, "points": float(points)})
        return rows
