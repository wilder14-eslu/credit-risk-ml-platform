"""Inferencia de rechazados (reject inference): marco metodológico.

**No se ejecuta sobre "Give Me Some Credit"**: el dataset solo contiene
solicitantes con desempeño observado y ningún rechazado. Nunca se inventan
outcomes de rechazados; estas funciones existen, están probadas con datos
sintéticos y se pueden usar cuando haya una población de rechazados real.
Todas dependen de supuestos que un validador debe revisar (MAR/MNAR).

Técnicas:

* ``reweight_by_acceptance_propensity``: pesos inversos a la probabilidad de
  haber sido aprobado (corrige el sesgo de selección observable).
* ``parceling``: asigna a cada rechazado un outcome simulado según la tasa de
  malos de su banda de score, escalada por ``bad_rate_multiplier`` (supuesto
  del analista: los rechazados suelen ser peores que los aprobados).
* ``fuzzy_augmentation``: duplica cada rechazado como bueno y malo con pesos
  ``1-p`` y ``p``.

No se implementa muestreo por rechazo (*reject sampling*): queda como
trabajo futuro.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def reweight_by_acceptance_propensity(
    x_accepted: pd.DataFrame,
    x_rejected: pd.DataFrame,
    clip: tuple[float, float] = (0.05, 1.0),
    random_state: int = 42,
) -> np.ndarray:
    """Pesos 1/P(aprobado | x) para las filas aprobadas, normalizados a media 1."""
    x_all = pd.concat([x_accepted, x_rejected], ignore_index=True)
    label = np.r_[np.ones(len(x_accepted)), np.zeros(len(x_rejected))]
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(x_all, label)
    propensity = model.predict_proba(x_accepted)[:, 1]
    weights = 1.0 / np.clip(propensity, clip[0], clip[1])
    return weights / weights.mean()


def parceling(
    scores_accepted: Any,
    y_accepted: Any,
    scores_rejected: Any,
    n_bands: int = 10,
    bad_rate_multiplier: float = 1.0,
    random_state: int = 42,
) -> np.ndarray:
    """Outcomes simulados (0/1) para rechazados por banda de score (PD)."""
    rng = np.random.default_rng(random_state)
    sa = np.asarray(scores_accepted, dtype=float)
    ya = np.asarray(y_accepted, dtype=int)
    sr = np.asarray(scores_rejected, dtype=float)
    edges = np.unique(np.quantile(sa, np.linspace(0, 1, n_bands + 1)[1:-1]))
    bands_a = np.digitize(sa, edges)
    bands_r = np.digitize(sr, edges)
    simulated = np.zeros(len(sr), dtype=int)
    for band in np.unique(bands_r):
        in_band = ya[bands_a == band]
        rate = float(in_band.mean()) if len(in_band) else float(ya.mean())
        rate = min(rate * bad_rate_multiplier, 1.0)
        idx = np.where(bands_r == band)[0]
        simulated[idx] = rng.binomial(1, rate, len(idx))
    return simulated


def fuzzy_augmentation(
    x_rejected: pd.DataFrame, pd_rejected: Any
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Cada rechazado aparece dos veces: como malo (peso p) y como bueno (1-p)."""
    p = np.clip(np.asarray(pd_rejected, dtype=float), 0.0, 1.0)
    x_aug = pd.concat([x_rejected, x_rejected], ignore_index=True)
    y_aug = np.r_[np.ones(len(x_rejected)), np.zeros(len(x_rejected))].astype(int)
    weights = np.r_[p, 1.0 - p]
    return x_aug, y_aug, weights
