# Informe de evaluación de modelos (auto-generado)

_Generado: 2026-09-27T15:25:39+00:00 con `python -m src.ml.evaluation`_

- Filas: 149,999 | Prevalencia de default: 6.68% | Desarrollo: 119,999 | Holdout: 30,000
- CV: 5 folds x 3 repeticiones (estratificada) | Bootstrap: 1000 remuestras
- Champion (por ROC-AUC medio en CV): **CatBoost**

## Validación cruzada (media ± desv. estándar [IC 95 % t corregido Nadeau-Bengio])

| Modelo | ROC-AUC | PR-AUC | KS | Brier | ECE | Brecha train-val |
|---|---|---|---|---|---|---|
| CatBoost | 0.8652 ± 0.0052 [0.8590, 0.8715] | 0.4032 ± 0.0113 [0.3896, 0.4169] | 0.5771 ± 0.0113 [0.5634, 0.5908] | 0.0489 | 0.0036 | +0.0082 |
| XGBoost | 0.8646 ± 0.0049 [0.8586, 0.8705] | 0.4009 ± 0.0101 [0.3888, 0.4131] | 0.5747 ± 0.0095 [0.5632, 0.5861] | 0.0491 | 0.0034 | +0.0156 |
| LightGBM | 0.8631 ± 0.0049 [0.8572, 0.8690] | 0.3975 ± 0.0101 [0.3853, 0.4097] | 0.5731 ± 0.0103 [0.5607, 0.5856] | 0.0492 | 0.0034 | +0.0476 |
| Regresión Logística | 0.8400 ± 0.0066 [0.8320, 0.8479] | 0.3627 ± 0.0127 [0.3474, 0.3779] | 0.5345 ± 0.0116 [0.5205, 0.5485] | 0.0511 | 0.0106 | +0.0003 |

## Tests vs champion

| Modelo | Δ AUC CV | p (Nadeau-Bengio) | Δ AUC holdout | p (DeLong) |
|---|---|---|---|---|
| XGBoost | +0.0007 | 0.223 | +0.0009 | 0.138 |
| LightGBM | +0.0021 | 0.00442 | +0.0031 | 0.000494 |
| Regresión Logística | +0.0253 | 4.89e-09 | +0.0243 | 2.99e-21 |

## Holdout (20 %, evaluado una sola vez)

| Modelo | ROC-AUC [IC 95 % DeLong] | PR-AUC [IC boot] | KS [IC boot] | Brier | ECE | Pendiente calib. | Latencia p50 (ms) |
|---|---|---|---|---|---|---|---|
| CatBoost | 0.8704 [0.8624, 0.8785] | 0.4066 [0.3817, 0.4283] | 0.5823 [0.5672, 0.6023] | 0.0487 | 0.0027 | 1.019 | 0.41 |
| XGBoost | 0.8696 [0.8614, 0.8777] | 0.4047 [0.3802, 0.4281] | 0.5831 [0.5678, 0.6025] | 0.0487 | 0.0025 | 1.008 | 1.51 |
| LightGBM | 0.8674 [0.8592, 0.8756] | 0.3976 [0.3739, 0.4207] | 0.5785 [0.5642, 0.5995] | 0.0491 | 0.0028 | 0.981 | 0.73 |
| Regresión Logística | 0.8462 [0.8367, 0.8557] | 0.3621 [0.3412, 0.3857] | 0.5455 [0.5300, 0.5673] | 0.0514 | 0.0120 | 1.001 | 2.53 |

## Umbral de decisión

- Elegido en predicciones out-of-fold (costo FN:FP = 5:1): **0.155** (umbral bayesiano teórico 0.167)
- Holdout con ese umbral: precisión 0.363, recall 0.582, F1 0.447, tasa de rechazo 10.7%

## Tabla de deciles (holdout, champion)

| Decil | Score mín-máx | Tasa default | Lift | % defaults capturados (acum.) | KS |
|---|---|---|---|---|---|
| 1 | 0.167-0.916 | 37.47% | 5.61 | 56.1% | 0.494 |
| 2 | 0.081-0.167 | 11.70% | 1.75 | 73.6% | 0.574 |
| 3 | 0.050-0.081 | 6.83% | 1.02 | 83.8% | 0.576 |
| 4 | 0.032-0.050 | 3.87% | 0.58 | 89.6% | 0.531 |
| 5 | 0.021-0.032 | 2.33% | 0.35 | 93.1% | 0.462 |
| 6 | 0.015-0.021 | 1.77% | 0.26 | 95.7% | 0.383 |
| 7 | 0.012-0.015 | 1.23% | 0.18 | 97.6% | 0.295 |
| 8 | 0.009-0.012 | 0.77% | 0.11 | 98.7% | 0.200 |
| 9 | 0.007-0.009 | 0.50% | 0.07 | 99.5% | 0.101 |
| 10 | 0.004-0.007 | 0.37% | 0.05 | 100.0% | 0.000 |
