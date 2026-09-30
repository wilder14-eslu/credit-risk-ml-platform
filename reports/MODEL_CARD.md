# Model Card: modelo de probabilidad de default (PD)

_Generada automáticamente por `python -m src.pipelines.full_report`._

## 1. Detalles del modelo

| Campo | Valor |
|---|---|
| Champion estadístico | CatBoost (mayor ROC-AUC medio en CV) |
| Modelo servido por la demo/API | XGBoost (valor por defecto de `src/ml/train.py`) |
| Benchmark interpretable | Scorecard WoE + regresión logística (`src/risk/scorecard.py`) |
| Salida | Probabilidad de default a 2 años (`SeriousDlqin2yrs`) |
| Umbral de decisión | 0.155 (mínimo costo, FN:FP = 5:1, elegido con predicciones out-of-fold) |
| Versión de código | `7b106d4` (con cambios sin commit al generar) |
| Dataset | `DATA/GiveMeSomeCredit/cs-training.csv` sha256 `1bd46da486a5708c...` |
| Generada | 2026-09-30T20:46:33+00:00 · Python 3.11.15 |
| Semilla | 42 (split, folds, bootstrap y modelos) |

## 2. Uso previsto

- Proyecto de portafolio que demuestra metodología de riesgo de crédito, validación estadística y MLOps sobre un dataset público.
- Ordenar solicitantes por riesgo y estimar una PD calibrada para análisis de escenarios.

## 3. Fuera de alcance

- Decisiones de crédito reales sobre personas: el modelo no fue validado por un área de riesgo de modelos ni aprobado por un regulador.
- Cálculo de provisiones o capital regulatorio (no hay LGD, EAD ni validación fuera de tiempo).
- Poblaciones distintas a la del dataset (otro país, producto o período).

## 4. Desempeño (holdout 20 %, evaluado una sola vez)

| Modelo | ROC-AUC [IC 95 % DeLong] | PR-AUC | KS | Brier | ECE | Pendiente calib. |
|---|---|---|---|---|---|---|
| CatBoost | 0.8704 [0.8624, 0.8785] | 0.4066 | 0.5823 | 0.0487 | 0.0027 | 1.019 |
| XGBoost | 0.8696 [0.8614, 0.8777] | 0.4047 | 0.5831 | 0.0487 | 0.0025 | 1.008 |
| LightGBM | 0.8674 [0.8592, 0.8756] | 0.3976 | 0.5785 | 0.0491 | 0.0028 | 0.981 |
| Regresión Logística | 0.8462 [0.8367, 0.8557] | 0.3621 | 0.5455 | 0.0514 | 0.0120 | 1.001 |
| Scorecard WoE | 0.8594 [0.8508, 0.8680] | 0.3798 | 0.5660 | 0.0504 | 0.0060 | 1.003 |

Con el umbral elegido, el champion detecta 58.2 % de los defaults con precisión 36.3 % y rechaza 10.7 % de las solicitudes del holdout.

## 5. Equidad (solo edad disponible)

| Grupo | n | Tasa de rechazo [IC 95 %] | TPR | FPR | Default observado | PD media |
|---|---|---|---|---|---|---|
| 18-29 | 1,803 | 20.3 % [18.5 %, 22.2 %] | 64.3 % | 14.0 % | 12.6 % | 11.6 % |
| 30-39 | 4,658 | 18.1 % [17.0 %, 19.2 %] | 65.3 % | 12.6 % | 10.3 % | 10.3 % |
| 40-49 | 6,930 | 14.1 % [13.3 %, 15.0 %] | 60.4 % | 9.8 % | 8.5 % | 8.5 % |
| 50-59 | 6,958 | 9.4 % [8.7 %, 10.1 %] | 52.7 % | 6.6 % | 6.1 % | 6.1 % |
| 60-69 | 5,813 | 4.8 % [4.2 %, 5.3 %] | 47.7 % | 3.3 % | 3.4 % | 3.7 % |
| 70+ | 3,838 | 2.4 % [2.0 %, 3.0 %] | 37.2 % | 1.7 % | 2.2 % | 2.3 % |

> Diferencias descriptivas. No se atribuyen causas: los grupos difieren en riesgo base y en otras variables. Solo 'age' está disponible como atributo potencialmente sensible.

## 6. Explicabilidad

Importancia global (media de |SHAP| en log-odds) del champion:

- `revolving_utilization_unsecured`: 0.582 (Spearman valor-SHAP +0.95)
- `number_of_time_30_59_days_past_due`: 0.309 (Spearman valor-SHAP +0.64)
- `age`: 0.249 (Spearman valor-SHAP -0.98)
- `number_of_times_90_days_late`: 0.190 (Spearman valor-SHAP +0.40)
- `number_open_credit_lines`: 0.155 (Spearman valor-SHAP +0.85)

> Los valores SHAP describen cómo usa cada variable el modelo ajustado (asociación). No muestran que cambiar una variable cambie el riesgo real.

## 7. Stress testing (análisis de escenarios, no pronóstico)

| Escenario | PD media | Cambio vs base | Tasa de rechazo |
|---|---|---|---|
| base | 6.68 % | +0.0% | 10.7 % |
| moderate_stress | 8.52 % | +27.4% | 15.9 % |
| severe_stress | 10.36 % | +55.0% | 20.5 % |

## 8. Champion vs challenger

- Champion vigente: xgboost (por defecto en src/ml/train.py, usado por la demo) · Challenger: catboost (mayor ROC-AUC en CV)
- Δ AUC (challenger - champion) = +0.0009, p DeLong = 0.138, margen práctico = 0.005
- Recomendación: **Mantener el champion: no hay mejora estadísticamente significativa.**
- `auto_promote = False`: ninguna promoción ocurre sin aprobación humana.

## 9. Monitoreo y política de reentrenamiento

- Calidad de datos (faltantes, esquema, valores inválidos), deriva de variables (PSI), deriva de predicciones (PSI del score), deriva de desempeño (AUC, PR-AUC, KS, Brier, calibración) y deriva de negocio (aprobación, composición de cartera, default observado).
- Las señales solo abren un **candidato de reentrenamiento**. La deriva nunca despliega un modelo: la promoción requiere `approved_by` (ver `docs/governance/RETRAINING_POLICY.md`).

## 10. Limitaciones

- **out_of_time**: Out-of-time validation cannot be performed with the current dataset.
- **reject_inference**: El dataset solo contiene solicitantes con desempeño observado; no existe población rechazada. El marco está en src/risk/reject_inference.py y solo se probó con datos sintéticos.
- **lgd_ead_expected_loss**: El dataset no tiene LGD, EAD, saldos ni recuperaciones. src/risk/expected_loss.py es un marco parametrizable que se niega a calcular sin insumos explícitos.
- **fairness_scope**: Solo hay edad; no existen género, estado civil, etnia ni geografía, así que la equidad solo puede evaluarse por edad.
- Split aleatorio (no temporal): el desempeño en producción suele ser menor.
- Costos FN:FP ilustrativos; deben reemplazarse por pérdidas reales.

## 11. Versiones de librerías

numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, scipy 1.17.1, xgboost 3.2.0, lightgbm 4.7.0, catboost 1.2.10, optuna 5.0.0, shap 0.51.0, mlflow 3.16.1, fastapi 0.142.2, streamlit 1.64.0
