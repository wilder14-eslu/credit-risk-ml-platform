# Auditoría técnica del repositorio

Fecha: 30/09/2026. Base: commit `7b106d4`. Todas las cifras citadas provienen de `reports/`.

## 1. Auditoría inicial (antes de modificar)

Línea base verificada: 42 tests pasando, Ruff limpio y la evaluación existente **se reprodujo exactamente** (AUC holdout CatBoost 0.8704, CV 0.8652, umbral 0.155).

| Área | Estado inicial | Gap | Riesgo | Acción | Prioridad |
|---|---|---|---|---|---|
| Validación | CV repetida 5x3, holdout aislado, DeLong, Nadeau-Bengio | Sin CV anidada ni tuning; sin corrección por comparaciones múltiples | Comparaciones múltiples sin control de FWER | CV anidada + Optuna; Holm; tamaños de efecto | Alta |
| Riesgo de crédito | PD, KS, Gini, deciles, umbral 5:1 | Sin scorecard WoE/IV, sin costos por ratio, sin EL, sin stress | Proyecto poco "bancario" | Scorecard, costos 1:1-20:1, marco EL, stress | Alta |
| Gobernanza | Gate champion/challenger | `evaluate_and_promote` promovía a Production **automáticamente** | Deriva → despliegue sin revisión | Separar evaluación y promoción con `approved_by` | Alta |
| Monitoreo | PSI por variable, AUC en vivo, tasa de rechazo | Sin calidad de datos, deriva de score, calibración ni negocio | Deriva no detectada | 5 familias de señales | Media |
| Explicabilidad | SHAP summary + local en API | Sin dependencia ni interacciones | Explicación incompleta | Dependencia, local, interacción | Media |
| IA responsable | Limitación de `age` mencionada | Sin métricas por grupo | Sesgo no medido | Diagnóstico por edad con IC | Media |
| Estabilidad | d.e. entre folds | Sin semillas ni segmentos | Estimación frágil | Semillas, segmentos, resumen IQR/IC | Media |
| Reproducibilidad | `make evaluate` | Sin comando único para todo; sin procedencia | Resultados no trazables | `make report`, Model/Data Card, sha256, commit | Media |
| README | Técnico, extenso, sin mapa de evidencia | Difícil de escanear para un recruiter | Valor no percibido | Resultados primero, skills → evidencia | Media |

## 2. Cambios realizados

- **Nuevo** `src/inference/`: Holm, d de Cohen pareado, significancia práctica, comparación pareada en CV y holdout.
- **Nuevo** `src/risk/`: scorecard WoE/IV con escalado a puntos, costos por ratio FN:FP, cartera y segmentos, equidad con IC de Wilson, stress testing, marco EL, OOT preparado, inferencia de rechazados (solo marco).
- **Nuevo** `src/ml/nested_cv.py` y `src/ml/stability.py`.
- **Nuevo** `src/monitoring/quality.py`, `performance.py`, `business.py`; `decision.py` con recomendación `auto_deploy = False`.
- **Corregido** `src/ml/validate_model.py`: `evaluate_candidate` (sin efectos) + `promote_candidate(approved_by=...)`.
- **Nuevo** `src/governance/` y `src/pipelines/full_report.py` (`make report`).
- **Salidas** movidas a `reports/` (antes `docs/results` y `docs/images/results`).
- **Bug propio detectado y corregido** antes de reportar: el binning del scorecard colapsaba variables con 94 % de ceros en un solo bin (IV de atrasos de 90 días = 0.04). Tras la corrección: IV 0.88, AUC del scorecard 0.831 → 0.859. Test de regresión agregado.

## 3. Tests y experimentos ejecutados

- `pytest`: **96 tests pasando** (42 originales, 1 reescrito por el cambio de gate, 54 nuevos). `ruff check .`: sin errores.
- Evaluación base CV 5x3 + holdout (reproducida).
- CV anidada 5x3 con Optuna, 12 trials, 4 algoritmos, más tuning final y holdout.
- Reporte completo: scorecard, 10 comparaciones con Holm, 6 ratios de costo, 10 semillas x 4 modelos, 3 segmentaciones, equidad por edad, 3 escenarios de stress, monitoreo con 2 ventanas, SHAP.

## 4. Auditoría final

| Área | Estado | Evidencia (archivo / test / resultado) |
|---|---|---|
| 1. Arquitectura | IMPLEMENTADO | `src/` por dominio; `src/pipelines/full_report.py`; diagrama en README |
| 2. Estadística | IMPLEMENTADO | `src/ml/stats.py`, `src/inference/`; `tests/test_stats.py`, `tests/test_inference.py` (Holm validado contra statsmodels); `reports/results.json → comparisons` |
| 3. Machine learning | IMPLEMENTADO | 4 algoritmos + CV anidada Optuna; `tests/test_nested_cv.py`; `reports/nested_cv.json` |
| 4. Riesgo de crédito | PARCIAL | PD, scorecard, KS/Gini/lift, costos, stress: implementados. EL solo como marco (no hay LGD/EAD) |
| 5. Validación de modelos | PARCIAL | CV, anidada, holdout, estabilidad: implementados. OOT imposible (sin fechas) |
| 6. Explicabilidad | IMPLEMENTADO | `src/ml/explainability.py`; figuras 09, 23-25; test SHAP en `tests/test_governance.py` |
| 7. IA responsable | PARCIAL | Equidad por edad (`src/risk/fairness.py`, figura 20). Otros atributos no existen en el dataset |
| 8. MLOps | PARCIAL | MLflow, FastAPI desplegada, CI, monitoreo de 5 familias. Sin orquestador programado ni tráfico real |
| 9. Ingeniería de software | IMPLEMENTADO | Type hints, docstrings, 96 tests, Ruff, CI |
| 10. Reproducibilidad | IMPLEMENTADO | `make nested-cv`, `make report`, semilla 42, sha256 del dataset y commit en `reports/MODEL_CARD.md` |
| 11. Legibilidad para recruiters | IMPLEMENTADO | README: resultados primero, qué demuestra, skills → evidencia, resumen de portafolio |

No implementado (documentado como limitación): validación fuera de tiempo, inferencia de rechazados sobre datos reales, pérdida esperada con LGD/EAD reales, muestreo de rechazados, Docker, orquestación programada del monitoreo.
