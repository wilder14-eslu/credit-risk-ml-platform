# Política de monitoreo y reentrenamiento

Documento de gobierno del proyecto. Describe qué hace el código hoy y qué
decisiones quedan en manos de personas.

## Principio

**La deriva nunca despliega un modelo.** Las señales de monitoreo solo abren
un *candidato de reentrenamiento*. Entrenar, validar y promover un modelo a
producción son pasos separados, y la promoción exige una aprobación humana
nombrada.

## Señales (implementadas)

| Familia | Métrica | Umbral por defecto | Código |
|---|---|---|---|
| Calidad de datos | faltantes, esquema, valores fuera de rango | +5 pp de faltantes; cualquier columna faltante/inesperada; cualquier valor fuera de rango | `src/monitoring/quality.py` |
| Deriva de variables | PSI por variable | 0.2 | `src/monitoring/drift.py` |
| Deriva de predicciones | PSI de la PD | 0.2 | `src/monitoring/business.py` (`score_drift`) |
| Deriva de desempeño | ROC-AUC, KS, Brier, pendiente de calibración | caída de AUC > 0.05, KS > 0.05, Brier > 0.01, pendiente fuera de [0.8, 1.2] | `src/monitoring/performance.py` |
| Deriva de negocio | tasa de rechazo, composición por decil, default observado | cambio de rechazo > 3 pp | `src/monitoring/business.py` |

`src/monitoring/decision.py::build_retraining_recommendation` convierte las
señales en una recomendación con `auto_deploy = False` y
`requires_human_approval = True` fijos.

## Flujo

1. **Señal** de monitoreo → se abre un candidato de reentrenamiento.
2. **Entrenamiento** del challenger con el mismo protocolo (`make report`).
3. **Validación** champion vs challenger en el mismo holdout
   (`src/governance/champion_challenger.py`): DeLong, margen práctico de
   0.005 de AUC y control de calibración.
4. **Revisión humana** del Model Card y del reporte.
5. **Promoción** solo con `src.ml.validate_model.promote_candidate(approved_by=...)`.
   `evaluate_candidate` nunca modifica el registry.

## Fuera de alcance hoy

- No hay tráfico de producción real en el repositorio: el monitoreo se
  demuestra offline con el holdout y con una ventana **simulada**.
- No hay orquestador programado que ejecute el monitoreo periódicamente.
