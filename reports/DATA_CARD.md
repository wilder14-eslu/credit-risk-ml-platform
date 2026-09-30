# Data Card: Give Me Some Credit

_Generada automáticamente por `python -m src.pipelines.full_report`._

| Campo | Valor |
|---|---|
| Fuente | Kaggle, competencia "Give Me Some Credit" (`cs-training.csv`) |
| Archivo / sha256 | `DATA/GiveMeSomeCredit/cs-training.csv` / `1bd46da486a5708c58c7b01a034fae2a13b327f6f7b62ea7ba4fe3b5824b24ac` |
| Filas tras limpieza | 149,999 (se descarta 1 fila con edad fuera de rango) |
| Variables | 10 numéricas |
| Objetivo | `SeriousDlqin2yrs`: atraso de 90+ días en los 2 años siguientes |
| Prevalencia de default | 6.68 % (10,026 defaults) |
| Filas duplicadas | 609 (se conservan) |
| Variable temporal | ninguna (no es posible validar fuera de tiempo) |
| Población rechazada | no incluida (no es posible inferencia de rechazados) |
| LGD / EAD | no incluidos |

## Variables

| Variable | Columna original | % faltante | Mín | Mediana | P99 | Máx |
|---|---|---|---|---|---|---|
| `revolving_utilization_unsecured` | `RevolvingUtilizationOfUnsecuredLines` | 0.0 % | 0 | 0.1542 | 1.093 | 5.071e+04 |
| `age` | `age` | 0.0 % | 21 | 52 | 87 | 109 |
| `number_of_time_30_59_days_past_due` | `NumberOfTime30-59DaysPastDueNotWorse` | 0.0 % | 0 | 0 | 4 | 98 |
| `debt_ratio` | `DebtRatio` | 0.0 % | 0 | 0.3665 | 4979 | 3.297e+05 |
| `monthly_income` | `MonthlyIncome` | 19.8 % | 0 | 5400 | 2.5e+04 | 3.009e+06 |
| `number_open_credit_lines` | `NumberOfOpenCreditLinesAndLoans` | 0.0 % | 0 | 8 | 24 | 58 |
| `number_of_times_90_days_late` | `NumberOfTimes90DaysLate` | 0.0 % | 0 | 0 | 3 | 98 |
| `number_real_estate_loans` | `NumberRealEstateLoansOrLines` | 0.0 % | 0 | 1 | 4 | 54 |
| `number_of_time_60_89_days_past_due` | `NumberOfTime60-89DaysPastDueNotWorse` | 0.0 % | 0 | 0 | 2 | 98 |
| `number_dependents` | `NumberOfDependents` | 2.6 % | 0 | 0 | 4 | 20 |

## Códigos centinela 96/98 en conteos de atraso

| Variable | Filas | Tasa de default en esas filas |
|---|---|---|
| `number_of_time_30_59_days_past_due` | 269 | 54.6 % |
| `number_of_time_60_89_days_past_due` | 269 | 54.6 % |
| `number_of_times_90_days_late` | 269 | 54.6 % |

## Consideraciones

- `age` es un atributo potencialmente protegido en regulación de crédito; se usa como variable del modelo y como único eje de diagnóstico de equidad.
- `MonthlyIncome` y `NumberOfDependents` tienen faltantes: se imputan con medianas ajustadas solo en entrenamiento (modelos ML) o forman un bin propio (scorecard).
- Colas extremas en `DebtRatio` y `RevolvingUtilizationOfUnsecuredLines`: robustas en árboles; la regresión logística usa transformación por cuantiles y el scorecard bins.
- Licencia: términos de la competencia de Kaggle; revisar antes de redistribuir.
