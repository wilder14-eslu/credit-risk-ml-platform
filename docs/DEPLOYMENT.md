# Guía de despliegue

Esta guía deja la plataforma funcionando de punta a punta con cuentas gratuitas:
Databricks Free Edition, Render (plan free) y GitHub Actions.

## 1. Databricks Free Edition

1. Crea tu cuenta en <https://www.databricks.com/learn/free-edition>. El catálogo por defecto es
   `workspace` (ya configurado en `databricks.yml`).
2. **Token personal**: *Settings → Developer → Access tokens → Generate new token*. Guárdalo.
3. **SQL Warehouse**: Free Edition trae uno (2X-Small). En *SQL Warehouses* copia su **ID**
   (lo usa el gateway para escribir y leer las tablas Delta).
4. Instala la CLI y autentícate:

   ```bash
   # Windows: winget install Databricks.DatabricksCLI   |   macOS/Linux: ver docs de la CLI
   databricks auth login --host https://<tu-workspace>.cloud.databricks.com
   ```

5. Despliega y ejecuta en **dev** (esquema `credit_risk_dev`, schedules pausados):

   ```bash
   databricks bundle validate -t dev
   databricks bundle deploy -t dev
   databricks bundle run -t dev ct_training_pipeline
   ```

   El primer run crea las tablas, entrena los 4 candidatos, registra el ganador como
   `workspace.credit_risk_dev.credit_default_model@champion` y crea el endpoint
   `credit-risk-endpoint` (tarda ~10-20 min la primera vez).

   > Si tu cuenta no permite Model Serving, despliega con
   > `databricks bundle deploy -t dev --var="deploy_serving=false"`: todo lo demás funciona y el
   > gateway puede usar `SERVING_BACKEND=local`.

6. Simula producción y drift:

   ```bash
   databricks bundle run -t dev production_monitoring                               # tráfico normal
   databricks bundle run -t dev production_monitoring --params scenario=covariate   # data drift
   databricks bundle run -t dev production_monitoring --params scenario=concept     # concept drift -> CT
   ```

7. Para **prod** (esquema `credit_risk`, schedules activos): `databricks bundle deploy -t prod`
   (normalmente lo hace el pipeline de CD).

## 2. GitHub Actions (CI/CD)

En *Settings → Secrets and variables → Actions* crea:

| Secreto | Valor |
|---|---|
| `DATABRICKS_HOST` | `https://<tu-workspace>.cloud.databricks.com` |
| `DATABRICKS_TOKEN` | token personal del paso 1.2 |
| `RENDER_DEPLOY_HOOK_GATEWAY` | deploy hook del servicio gateway (paso 3) |
| `RENDER_DEPLOY_HOOK_DASHBOARD` | deploy hook del dashboard (paso 3) |

En *Settings → Environments* crea `dev` y `prod`; en `prod` activa **Required reviewers**
para aprobar manualmente cada promoción a producción.

Sin secretos, CI sigue en verde (lint, tests, smoke test, validación de YAML) y CD se omite
con un aviso: el repo nunca queda en rojo por falta de credenciales.

## 3. Render

1. *New → Blueprint* y selecciona este repositorio: Render lee `render.yaml` y crea
   `credit-risk-gateway` y `credit-risk-dashboard`.
2. En el gateway completa `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, `DATABRICKS_WAREHOUSE_ID` y
   ajusta `UC_SCHEMA` (`credit_risk` para prod). `API_KEYS` se genera sola y el dashboard la
   recibe automáticamente.
3. En cada servicio: *Settings → Deploy Hook* → copia la URL a los secretos de GitHub.
4. Prueba:

   ```bash
   curl https://credit-risk-gateway.onrender.com/health
   curl -X POST https://credit-risk-gateway.onrender.com/api/v1/predict \
     -H "X-API-Key: <API_KEYS>" -H "Content-Type: application/json" \
     -d '{"applicant_id":"APP-1","features":{"revolving_utilization_unsecured":0.4,"age":35,"debt_ratio":0.3,"monthly_income":4000,"number_open_credit_lines":6,"number_dependents":1}}'
   ```

> El plan free de Render suspende el servicio tras 15 min sin tráfico (primer request lento) y
> el endpoint de Databricks escala a cero: la primera predicción puede tardar ~1 min.

## 4. Operación

| Acción | Comando |
|---|---|
| Reentrenar manualmente | `databricks bundle run -t prod ct_training_pipeline --params trigger=manual` |
| Rollback del modelo | `databricks bundle run -t prod model_rollback` |
| Cambiar % de tráfico A/B | editar `ab_testing.challenger_traffic` en `config/platform.yaml` y hacer push |
| Cambiar umbrales de drift | editar `monitoring` en `config/platform.yaml` y hacer push |
