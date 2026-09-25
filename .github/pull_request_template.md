## Qué cambia

## Checklist
- [ ] `ruff check .` y `ruff format --check .` en verde
- [ ] `pytest` en verde (cobertura >= 80%)
- [ ] Si cambian umbrales, actualicé `config/platform.yaml` y la documentación
- [ ] `databricks bundle validate -t dev` sin errores (si toqué `resources/` o `databricks.yml`)
