"""Tests de contrato de la infraestructura como código (bundle de Databricks y Render)."""

from pathlib import Path

import pytest
import yaml

from credit_risk.config import PROJECT_ROOT, UCNames
from credit_risk.registry.serving_endpoint import build_config

JOBS = yaml.safe_load((PROJECT_ROOT / "resources" / "jobs.yml").read_text(encoding="utf-8"))["resources"][
    "jobs"
]
BUNDLE = yaml.safe_load((PROJECT_ROOT / "databricks.yml").read_text(encoding="utf-8"))


def test_bundle_has_dev_and_prod_targets():
    assert {"dev", "prod"} <= set(BUNDLE["targets"])
    assert BUNDLE["targets"]["dev"]["mode"] == "development"
    assert BUNDLE["targets"]["prod"]["mode"] == "production"


@pytest.mark.parametrize("job_key", list(JOBS))
def test_python_tasks_exist_and_run_serverless(job_key):
    job = JOBS[job_key]
    env_keys = {e["environment_key"] for e in job.get("environments", [])}
    for task in job["tasks"]:
        if "spark_python_task" in task:
            path = (PROJECT_ROOT / "resources" / task["spark_python_task"]["python_file"]).resolve()
            assert path.exists(), path
            assert task["environment_key"] in env_keys  # serverless: sin clusters


@pytest.mark.parametrize("job_key", list(JOBS))
def test_task_dependencies_are_valid(job_key):
    keys = {t["task_key"] for t in JOBS[job_key]["tasks"]}
    for task in JOBS[job_key]["tasks"]:
        for dep in task.get("depends_on", []):
            assert dep["task_key"] in keys


def test_monitoring_closes_the_loop_with_ct():
    tasks = {t["task_key"]: t for t in JOBS["production_monitoring"]["tasks"]}
    cond = tasks["needs_retraining"]["condition_task"]
    assert "monitor_drift.values.retrain" in cond["left"]
    trigger = tasks["trigger_continuous_training"]
    assert trigger["run_job_task"]["job_id"] == "${resources.jobs.ct_training_pipeline.id}"


def test_no_classic_clusters_for_free_edition():
    text = (PROJECT_ROOT / "resources" / "jobs.yml").read_text(encoding="utf-8")
    assert "new_cluster" not in text and "existing_cluster_id" not in text


def test_render_blueprint_services():
    render = yaml.safe_load(Path(PROJECT_ROOT / "render.yaml").read_text(encoding="utf-8"))
    names = {s["name"] for s in render["services"]}
    assert names == {"credit-risk-gateway", "credit-risk-dashboard"}


def test_serving_config_with_and_without_challenger():
    names = UCNames(catalog="workspace", schema="credit_risk")
    solo = build_config(names, "3", None, 0.2)
    assert [e["name"] for e in solo["served_entities"]] == ["champion"]
    assert solo["traffic_config"]["routes"][0]["traffic_percentage"] == 100

    ab = build_config(names, "3", "4", 0.2)
    routes = {r["served_model_name"]: r["traffic_percentage"] for r in ab["traffic_config"]["routes"]}
    assert routes == {"champion": 80, "challenger": 20}
    assert ab["served_entities"][1]["entity_version"] == "4"
    assert ab["served_entities"][0]["entity_name"] == "workspace.credit_risk.credit_default_model"


def test_uc_names():
    n = UCNames(catalog="workspace", schema="credit_risk_dev")
    assert n.inference_log == "workspace.credit_risk_dev.inference_log"
    assert n.volume_path == "/Volumes/workspace/credit_risk_dev/artifacts"
