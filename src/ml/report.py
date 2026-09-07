"""Human-readable training report: metrics tables, fit diagnosis and the
top drivers of the model, exportable as Markdown and as a printable PDF.

The narrative paragraph is built from a deterministic template by default
(no external dependency, fully reproducible in CI and offline). If
``ANTHROPIC_API_KEY`` is set and the optional ``anthropic`` package is
installed, ``build_report_sections(..., use_llm=True)`` instead asks Claude
to turn the same structured metrics into a more natural summary -- see the
"Informe de entrenamiento" section of the README for why Claude (and which
model) is the recommended choice here.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_METRIC_ROWS: tuple[tuple[str, str], ...] = (
    ("ROC-AUC", "roc_auc"),
    ("PR-AUC", "pr_auc"),
    ("Gini", "gini"),
    ("KS", "ks_statistic"),
    ("Precisión", "precision"),
    ("Recall", "recall"),
    ("F1", "f1"),
    ("Log loss", "log_loss"),
    ("Brier score (calibración)", "brier_score"),
)


def top_feature_importances(
    model: Any, feature_names: list[str], top_n: int = 5
) -> list[dict[str, Any]]:
    """Global feature importance for the report, agnostic to the algorithm:
    tree models expose ``feature_importances_``; the Logistic Regression
    baseline (wrapped in a `Pipeline`) exposes ``coef_`` on its last step.
    """
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = list(model.named_steps.values())[-1]

    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        coefficients = getattr(estimator, "coef_", None)
        if coefficients is not None:
            importances = [abs(value) for value in coefficients[0]]
    if importances is None:
        return []

    ranked = sorted(zip(feature_names, importances), key=lambda pair: pair[1], reverse=True)
    return [{"feature": name, "impact": float(value)} for name, value in ranked[:top_n]]


def _format_metrics_table(train_metrics: dict[str, float], test_metrics: dict[str, float]) -> list[tuple[str, str, str]]:
    return [
        (label, f"{train_metrics[key]:.3f}", f"{test_metrics[key]:.3f}")
        for label, key in _METRIC_ROWS
    ]


def _default_narrative(result: dict[str, Any]) -> str:
    diagnosis = result["diagnosis"]
    test_metrics = result["test_metrics"]
    return (
        f"El modelo ({result['algorithm']}) obtuvo un ROC-AUC de prueba de "
        f"{test_metrics['roc_auc']:.3f} (PR-AUC {test_metrics['pr_auc']:.3f}, "
        f"Gini {test_metrics['gini']:.3f}, KS {test_metrics['ks_statistic']:.3f}), con una "
        f"latencia de inferencia de {result['latency_ms_per_row']:.3f} ms por solicitante. "
        f"{diagnosis['explanation']}"
    )


def generate_narrative_with_llm(
    result: dict[str, Any], model: str = "claude-haiku-4-5"
) -> str | None:
    """Best-effort: ask Claude to turn the metrics into a short prose summary.

    Returns ``None`` (never raises) when ``ANTHROPIC_API_KEY`` is not set or
    the ``anthropic`` package is not installed, so the report always falls
    back to `_default_narrative`. A small, fast model (Claude Haiku) is
    enough here: the task is short structured-data summarization in
    Spanish, not open-ended reasoning, so Haiku keeps this optional step
    cheap; use Sonnet instead if you want richer, more explanatory prose
    for a report handed to non-technical stakeholders.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        logger.info("`anthropic` no está instalado; se usa el resumen por plantilla.")
        return None

    try:
        client = anthropic.Anthropic()
        prompt = (
            "Redacta un párrafo breve (máximo 120 palabras), en español, para un "
            "informe técnico de un modelo de riesgo crediticio, explicando de forma "
            "clara para un lector no técnico estas métricas de entrenamiento vs "
            f"prueba: {result['train_metrics']} (entrenamiento) vs "
            f"{result['test_metrics']} (prueba). Diagnóstico automático: "
            f"{result['diagnosis']['label']} -- {result['diagnosis']['explanation']}"
        )
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception:  # pragma: no cover - depends on a live API call
        logger.exception("No se pudo generar la narrativa con Claude; se usa la plantilla.")
        return None


def build_report_sections(
    result: dict[str, Any],
    top_features: list[dict[str, Any]] | None = None,
    monitoring_status: dict[str, Any] | None = None,
    use_llm: bool = False,
) -> dict[str, Any]:
    narrative = (generate_narrative_with_llm(result) if use_llm else None) or _default_narrative(
        result
    )
    return {
        "title": "Informe de entrenamiento - Credit Risk ML Platform",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": result["algorithm"],
        "run_id": result.get("run_id"),
        "narrative": narrative,
        "diagnosis": result["diagnosis"],
        "latency_ms_per_row": result["latency_ms_per_row"],
        "metrics_table": _format_metrics_table(result["train_metrics"], result["test_metrics"]),
        "top_features": top_features or [],
        "monitoring_status": monitoring_status,
    }


def render_markdown(sections: dict[str, Any]) -> str:
    lines = [
        f"# {sections['title']}",
        "",
        f"_Generado: {sections['generated_at']}_",
        "",
        f"**Algoritmo:** {sections['algorithm']}  ",
        f"**Diagnóstico:** {sections['diagnosis']['label']} "
        f"(brecha train-test: {sections['diagnosis']['gap']})  ",
        f"**Latencia de inferencia:** {sections['latency_ms_per_row']:.3f} ms/solicitante",
        "",
        sections["narrative"],
        "",
        "## Métricas de entrenamiento vs prueba",
        "",
        "| Métrica | Train | Test |",
        "|---|---|---|",
    ]
    lines += [f"| {name} | {train} | {test} |" for name, train, test in sections["metrics_table"]]
    lines.append("")

    if sections["top_features"]:
        lines += ["## Factores más influyentes", ""]
        lines += [
            f"- **{item['feature']}**: impacto {item['impact']:.4f}"
            for item in sections["top_features"]
        ]
        lines.append("")

    if sections["monitoring_status"]:
        lines += [
            "## Estado de monitoreo al momento del entrenamiento",
            "",
            "```",
            str(sections["monitoring_status"]),
            "```",
            "",
        ]

    return "\n".join(lines)


def render_pdf(sections: dict[str, Any], output_path: Path | str) -> Path:
    """Render the same structured report as a printable PDF.

    Uses `fpdf2` (pure Python, no system dependency like wkhtmltopdf/Cairo)
    so this works the same on Windows, in CI and inside the Docker images.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 10, sections["title"], new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 6, f"Generado: {sections['generated_at']}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(
        0,
        8,
        f"Algoritmo: {sections['algorithm']}  |  Diagnóstico: {sections['diagnosis']['label']}"
        f"  |  Latencia: {sections['latency_ms_per_row']:.3f} ms/solicitante",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, sections["narrative"], new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(0, 8, "Métricas de entrenamiento vs prueba", new_x="LMARGIN", new_y="NEXT")
    col_widths = (80, 50, 50)
    pdf.set_font("Helvetica", "B", 10)
    for header, width in zip(("Métrica", "Train", "Test"), col_widths):
        pdf.cell(width, 8, header, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 10)
    for name, train, test in sections["metrics_table"]:
        for value, width in zip((name, train, test), col_widths):
            pdf.cell(width, 8, value, border=1)
        pdf.ln()
    pdf.ln(4)

    if sections["top_features"]:
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 8, "Factores más influyentes", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for item in sections["top_features"]:
            pdf.multi_cell(
                0,
                6,
                f"- {item['feature']}: impacto {item['impact']:.4f}",
                new_x="LMARGIN",
                new_y="NEXT",
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    return output_path
