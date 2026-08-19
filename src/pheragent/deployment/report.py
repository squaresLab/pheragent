from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analyzer import AnalysisResult


def render_analysis_report(result: AnalysisResult) -> str:
    """Render the human-readable summary for one repository-analysis result."""
    document = result.document
    lines = [
        f"# Deployment Analysis: {document.system}",
        "",
        "## Repositories inspected",
        "",
    ]
    for source in result.acquisition.sources:
        revision = source.manifest.resolved_revision or source.manifest.content_hash
        lines.append(f"- `{source.id}`: `{revision}`")
    lines.extend(
        [
            "",
            "## Technologies detected",
            "",
            ", ".join(result.inventory.detected_technologies) or "None.",
            "",
            "## Deployment entrypoints selected",
            "",
        ]
    )
    for root in result.signals.deployment_roots:
        lines.append(
            f"- `{root.role}`: `{root.source_ref.repo_id}:{root.source_ref.path}` "
            f"(score {root.score:.1f})"
        )
    lines.extend(
        [
            "",
            "## Candidate path",
            "",
            f"- Graph nodes traversed: {len(result.selected_paths)}",
            f"- Candidate components: {len(result.signals.candidate_components)}",
            f"- Bound deployment actions: {len(result.signals.deployment_actions)}",
            f"- Source-derived relations: {document.evaluation.source_derived_relation_count}",
        ]
    )
    for index, level in enumerate(document.levels):
        lines.append(f"- DAG level {index}: {', '.join(f'`{item}`' for item in level)}")
    _append_investigation_summary(lines, result)
    _append_questions_and_warnings(lines, result)
    _append_evaluation_summary(lines, result)
    lines.append("")
    return "\n".join(lines)


def _append_investigation_summary(lines: list[str], result: AnalysisResult) -> None:
    coverage = result.workflow.coverage
    synthesis = result.investigation.synthesis
    lines.extend(
        [
            "",
            "## Evidence-guided investigation",
            "",
            f"- Model-selected queries: {len(result.investigation.plan.queries)}",
            f"- Redacted evidence observations: {len(result.investigation.observations)}",
            "- Prompt-injection observations isolated: "
            f"{sum(item.prompt_injection_detected for item in result.investigation.observations)}",
            "- Dynamic deployment observations blocked: "
            f"{sum(item.blocks_execution for item in result.investigation.observations)}",
            "- Dynamic behavior observations requiring resolution: "
            f"{sum(item.requires_resolution for item in result.investigation.observations)}",
            "- Dynamic behavior observations reviewed: "
            f"{sum(item.dynamic_deployment for item in result.investigation.observations)}",
            f"- Grounded semantic facts: {len(synthesis.facts) if synthesis else 0}",
            "- Documentation/repository disagreements: "
            f"{len(synthesis.disagreements) if synthesis else 0}",
            f"- Ready for execution: {result.workflow.ready_for_execution}",
            "- Execution steps ready: "
            f"{sum(step.status == 'ready' for step in result.workflow.steps)}",
            "- Execution steps blocked: "
            f"{sum(step.status == 'blocked' for step in result.workflow.steps)}",
            f"- External requirements: {len(result.workflow.external_requirements)}",
            f"- Grounded validation checks: {len(result.workflow.validation_checks)}",
            f"- Primary roots traced: {coverage.primary_roots_traced}",
            f"- Deployment actions accounted: {coverage.deployment_actions_accounted}",
            f"- Independent gap search complete: {coverage.independent_gap_search_complete}",
            "- Documentation/repository disagreements reviewed: "
            f"{coverage.docs_repo_disagreements_reviewed}",
            "- Documentation/repository disagreements resolved: "
            f"{coverage.docs_repo_disagreements_resolved}",
            f"- External requirements reviewed: {coverage.external_requirements_reviewed}",
            f"- Dynamic deployments resolved: {coverage.dynamic_deployments_resolved}",
            f"- Deployable components grounded: {coverage.deployable_components_grounded}",
            f"- Semantic coverage complete: {coverage.semantic_coverage_complete}",
            f"- Validation checks grounded: {coverage.validation_checks_grounded}",
            f"- Documentation evidence reviewed: {coverage.documentation_evidence_reviewed}",
        ]
    )


def _append_questions_and_warnings(lines: list[str], result: AnalysisResult) -> None:
    lines.extend(["", "## Unresolved questions", ""])
    lines.extend(
        [f"- **{item.question}** {item.reason}" for item in result.workflow.unresolved] or ["None."]
    )
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in result.warnings] or ["None."])


def _append_evaluation_summary(lines: list[str], result: AnalysisResult) -> None:
    evaluation = result.document.evaluation
    lines.extend(
        [
            "",
            "## Evaluation summary",
            "",
            f"- Component candidates: {evaluation.candidate_component_count}",
            f"- Components: {evaluation.component_count}",
            f"- Implementation details suppressed: {evaluation.implementation_detail_count}",
            f"- Uncertain components: {evaluation.uncertain_component_count}",
            f"- Deployability coverage: {evaluation.deployability_coverage:.1%}",
            f"- Grounded component rate: {evaluation.grounded_component_rate:.1%}",
            f"- Discovery closure: {evaluation.discovery_closure:.1%}",
            f"- Executable route coverage: {evaluation.executable_route_coverage:.1%}",
            f"- Orphan components: {evaluation.orphan_component_count}",
            f"- Independently grounded: {evaluation.independently_grounded}",
            f"- Forbidden components: {evaluation.forbidden_component_count}",
            f"- Relations: {evaluation.relation_count}",
            f"- Artifact lines: {evaluation.artifact_line_count}",
            f"- LLM input tokens: {evaluation.llm_input_tokens}",
            f"- LLM output tokens: {evaluation.llm_output_tokens}",
            f"- LLM requests this run: {evaluation.llm_requests}",
            "- LLM stages: "
            + "; ".join(f"{stage}={status}" for stage, status in result.llm_stage_statuses.items()),
        ]
    )
    for path in result.llm_failure_history_paths:
        lines.append(f"- LLM failure history: `{path}`")
    for label, value in (
        ("Component precision", evaluation.component_precision),
        ("Component recall", evaluation.component_recall),
        ("Classification accuracy", evaluation.classification_accuracy),
        ("Edge precision", evaluation.edge_precision),
        ("Edge recall", evaluation.edge_recall),
        ("Entrypoint accuracy", evaluation.entrypoint_accuracy),
        ("Pairwise grouping F1", evaluation.grouping_f1),
    ):
        if value is not None:
            lines.append(f"- {label}: {value:.1%}")
    lines.append(f"- Hallucination rate: {evaluation.hallucination_rate:.1%}")
