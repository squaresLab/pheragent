from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .output import AtomicOutputTransaction
from .serialization import load_yaml, write_json, write_text, write_yaml

if TYPE_CHECKING:
    from .analyzer import AnalysisResult
    from .execution import ExecutionReport, PreparedExecution
    from .recovery import RunWorkspace


def publish_analysis_artifacts(
    run_dir: Path,
    result: AnalysisResult,
    *,
    debug: bool = False,
) -> tuple[Path, ...]:
    """Publish the stable Phase 1 artifact contract for product and research callers."""
    with AtomicOutputTransaction(run_dir, write_manifest=False) as transaction:
        staging = transaction.staging_dir
        write_yaml(staging / "functional-blocks.yaml", result.document)
        write_yaml(staging / "deployment-workflow.yaml", result.workflow)
        write_yaml(staging / "unresolved-work.yaml", _unresolved_work(result))
        if result.runtime_context is not None:
            write_json(staging / ".heragent" / "runtime-context.json", result.runtime_context)
        if result.runtime_reconciliation is not None:
            write_json(
                staging / ".heragent" / "runtime-reconciliation.json",
                result.runtime_reconciliation,
            )
        if debug:
            _write_debug_artifacts(staging / ".heragent" / "debug", result)
        return transaction.commit()


def _unresolved_work(result: AnalysisResult) -> dict[str, Any]:
    blocked = [step for step in result.workflow.steps if step.blockers]
    return {
        "system": result.workflow.system,
        "ready_for_execution": result.workflow.ready_for_execution,
        "blocked_steps": [
            {
                "id": step.id,
                "components": [target.id for target in step.targets],
                "reasons": step.blockers,
            }
            for step in blocked
        ],
        "questions": result.workflow.unresolved,
    }


def publish_execution_artifacts(
    run_dir: Path,
    prepared: PreparedExecution,
    report: ExecutionReport,
    workspace: RunWorkspace,
) -> tuple[Path, ...]:
    """Publish only the files an operator needs to review or replay the run."""
    with AtomicOutputTransaction(run_dir, write_manifest=False) as transaction:
        staging = transaction.staging_dir
        blocks = prepared.workflow_path.with_name("functional-blocks.yaml")
        if report.functional_blocks is not None:
            write_yaml(staging / "functional-blocks.yaml", report.functional_blocks)
        elif blocks.is_file():
            write_yaml(staging / blocks.name, load_yaml(blocks))
        write_yaml(
            staging / "deployment-workflow.yaml",
            report.workflow or prepared.workflow,
        )
        write_yaml(staging / "unresolved-work.yaml", _execution_unresolved(report))
        changed = workspace.export_changes(staging / "updated-source")
        write_text(staging / "changes.md", _changes_wiki(report, changed, staging))
        return transaction.commit()


def _execution_unresolved(report: ExecutionReport) -> dict[str, Any]:
    classified = {resolution.step_id: resolution for resolution in report.recoveries}
    return {
        "complete": report.verified,
        "unverified_components": [
            {
                "component": state.component_id,
                "reason": "no successful source-grounded validation was observed",
            }
            for state in report.component_states
            if state.status == "unverified"
        ],
        "failed": [
            {
                "step": issue.step_id,
                "kind": classified[issue.step_id].failure_kind.value,
                "reason": issue.reason,
            }
            if issue.step_id in classified
            else {"step": issue.step_id, "reason": issue.reason}
            for issue in report.failed
        ],
        "skipped": [
            {"step": issue.step_id, "reason": issue.reason} for issue in report.skipped
        ],
    }


def _changes_wiki(
    report: ExecutionReport,
    changed: tuple[Path, ...],
    staging: Path,
) -> str:
    accepted = [
        resolution
        for resolution in report.recoveries
        if resolution.status.value == "resolved"
        and (resolution.patch or resolution.plan_update)
    ]
    lines = ["# Deployment changes", ""]
    if not accepted:
        lines.append("No source changes were accepted.")
        return "\n".join(lines) + "\n"
    for resolution in accepted:
        lines.extend(
            [
                f"## {resolution.step_id}",
                "",
                f"Failure: `{resolution.failure_kind.value}`",
                "",
                resolution.reason,
                "",
            ]
        )
        if resolution.plan_update:
            lines.extend(
                [
                    f"Plan update: `{resolution.plan_update.kind.value}`",
                    "",
                    resolution.plan_update.reason,
                    "",
                ]
            )
    if changed:
        lines.extend(
            [
                "## Updated files",
                "",
                *[
                    f"- `{path.relative_to(staging).as_posix()}`"
                    for path in changed
                ],
            ]
        )
    return "\n".join(lines) + "\n"


def analysis_run_metrics(result: AnalysisResult) -> dict[str, Any]:
    """Return neutral run telemetry; quality scoring belongs to pheragent.evaluation."""
    components = [component for block in result.document.blocks for component in block.components]
    deployable = {component.id for component in components if component.deployable}
    routed = {
        target.id
        for step in result.workflow.steps
        if step.command and not step.blockers
        for target in step.targets
    }
    return {
        "candidate_component_count": len(result.signals.candidate_components),
        "component_count": len(components),
        "deployable_component_count": len(deployable),
        "executable_route_coverage": len(deployable & routed) / len(deployable)
        if deployable
        else 1.0,
        "relation_count": len(result.signals.relations),
        "evidence_observations": len(result.investigation.observations),
        "evidence_characters": result.evidence_characters,
        "retrieval_queries": len(result.retrieval_queries),
        "investigation_synthesis_rounds": result.investigation.synthesis_rounds,
        "investigation_stop_reason": result.investigation.stop_reason,
        "initial_unresolved": result.investigation.initial_unresolved,
        "final_unresolved": (
            len(result.investigation.synthesis.unresolved)
            if result.investigation.synthesis is not None
            else 0
        ),
        "llm_input_tokens": int(result.llm_usage.get("input_tokens", 0)),
        "llm_output_tokens": int(result.llm_usage.get("output_tokens", 0)),
        "llm_requests": int(result.llm_usage.get("requests", 0)),
        "runtime_probes": len(result.runtime_context.probes) if result.runtime_context else 0,
        "runtime_probe_failures": (
            sum(not probe.succeeded for probe in result.runtime_context.probes)
            if result.runtime_context
            else 0
        ),
        "runtime_missing_capabilities": (
            sum(check.status == "missing" for check in result.runtime_reconciliation.checks)
            if result.runtime_reconciliation
            else 0
        ),
        "runtime_unknown_capabilities": (
            sum(check.status == "unknown" for check in result.runtime_reconciliation.checks)
            if result.runtime_reconciliation
            else 0
        ),
    }


def _write_debug_artifacts(debug: Path, result: AnalysisResult) -> None:
    write_json(debug / "repository-index.json", list(result.repository_files))
    write_json(
        debug / "reference-graph.json",
        {
            "nodes": [item.model_dump(mode="json") for item in result.reference_nodes],
            "relations": [item.model_dump(mode="json") for item in result.reference_relations],
        },
    )
    write_json(debug / "component-candidates.json", result.signals.candidate_components)
    write_json(debug / "deployment-signals.json", result.signals)
    write_yaml(debug / "investigation-plan-input.yaml", result.investigation.plan_input)
    write_yaml(debug / "investigation-plan.yaml", result.investigation.plan)
    write_json(debug / "investigation-evidence.json", list(result.investigation.observations))
    write_yaml(
        debug / "investigation-synthesis-input.yaml",
        result.investigation.synthesis_input,
    )
    if result.investigation.synthesis is not None:
        write_yaml(debug / "investigation-synthesis.yaml", result.investigation.synthesis)
