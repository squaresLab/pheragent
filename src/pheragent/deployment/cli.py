from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .analysis_llm import (
    DEFAULT_ANALYSIS_MODEL,
    AnalysisLLMConfig,
    ClassificationOutcome,
    LLMRequestBudget,
)
from .analyzer import AnalysisConfig, AnalysisResult, run_repository_analysis
from .artifacts import analysis_metrics, publish_analysis_artifacts
from .enums import AnalysisTreatment
from .errors import DeploymentError, DeploymentInputError
from .execution import ExecutionReport, prepare_execution
from .output import create_timestamped_run_directory
from .recovery import RecoveryAgent, ThreadedRecoveryQueue
from .run_records import RunRecorder
from .runtime_context import (
    RuntimeInspectionConfig,
    inspect_runtime_context,
    load_runtime_context,
)
from .serialization import write_json

_PRODUCT_ANALYSIS_POLICY = "deployment-analysis-v1"
_PRODUCT_ANALYSIS_METHOD = AnalysisTreatment.HYBRID


def add_deployment_parser(subparsers: Any) -> None:
    deployment = subparsers.add_parser(
        "deployment",
        help="Inspect deployment sources and review deployment artifacts.",
    )
    commands = deployment.add_subparsers(dest="deployment_command", required=True)
    _add_runtime_parser(commands)
    _add_analyze_parser(commands)
    _add_run_parser(commands)


def _add_runtime_parser(commands: Any) -> None:
    inspect = commands.add_parser(
        "inspect-runtime",
        help="Save read-only AWS and Kubernetes environment observations.",
    )
    inspect.add_argument("--output", required=True, type=Path)
    inspect.add_argument("--aws-profile", default=None)
    inspect.add_argument("--aws-region", default=None)
    inspect.add_argument("--kube-context", default=None)
    inspect.add_argument("--timeout", type=_positive_float, default=30.0)


def _add_analyze_parser(commands: Any) -> None:
    analyze = commands.add_parser(
        "analyze",
        help="Discover a compact functional deployment block graph without executing sources.",
    )
    analyze.add_argument("--repo", action="append", default=[])
    analyze.add_argument("--docs", action="append", default=[])
    analyze.add_argument("--context", required=True, type=Path)
    analyze.add_argument(
        "--runtime-context",
        type=Path,
        default=None,
        help="Snapshot produced by deployment inspect-runtime.",
    )
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument(
        "--run-name",
        default=None,
        help="Label for the timestamped run folder; defaults to the context system name.",
    )
    analyze.add_argument("--node-budget", type=_positive_int, default=200)
    analyze.add_argument("--source-timeout", type=float, default=900.0)
    analyze.add_argument("--strict", action="store_true")
    analyze.add_argument("--model", default=None)
    analyze.add_argument("--openai-base-url", default=None)
    analyze.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    analyze.add_argument("--openai-base-url-env", default="OPENAI_BASE_URL")
    analyze.add_argument("--llm-timeout", type=float, default=120.0)
    analyze.add_argument("--llm-max-tokens", type=_positive_int, default=5000)
    analyze.add_argument(
        "--llm-reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        default=None,
        help="Optional reasoning effort for models that support it.",
    )
    analyze.add_argument(
        "--llm-max-requests",
        type=_positive_int,
        default=2,
        help=(
            "Hard cap for investigation planning and synthesis requests; "
            "set at least 3 to enable one unresolved-question follow-up."
        ),
    )
    analyze.add_argument(
        "--investigation-max-observations",
        type=_positive_int,
        default=32,
        help="Maximum redacted evidence observations sent to grounded synthesis.",
    )
    analyze.add_argument(
        "--investigation-max-evidence-chars",
        type=_positive_int,
        default=18000,
        help="Character budget for all source evidence sent to grounded synthesis.",
    )
    analyze.add_argument(
        "--retry-failed-llm",
        action="store_true",
        help="Retry an identical request recorded in the failed-LLM history.",
    )
    analyze.add_argument(
        "--refresh-llm",
        action="store_true",
        help="Make fresh LLM requests while preserving repository and LLM caches.",
    )
    analyze.add_argument(
        "--debug",
        action="store_true",
        help="Write detailed repository and signal artifacts beneath debug/.",
    )


def _add_run_parser(commands: Any) -> None:
    run = commands.add_parser(
        "run",
        help="Preview or execute an approved deployment workflow on this node.",
    )
    run.add_argument("workflow", type=Path)
    run.add_argument(
        "--source-root",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="Local checkout used by workflow source ID; repeat for multiple repositories.",
    )
    run.add_argument(
        "--execute",
        action="store_true",
        help="Execute the reviewed plan; without this flag the command is always a dry-run.",
    )
    run.add_argument(
        "--allow-unready",
        action="store_true",
        help=(
            "Trial mode: select grounded ready steps even when the overall workflow is unready; "
            "blocked steps and their dependents remain excluded."
        ),
    )
    run.add_argument(
        "--block",
        default=None,
        metavar="BLOCK_ID",
        help=(
            "Run only one discovered functional block whose prerequisites are provided; "
            "reads functional-blocks.yaml beside the workflow."
        ),
    )
    run.add_argument(
        "--approve",
        default=None,
        metavar="TOKEN",
        help="Approval token printed by an identical dry-run; required with --execute.",
    )
    run.add_argument("--command-timeout", type=_positive_float, default=900.0)
    run.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Execution-run root; defaults beside the analysis runs directory.",
    )
    run.add_argument("--repair-model", default=DEFAULT_ANALYSIS_MODEL)
    run.add_argument("--repair-workers", type=_positive_int, default=2)
    run.add_argument("--max-repair-attempts", type=_positive_int, default=2)
    run.add_argument("--repair-max-requests", type=_positive_int, default=8)
    run.add_argument("--repair-max-tokens", type=_positive_int, default=3000)
    run.add_argument("--repair-llm-timeout", type=_positive_float, default=120.0)
    run.add_argument(
        "--repair-reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        default="low",
    )


def run_deployment_command(args: argparse.Namespace) -> int:
    try:
        if args.deployment_command == "inspect-runtime":
            return _run_runtime_inspection(args)
        if args.deployment_command == "analyze":
            return _run_analyze(args)
        if args.deployment_command == "run":
            return _run_workflow(args)
        raise DeploymentInputError(f"unsupported deployment command: {args.deployment_command}")
    except (
        DeploymentError,
        OSError,
        RuntimeError,
        ValueError,
        ValidationError,
        yaml.YAMLError,
    ) as exc:
        print(f"deployment error: {exc}", file=sys.stderr)
        return 1


def _run_runtime_inspection(args: argparse.Namespace) -> int:
    snapshot = inspect_runtime_context(
        RuntimeInspectionConfig(
            aws_profile=args.aws_profile,
            aws_region=args.aws_region,
            kube_context=args.kube_context,
            timeout=args.timeout,
        )
    )
    output = args.output.expanduser().resolve()
    write_json(output, snapshot)
    succeeded = sum(probe.succeeded for probe in snapshot.probes)
    print(f"runtime context: {output}")
    print(f"read-only probes: {succeeded}/{len(snapshot.probes)} succeeded")
    for warning in snapshot.warnings:
        print(f"runtime warning: {warning}", file=sys.stderr)
    return 0 if snapshot.aws.available or snapshot.kubernetes.available else 1


def _run_analyze(args: argparse.Namespace) -> int:
    output_root = args.output.expanduser().resolve()
    run_name = args.run_name or _context_system_name(args.context)
    run_dir = create_timestamped_run_directory(output_root, name=run_name)
    recorder = RunRecorder.start(
        run_dir,
        run_kind="product",
        analysis_method=_PRODUCT_ANALYSIS_POLICY,
        inputs={
            "repositories": args.repo,
            "documentation": args.docs,
            "context": args.context,
            "runtime_context": args.runtime_context,
            "model": args.model,
            "budgets": {
                "node": args.node_budget,
                "llm_requests": args.llm_max_requests,
                "llm_output_tokens": args.llm_max_tokens,
                "evidence_observations": args.investigation_max_observations,
                "evidence_characters": args.investigation_max_evidence_chars,
            },
        },
    )

    def progress(message: str) -> None:
        recorder.record_event("analysis", message)
        print(f"analyze: {message}", file=sys.stderr, flush=True)

    try:
        result = run_repository_analysis(
            _analysis_config(args, output_root),
            progress=progress,
        )
        published = publish_analysis_artifacts(run_dir, result, debug=args.debug)
        for outcome in _analysis_outcomes(result):
            _record_llm_call(recorder, outcome, model=_analysis_model(args))
        recorder.complete(
            metrics=analysis_metrics(result),
            sources=result.acquisition.manifest.model_dump(mode="json"),
            llm={
                "usage": result.llm_usage,
                "stages": result.llm_stage_statuses,
            },
        )
    except Exception as exc:
        recorder.fail(exc)
        progress(f"failed; run directory retained at {run_dir}")
        raise

    published_count = len(tuple(path for path in run_dir.rglob("*") if path.is_file()))
    _print_analysis_summary(run_dir, result, max(len(published), published_count))
    return 0


def _analysis_config(args: argparse.Namespace, output_root: Path) -> AnalysisConfig:
    return AnalysisConfig(
        repositories=args.repo,
        documentation=args.docs,
        context_path=args.context,
        cache_dir=output_root / ".source-cache",
        node_budget=args.node_budget,
        strict=args.strict,
        source_timeout=args.source_timeout,
        gold_path=None,
        model=_analysis_model(args),
        api_key_env=args.openai_api_key_env,
        base_url_env=args.openai_base_url_env,
        base_url=args.openai_base_url,
        llm_timeout=args.llm_timeout,
        llm_max_output_tokens=args.llm_max_tokens,
        llm_reasoning_effort=args.llm_reasoning_effort,
        llm_max_requests=args.llm_max_requests,
        llm_cache_dir=output_root / ".llm-cache",
        retry_failed_llm=args.retry_failed_llm,
        refresh_llm=args.refresh_llm,
        investigation_max_observations=args.investigation_max_observations,
        investigation_max_evidence_chars=args.investigation_max_evidence_chars,
        treatment=_PRODUCT_ANALYSIS_METHOD,
        runtime_context=(
            load_runtime_context(args.runtime_context) if args.runtime_context else None
        ),
    )


def _print_analysis_summary(
    run_dir: Path,
    result: AnalysisResult,
    published_count: int,
) -> None:
    evaluation = result.document.evaluation
    print(f"run: {run_dir}")
    print(f"functional blocks: {run_dir / 'functional-blocks.yaml'}")
    print(f"analysis report: {run_dir / 'analysis-report.md'}")
    print(f"deployment workflow: {run_dir / 'deployment-workflow.yaml'}")
    if result.runtime_context is not None:
        print(f"runtime context: {run_dir / 'runtime-context.json'}")
    print(
        "LLM stages: "
        + "; ".join(f"{stage}={status}" for stage, status in result.llm_stage_statuses.items())
    )
    print(f"ready for execution: {str(result.workflow.ready_for_execution).lower()}")
    print(f"LLM requests this run: {evaluation.llm_requests}")
    outcomes = [
        result.investigation.plan_outcome,
        result.investigation.synthesis_outcome,
    ]
    if (
        result.investigation.follow_up_outcome is not None
        and result.investigation.follow_up_outcome is not result.investigation.synthesis_outcome
    ):
        outcomes.append(result.investigation.follow_up_outcome)
    for outcome in outcomes:
        if outcome.value is None and outcome.warning:
            print(f"LLM {outcome.stage}: {outcome.warning}")
    print(f"follow-up synthesis: {result.investigation.follow_up_status}")
    for failure_path in result.llm_failure_history_paths:
        print(f"LLM failure history: {failure_path}")
    print(
        f"components: {evaluation.component_count} from "
        f"{evaluation.candidate_component_count} candidate(s); "
        f"deployability coverage: {evaluation.deployability_coverage:.1%}; "
        f"forbidden components: {evaluation.forbidden_component_count}"
    )
    print(f"published files: {published_count}")


def _context_system_name(context_path: Path) -> str:
    payload = yaml.safe_load(context_path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("system"), str):
        raise ValueError("deployment context must define a system name")
    system = payload["system"].strip()
    if not system:
        raise ValueError("deployment context system name cannot be empty")
    return system


def _analysis_model(args: argparse.Namespace) -> str:
    return (
        args.model
        or os.getenv("PHERAGENT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or DEFAULT_ANALYSIS_MODEL
    )


def _run_workflow(args: argparse.Namespace) -> int:
    source_roots = _parse_source_roots(args.source_root)
    prepared = prepare_execution(
        args.workflow,
        source_roots,
        allow_unready=args.allow_unready,
        block_id=args.block,
    )
    if not args.execute:
        print(prepared.render(), end="")
        return 0 if prepared.executable else 1
    if not args.approve:
        raise DeploymentInputError("--execute requires the approval token printed by a dry-run")
    output_root = args.output.expanduser().resolve() if args.output else _execution_output_root(
        args.workflow
    )
    run_dir = create_timestamped_run_directory(
        output_root,
        name=f"{prepared.workflow.system}-{args.block or 'all'}",
    )
    recorder = RunRecorder.start(
        run_dir,
        run_kind="deployment_execution",
        analysis_method="queued-recovery-v1",
        inputs={
            "workflow": args.workflow,
            "source_roots": source_roots,
            "block": args.block,
            "allow_unready": args.allow_unready,
            "repair": {
                "model": args.repair_model,
                "workers": args.repair_workers,
                "max_attempts": args.max_repair_attempts,
                "max_requests": args.repair_max_requests,
            },
        },
    )

    def progress(message: str) -> None:
        recorder.record_event("execution", message)
        print(f"execute: {message}", file=sys.stderr, flush=True)

    budget = LLMRequestBudget(args.repair_max_requests)
    agent = RecoveryAgent.create(
        AnalysisLLMConfig(
            model=args.repair_model,
            timeout=args.repair_llm_timeout,
            max_output_tokens=args.repair_max_tokens,
            max_requests=args.repair_max_requests,
            cache_dir=output_root / ".llm-cache",
            reasoning_effort=args.repair_reasoning_effort,
        ),
        budget,
        sandbox_root=output_root / ".recovery-workspaces" / run_dir.name,
    )
    try:
        with ThreadedRecoveryQueue(agent.resolve, workers=args.repair_workers) as recovery_queue:
            report = prepared.execute(
                approval_token=args.approve,
                timeout=args.command_timeout,
                progress=progress,
                recovery_queue=recovery_queue,
                max_repair_attempts=args.max_repair_attempts,
                output_directory=run_dir,
            )
        for resolution in report.recoveries:
            for call in resolution.llm_calls:
                recorder.record_event(
                    "llm",
                    str(call["stage"]),
                    {"model": args.repair_model, **call},
                )
        recorder.complete(
            metrics=_execution_metrics(report),
            sources={source_id: path for source_id, path in source_roots.items()},
            llm={
                "model": args.repair_model,
                "usage": _recovery_usage(report),
                "requests_attempted": budget.attempted,
            },
        )
    except Exception as exc:
        recorder.fail(exc)
        raise
    print(f"execution run: {run_dir}")
    if report.successful:
        print(f"execution complete: {len(report.completed)} operation(s)")
        return 0
    print(
        "execution incomplete: "
        f"{len(report.completed)} completed, {len(report.failed)} failed, "
        f"{len(report.skipped)} skipped; selected block was not deployed",
        file=sys.stderr,
    )
    for issue in report.failed:
        print(f"failed: {issue.step_id}: {issue.reason}", file=sys.stderr)
    for issue in report.skipped:
        print(f"skipped: {issue.step_id}: {issue.reason}", file=sys.stderr)
    return 1


def _execution_output_root(workflow: Path) -> Path:
    resolved = workflow.expanduser().resolve()
    if resolved.parent.parent.name == "runs":
        return resolved.parent.parent.parent / "execution-runs"
    return Path(".pheragent/deployment-executions").resolve()


def _execution_metrics(report: ExecutionReport) -> dict[str, Any]:
    return {
        "completed_steps": len(report.completed),
        "failed_steps": len(report.failed),
        "skipped_steps": len(report.skipped),
        "execution_attempts": len(report.attempts),
        "failures_captured": len(report.failures_seen),
        "repair_attempts": len(report.recoveries),
        "repair_successes": sum(
            resolution.status.value == "resolved" for resolution in report.recoveries
        ),
        "duration_seconds": round(
            sum(attempt.duration_seconds for attempt in report.attempts),
            6,
        ),
    }


def _recovery_usage(report: ExecutionReport) -> dict[str, int]:
    usage: dict[str, int] = {}
    for recovery in report.recoveries:
        for key, value in recovery.usage.items():
            usage[key] = usage.get(key, 0) + int(value)
    return usage


def _analysis_outcomes(result: AnalysisResult) -> list[ClassificationOutcome[Any]]:
    outcomes = [result.investigation.plan_outcome, result.investigation.synthesis_outcome]
    follow_up = result.investigation.follow_up_outcome
    if follow_up is not None and all(follow_up is not outcome for outcome in outcomes):
        outcomes.append(follow_up)
    return outcomes


def _record_llm_call(
    recorder: RunRecorder,
    outcome: ClassificationOutcome[Any],
    *,
    model: str,
) -> None:
    recorder.record_event(
        "llm",
        outcome.stage,
        {
            "model": model,
            "status": outcome.status,
            "usage": outcome.usage,
            "input_tokens_estimate": outcome.input_tokens_estimate,
            "duration_seconds": outcome.duration_seconds,
        },
    )


def _parse_source_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        source_id, separator, raw_path = value.partition("=")
        if not separator or not source_id.strip() or not raw_path.strip():
            raise ValueError("--source-root must use ID=PATH")
        source_id = source_id.strip()
        if source_id in roots:
            raise ValueError(f"duplicate --source-root ID: {source_id}")
        roots[source_id] = Path(raw_path.strip())
    return roots


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed
