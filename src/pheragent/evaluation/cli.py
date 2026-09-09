from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from pheragent.deployment.analysis_llm import DEFAULT_ANALYSIS_MODEL
from pheragent.deployment.serialization import write_json

from ._judge import PhaseOneJudgeConfig
from .phase_one import (
    MetricResult,
    PhaseOneEvaluationInput,
    PhaseOneEvaluationReport,
    evaluate_phase_one,
)


def add_evaluation_parser(subparsers: Any) -> None:
    evaluation = subparsers.add_parser(
        "evaluation",
        help="Evaluate sealed HerAgent artifacts without rerunning deployment analysis.",
    )
    commands = evaluation.add_subparsers(dest="evaluation_command", required=True)
    phase_one = commands.add_parser(
        "phase-one",
        help="Evaluate component discovery and deployment-plan artifacts.",
    )
    phase_one.add_argument("--run", action="append", required=True, type=Path)
    phase_one.add_argument(
        "--source-root",
        action="append",
        required=True,
        metavar="ID=PATH",
        help="Pinned local source root used by the run; repeat for multiple sources.",
    )
    phase_one.add_argument("--context", required=True, type=Path)
    phase_one.add_argument("--output", required=True, type=Path)
    phase_one.add_argument(
        "--no-llm",
        action="store_true",
        help="Run only deterministic checks; semantic metrics remain a conservative baseline.",
    )
    phase_one.add_argument("--model", default=None)
    phase_one.add_argument("--openai-base-url", default=None)
    phase_one.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    phase_one.add_argument("--openai-base-url-env", default="OPENAI_BASE_URL")
    phase_one.add_argument("--llm-timeout", type=_positive_float, default=120.0)
    phase_one.add_argument("--llm-max-tokens", type=_positive_int, default=5000)
    phase_one.add_argument(
        "--llm-max-requests",
        type=_positive_int,
        default=4,
        help=(
            "Hard LLM request limit per evaluated run; the default covers up to three "
            "component batches plus completeness."
        ),
    )
    phase_one.add_argument(
        "--llm-max-evidence-chars",
        type=_positive_int,
        default=18000,
        help=(
            "Redacted evidence budget shared by component batches and separately available "
            "to completeness."
        ),
    )
    phase_one.add_argument(
        "--llm-reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
        default=None,
        help="Optional reasoning effort for models that support it.",
    )
    phase_one.add_argument(
        "--retry-failed-llm",
        action="store_true",
        help="Retry an identical request recorded in failed-LLM history.",
    )
    phase_one.add_argument(
        "--refresh-llm",
        action="store_true",
        help="Make fresh judge requests instead of reading successful or failed caches.",
    )


def run_evaluation_command(args: argparse.Namespace) -> int:
    try:
        if args.evaluation_command != "phase-one":
            raise ValueError(f"unsupported evaluation command: {args.evaluation_command}")
        evaluation_input = PhaseOneEvaluationInput(
            run_directories=tuple(args.run),
            source_roots=_parse_source_roots(args.source_root),
            deployment_context=args.context,
        )
        output = args.output.expanduser().resolve()
        _reject_output_inside_runs(output, evaluation_input.run_directories)
        judge_config = None if args.no_llm else _judge_config(args, output)
        report = evaluate_phase_one(evaluation_input, judge_config=judge_config)
        write_json(output, report)
        _print_report(report, output)
        return 0 if args.no_llm or all(run.judge.complete for run in report.runs) else 1
    except (OSError, ValueError, ValidationError, yaml.YAMLError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 1


def _parse_source_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        source_id, separator, raw_path = value.partition("=")
        source_id = source_id.strip()
        raw_path = raw_path.strip()
        if not separator or not source_id or not raw_path:
            raise ValueError("--source-root must use ID=PATH")
        if source_id in roots:
            raise ValueError(f"duplicate --source-root ID: {source_id}")
        roots[source_id] = Path(raw_path)
    return roots


def _judge_config(args: argparse.Namespace, output: Path) -> PhaseOneJudgeConfig:
    model = (
        args.model
        or os.getenv("PHERAGENT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or DEFAULT_ANALYSIS_MODEL
    )
    return PhaseOneJudgeConfig(
        model=model,
        api_key_env=args.openai_api_key_env,
        base_url_env=args.openai_base_url_env,
        base_url=args.openai_base_url,
        timeout=args.llm_timeout,
        max_output_tokens=args.llm_max_tokens,
        max_requests_per_run=args.llm_max_requests,
        max_evidence_characters=args.llm_max_evidence_chars,
        cache_dir=output.parent / ".llm-cache",
        retry_failed=args.retry_failed_llm,
        refresh_cache=args.refresh_llm,
        reasoning_effort=args.llm_reasoning_effort,
    )


def _print_report(report: PhaseOneEvaluationReport, output: Path) -> None:
    print(f"evaluation report: {output}")
    for run in report.runs:
        print(
            f"{run.run_id}: validity={_score(run.validity.score)}; "
            f"relevance={_score(run.relevance.score)}; "
            f"completeness={_score(run.completeness.score)}; "
            f"deployability={_score(run.deployability.score)}"
        )
        for metric in (run.validity, run.relevance, run.completeness, run.deployability):
            _print_metric_issues(metric)
        if run.judge.enabled:
            stages = "; ".join(
                f"{stage}={status}" for stage, status in sorted(run.judge.stages.items())
            )
            print(f"  LLM judge: {stages or 'not run'}; model={run.judge.model}")
            for warning in run.judge.warnings:
                print(f"  LLM warning: {warning}", file=sys.stderr)
    print(f"consistency: {_score(report.consistency.score)}")
    _print_metric_issues(report.consistency)


def _print_metric_issues(metric: MetricResult) -> None:
    if metric.issues:
        print(f"  {metric.dimension.value} issues ({len(metric.issues)}):")
        grouped: dict[str, list[str]] = {}
        for issue in metric.issues:
            grouped.setdefault(issue.reason, []).append(issue.subject)
        for reason, subjects in grouped.items():
            print(f"    - {reason} ({len(subjects)}): {', '.join(subjects)}")
    elif metric.explanation:
        print(f"  {metric.dimension.value} note: {metric.explanation}")


def _reject_output_inside_runs(output: Path, run_directories: tuple[Path, ...]) -> None:
    for run_directory in run_directories:
        if output.is_relative_to(run_directory.expanduser().resolve()):
            raise ValueError("evaluation output must be outside sealed run directories")


def _score(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.3f}"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed
