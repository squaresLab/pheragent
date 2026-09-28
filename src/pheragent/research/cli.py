from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from pheragent.deployment.one_shot import run_one_shot
from pheragent.deployment.recursive_plan import run_recursive_planning

from .runner import (
    ResearchTreatment,
    describe_study,
    load_study,
    run_study,
    summarize_study,
    validate_study_inputs,
)


def add_research_parser(subparsers: Any) -> None:
    research = subparsers.add_parser(
        "research",
        help="Run reproducible experiments over the deployment analyzer.",
    )
    commands = research.add_subparsers(dest="research_command", required=True)
    run = commands.add_parser("run", help="Preflight or execute a configured study.")
    run.add_argument("--study", required=True, type=Path)
    run.add_argument("--output", type=Path, default=Path(".pheragent/research"))
    run.add_argument("--case", action="append", default=[])
    run.add_argument(
        "--treatment",
        action="append",
        choices=tuple(item.value for item in ResearchTreatment),
        default=[],
    )
    run.add_argument("--repetitions", type=_positive_int, default=None)
    run.add_argument("--refresh-llm", action="store_true")
    run.add_argument("--debug", action="store_true")
    run.add_argument(
        "--execute",
        action="store_true",
        help="Run the study; without this flag only the run and LLM request ceilings are shown.",
    )
    summarize = commands.add_parser("summarize", help="Rebuild summaries from sealed runs.")
    summarize.add_argument("results", type=Path)
    one_shot = commands.add_parser(
        "one-shot",
        help="Send all inventoried deployment files to one LLM request.",
    )
    one_shot.add_argument("--sources", required=True, type=Path)
    one_shot.add_argument("--context", required=True, type=Path)
    one_shot.add_argument("--output", type=Path, default=Path(".pheragent/research/one-shot"))
    one_shot.add_argument("--run-name")
    one_shot.add_argument("--model", default="gpt-5.6-terra")
    one_shot.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    one_shot.add_argument("--timeout", type=float, default=600.0)
    recursive = commands.add_parser(
        "recursive-plan",
        help="Build a bounded deployment tree by expanding one sourced step at a time.",
    )
    recursive.add_argument("--sources", required=True, type=Path)
    recursive.add_argument("--context", required=True, type=Path)
    recursive.add_argument(
        "--output",
        type=Path,
        default=Path(".pheragent/research/recursive-plan"),
    )
    recursive.add_argument("--run-name")
    recursive.add_argument("--model", default="gpt-5.6-terra")
    recursive.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    recursive.add_argument("--max-depth", type=_positive_int, default=5)
    recursive.add_argument("--max-nodes", type=_positive_int, default=60)
    recursive.add_argument("--max-requests", type=_positive_int, default=20)
    recursive.add_argument("--action-budget", type=_positive_int, default=None)
    recursive.add_argument("--evidence-characters", type=_positive_int, default=24_000)
    recursive.add_argument("--max-output-tokens", type=_positive_int, default=5_000)
    recursive.add_argument("--timeout", type=float, default=300.0)
    recursive.add_argument(
        "--runtime-context",
        type=Path,
        help="Optional read-only runtime snapshot produced by deployment inspect-runtime.",
    )
    recursive.add_argument(
        "--resume-tree",
        type=Path,
        help="Resume only blocked source requests after adding approved sources to sources.yaml.",
    )


def run_research_command(args: argparse.Namespace) -> int:
    try:
        if args.research_command == "summarize":
            summarize_study(args.results.expanduser().resolve())
            return 0
        if args.research_command == "one-shot":
            result = run_one_shot(
                args.sources,
                args.context,
                args.output,
                run_name=args.run_name,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                timeout=args.timeout,
                progress=lambda message: print(f"research: {message}", file=sys.stderr, flush=True),
            )
            usage = result.usage
            print(f"run: {result.run_dir}")
            print(f"corpus: {result.run_dir / 'corpus.txt'}")
            print(f"deployment outline: {result.run_dir / 'deployment-outline.yaml'}")
            print(
                "LLM usage: "
                f"input={usage.get('input_tokens', 0)}; "
                f"output={usage.get('output_tokens', 0)}; "
                f"reasoning={usage.get('reasoning_tokens', 0)}; "
                f"total={usage.get('total_tokens', 0)}"
            )
            return 0
        if args.research_command == "recursive-plan":
            result = run_recursive_planning(
                args.sources,
                args.context,
                args.output,
                run_name=args.run_name,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                max_depth=args.max_depth,
                max_nodes=args.max_nodes,
                max_requests=args.max_requests,
                max_actions=args.action_budget,
                evidence_characters=args.evidence_characters,
                max_output_tokens=args.max_output_tokens,
                timeout=args.timeout,
                runtime_context_path=args.runtime_context,
                resume_tree=args.resume_tree,
                progress=lambda message: print(f"research: {message}", file=sys.stderr, flush=True),
            )
            usage = result.usage
            print(f"run: {result.run_dir}")
            print(f"deployment tree: {result.run_dir / 'deployment-tree.yaml'}")
            print(f"analysis trace: {result.run_dir / 'analysis-trace.md'}")
            print(f"planning complete: {str(result.plan.planning_complete).lower()}")
            print(f"deployment ready: {str(result.plan.deployment_ready).lower()}")
            print(
                "LLM usage: "
                f"requests={usage.get('requests', 0)}; "
                f"input={usage.get('input_tokens', 0)}; "
                f"output={usage.get('output_tokens', 0)}; "
                f"total={usage.get('total_tokens', 0)}"
            )
            return 0
        if args.research_command != "run":
            raise ValueError(f"unsupported research command: {args.research_command}")
        selected_treatments = {ResearchTreatment(value) for value in args.treatment} or None
        selected_cases = set(args.case) or None
        study = load_study(args.study)
        validate_study_inputs(args.study, study, selected_cases=selected_cases)
        preflight = describe_study(
            study,
            selected_cases=selected_cases,
            selected_treatments=selected_treatments,
            repetitions=args.repetitions,
        )
        print(f"study: {study.id}")
        print(f"planned runs: {preflight['runs']}")
        print(f"maximum LLM requests: {preflight['maximum_llm_requests']}")
        if not args.execute:
            print("preflight only; add --execute to run")
            return 0
        failures, study_root = run_study(
            args.study,
            args.output,
            selected_cases=selected_cases,
            selected_treatments=selected_treatments,
            repetitions=args.repetitions,
            refresh_llm=args.refresh_llm,
            debug=args.debug,
            progress=lambda message: print(f"research: {message}", file=sys.stderr, flush=True),
        )
        print(f"results: {study_root}")
        return 1 if failures else 0
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        print(f"research error: {exc}", file=sys.stderr)
        return 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed
