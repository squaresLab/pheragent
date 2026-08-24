from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from pheragent.deployment.enums import AnalysisTreatment

from .evaluation import summarize_study
from .runner import describe_study, load_study, run_study, validate_study_inputs


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
        choices=tuple(item.value for item in AnalysisTreatment),
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


def run_research_command(args: argparse.Namespace) -> int:
    try:
        if args.research_command == "summarize":
            summarize_study(args.results.expanduser().resolve())
            return 0
        if args.research_command != "run":
            raise ValueError(f"unsupported research command: {args.research_command}")
        selected_treatments = {AnalysisTreatment(value) for value in args.treatment} or None
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
