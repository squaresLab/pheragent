"""Progressive deployment command."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from .agent import run_deployment_agent


def add_deployment_parser(subparsers) -> None:
    deployment = subparsers.add_parser("deployment", help="Deploy from a task and live state.")
    commands = deployment.add_subparsers(dest="deployment_command", required=True)
    run = commands.add_parser("run", help="Observe, act, and validate progressively.")
    run.add_argument("task", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--model", default="gpt-5.6-terra")
    run.add_argument("--execute", action="store_true", help="Permit approved changes.")


def run_deployment_command(args: argparse.Namespace) -> int:
    output = args.output or (
        Path(".pheragent/agent-runs") / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    try:
        report = run_deployment_agent(args.task, output, model=args.model, execute=args.execute)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"deployment error: {exc}", file=sys.stderr)
        return 2
    print(f"run: {output.resolve()}")
    print(f"result: {report['status']} — {report['reason']}")
    usage = report["usage"]
    print(
        "LLM: "
        f"{usage.get('requests', 0)} requests; "
        f"{usage.get('total_tokens', 0)} tokens "
        f"(input {usage.get('input_tokens', 0)}, output {usage.get('output_tokens', 0)})"
    )
    print(f"Actions attempted: {report['mutating_actions']}")
    print(f"Validated changes: {', '.join(report['state']['milestones']) or 'none'}")
    print(
        "New verified outcomes: "
        f"{', '.join(item['id'] for item in report['state']['verified_outcomes']) or 'none'}"
    )
    return 0 if report["status"] == "SUCCESS" else 1
