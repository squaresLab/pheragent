"""Progressive deployment command."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from .agent import run_deployment_agent
from .telemetry import flush, set_output, span


def add_deployment_parser(subparsers) -> None:
    deploy = subparsers.add_parser(
        "deploy", help="Deploy progressively from source and live state."
    )
    deploy.add_argument("task", type=Path)
    deploy.add_argument("--output", type=Path)
    deploy.add_argument("--model", default="gpt-5.6-terra")
    deploy.add_argument("--execute", action="store_true", help="Run model-selected commands.")
    deploy.add_argument("--resume", action="store_true", help="Continue an existing output run.")


def run_deployment_command(args: argparse.Namespace) -> int:
    output = args.output or (
        Path(".pheragent/agent-runs") / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    try:
        with span(
            "pheragent.deploy",
            input={"task": str(args.task), "output": str(output)},
            task=str(args.task),
        ) as active_span:
            report = run_deployment_agent(
                args.task,
                output,
                model=args.model,
                execute=args.execute,
                resume=args.resume,
            )
            set_output(
                active_span,
                {"status": report["status"], "reason": report["reason"]},
                report["usage"],
            )
    except (OSError, RuntimeError, ValueError) as exc:
        flush()
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
    deployed = [item["name"] for item in report["components"] if item["deployed"]]
    not_deployed = [item["name"] for item in report["components"] if not item["deployed"]]
    print(f"Deployed components: {', '.join(deployed) or 'none'}")
    print(f"Not deployed: {', '.join(not_deployed) or 'none'}")
    flush()
    return 0 if report["status"] in {"SUCCESS", "PARTIAL_SUCCESS"} else 1
