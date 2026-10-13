"""Progressive deployment command."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from .agent import run_deployment_agent
from .oneshot import run_oneshot_agent
from .telemetry import flush, set_output, span


def add_deployment_parser(subparsers) -> None:
    deployment = subparsers.add_parser("deployment", help="Deploy from a task and live state.")
    commands = deployment.add_subparsers(dest="deployment_command", required=True)
    run = commands.add_parser("run", help="Observe, act, and validate progressively.")
    run.add_argument("task", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--model", default="gpt-5.6-terra")
    run.add_argument("--execute", action="store_true", help="Permit approved changes.")
    run.add_argument("--resume", action="store_true", help="Continue an existing output run.")
    oneshot = commands.add_parser(
        "oneshot", help="Research: deploy as much of the source-defined system as possible."
    )
    oneshot.add_argument("task", type=Path)
    oneshot.add_argument("--output", type=Path)
    oneshot.add_argument("--model", default="gpt-6-astra")
    oneshot.add_argument("--reasoning-effort", default="max")
    oneshot.add_argument("--execute", action="store_true", help="Run policy-permitted commands.")
    oneshot.add_argument("--resume", action="store_true", help="Continue an existing run.")


def run_deployment_command(args: argparse.Namespace) -> int:
    output = args.output or (
        Path(".pheragent/agent-runs") / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    try:
        with span(
            f"pheragent.deployment.{args.deployment_command}",
            input={"task": str(args.task), "output": str(output)},
            task=str(args.task),
        ) as active_span:
            if args.deployment_command == "oneshot":
                report = run_oneshot_agent(
                    args.task,
                    output,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    execute=args.execute,
                    resume=args.resume,
                )
            else:
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
    print(f"Actions attempted: {report['mutating_actions']}")
    if "components" in report:
        verified = [item["name"] for item in report["components"] if item["status"] == "verified"]
        deferred = [item["name"] for item in report["components"] if item["status"] == "deferred"]
        print(f"Verified components: {', '.join(verified) or 'none'}")
        print(f"Deferred components: {', '.join(deferred) or 'none'}")
    else:
        print(f"Validated changes: {', '.join(report['state']['milestones']) or 'none'}")
        print(
            "New verified outcomes: "
            f"{', '.join(item['id'] for item in report['state']['verified_outcomes']) or 'none'}"
        )
    flush()
    return 0 if report["status"] in {"SUCCESS", "PARTIAL_SUCCESS"} else 1
