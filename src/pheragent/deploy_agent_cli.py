"""Command line entry point for the deployment agent prototype."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from .deploy_agent import run_deployment_agent
from .env import load_dotenv


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path(".env"))
    parser = argparse.ArgumentParser(prog="heragentdeploy")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Investigate and deploy one bounded task.")
    run.add_argument("task", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--model", default="gpt-5.6-terra")
    run.add_argument("--execute", action="store_true", help="Permit policy-approved mutations.")
    args = parser.parse_args(argv)
    output = args.output or (
        Path(".pheragent/agent-runs") / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    try:
        report = run_deployment_agent(args.task, output, model=args.model, execute=args.execute)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"deployment agent error: {exc}\n")
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


if __name__ == "__main__":
    raise SystemExit(main())
