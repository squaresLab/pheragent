from __future__ import annotations

import json
from pathlib import Path

import yaml

from pheragent.deployment.oneshot import OneShotDecision, run_oneshot_agent
from pheragent.deployment.runtime import classify_command
from pheragent.deployment.task import DeploymentTask


def _task(path: Path, source: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "task": {"objective": "Deploy as much of the sample system as possible"},
                "sources": {"repositories": [str(source)]},
                "environment": {"type": "shell", "sandbox": True},
                "budgets": {
                    "max_cycles": 8,
                    "max_mutating_actions": 6,
                    "max_runtime_minutes": 2,
                    "max_read_actions_per_cycle": 3,
                },
            }
        )
    )


def _decision(action: str, reason: str, **values) -> OneShotDecision:
    return OneShotDecision.model_validate({"action": action, "reason": reason, **values})


def test_oneshot_continues_after_denial_and_records_partial_success(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "install.sh").write_text("#!/bin/sh\ntouch ready\necho installed\n")
    (source / "install.sh").chmod(0o755)
    task_path = tmp_path / "task.yaml"
    _task(task_path, source)
    output = tmp_path / "run"
    monkeypatch.setattr(
        "pheragent.deployment.oneshot.observe_target",
        lambda _task: {"host": {"exit_code": 0, "stdout": "ready", "stderr": ""}},
    )
    decisions = iter(
        [
            _decision("command", "unsafe route", command=["rm", "ready"]),
            _decision(
                "command",
                "run documented installer",
                command=["./install.sh"],
                working_directory="repository-1:.",
                component="sample",
            ),
            _decision(
                "command",
                "verify installed file",
                command=["ls", "ready"],
                working_directory="repository-1:.",
                component="sample",
            ),
            _decision("record", "sample is healthy", component="sample", status="verified"),
            _decision(
                "record",
                "vendor endpoint is optional",
                component="vendor-api",
                status="deferred",
                placeholder="https://vendor.pending.invalid",
                human_action="replace the placeholder with the vendor endpoint",
            ),
            _decision("finish", "no independent work remains"),
            _decision("finish", "final audit found no independent work"),
        ]
    )

    report = run_oneshot_agent(
        task_path,
        output,
        execute=True,
        decide=lambda _payload, _iteration: (next(decisions), {"requests": 1}),
    )

    assert report["status"] == "PARTIAL_SUCCESS"
    assert (output / "workspace/repository-1/ready").is_file()
    statuses = {item["name"]: item["status"] for item in report["components"]}
    assert statuses == {"sample": "verified", "vendor-api": "deferred"}
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert any(
        event.get("result", {}).get("status") == "policy_denied" for event in events
    )


def test_host_package_changes_need_high_risk_approval() -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "Deploy the system"},
            "sources": {"repositories": ["."]},
            "environment": {"type": "shell"},
            "constraints": {"allow_destructive_actions": True},
        }
    )

    assert classify_command(["apt-get", "install", "curl"], task, mutating=True).startswith(
        "high_risk_approval"
    )
    assert classify_command(["./uninstall.sh"], task, mutating=True).startswith(
        "high_risk_approval"
    )


def test_oneshot_resumes_without_overwriting_history(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "install.sh").write_text("#!/bin/sh\ntouch ready\necho installed\n")
    (source / "install.sh").chmod(0o755)
    task_path = tmp_path / "task.yaml"
    _task(task_path, source)
    task = yaml.safe_load(task_path.read_text())
    task["budgets"]["max_cycles"] = 1
    task_path.write_text(yaml.safe_dump(task))
    output = tmp_path / "run"
    monkeypatch.setattr(
        "pheragent.deployment.oneshot.observe_target",
        lambda _task: {"host": {"exit_code": 0, "stdout": "ready", "stderr": ""}},
    )

    first = run_oneshot_agent(
        task_path,
        output,
        execute=True,
        decide=lambda _payload, _iteration: (
            _decision(
                "command",
                "install sample",
                command=["./install.sh"],
                working_directory="repository-1:.",
                component="sample",
            ),
            {},
        ),
    )
    assert first["status"] == "BUDGET_EXHAUSTED"

    task["budgets"]["max_cycles"] = 5
    task_path.write_text(yaml.safe_dump(task))
    resumed_decisions = iter(
        [
            _decision(
                "command",
                "verify sample",
                command=["ls", "ready"],
                working_directory="repository-1:.",
                component="sample",
            ),
            _decision("record", "sample is healthy", component="sample", status="verified"),
            _decision("finish", "no work remains"),
            _decision("finish", "audit complete"),
        ]
    )
    resumed = run_oneshot_agent(
        task_path,
        output,
        execute=True,
        resume=True,
        decide=lambda _payload, _iteration: (next(resumed_decisions), {}),
    )

    assert resumed["status"] == "SUCCESS"
    assert resumed["cycles"] == 5
    assert (output / "actions/0001.log").is_file()
    assert (output / "actions/0002.log").is_file()
    decisions = [
        event
        for event in (
            json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
        )
        if event["event"] == "decision"
    ]
    assert [event["iteration"] for event in decisions] == [1, 2, 3, 4, 5]
