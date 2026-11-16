from __future__ import annotations

import json
from pathlib import Path

import yaml

from pheragent.deployment.agent import run_deployment_agent
from pheragent.deployment.runtime import classify_command
from pheragent.deployment.task import Decision, DeploymentTask

MEMORY = """# Overview
- [ ] sample application
# Current
Install and verify the sample application.
"""


def _task(path: Path, source: Path, *, max_cycles: int = 6) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "task": {"objective": "Deploy the sample application"},
                "sources": {"repositories": [str(source)]},
                "environment": {"type": "shell", "sandbox": True},
                "budgets": {"max_cycles": max_cycles, "max_runtime_minutes": 2},
            }
        )
    )


def _decision(action: str, reason: str, **values) -> Decision:
    return Decision.model_validate(
        {"action": action, "reason": reason, "memory": MEMORY, **values}
    )


def test_agent_deploys_verifies_and_records_a_component(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "install.sh").write_text("#!/bin/sh\ntouch ready\n")
    (source / "install.sh").chmod(0o755)
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source)
    evidence: list[int] = []

    def decide(payload, iteration):
        if iteration == 1:
            return _decision(
                "shell",
                "run the documented installer",
                command=["./install.sh"],
                working_directory="repository-1:.",
            ), {"requests": 1}
        if iteration == 2:
            evidence.append(payload["latest_feedback"]["event_sequence"])
            return _decision(
                "shell",
                "verify the installed artifact",
                command=["test", "-f", "ready"],
                working_directory="repository-1:.",
            ), {"requests": 1}
        if iteration == 3:
            evidence.append(payload["latest_feedback"]["event_sequence"])
            return _decision(
                "record",
                "the sample application is deployed",
                component="sample application",
                deployed=True,
                evidence_events=evidence,
            ), {"requests": 1}
        return _decision("finish", "all overview outcomes are deployed"), {"requests": 1}

    report = run_deployment_agent(task, output, execute=True, decide=decide)

    assert report["status"] == "SUCCESS"
    assert report["components"][0]["deployed"] is True
    assert (output / "workspace/repository-1/ready").is_file()
    assert "./install.sh" in (output / "runbook.sh").read_text()
    assert "test -f ready" not in (output / "runbook.sh").read_text()
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events].count("memory_updated") == 1


def test_agent_resumes_from_memory_and_latest_feedback(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "install.sh").write_text("#!/bin/sh\ntouch ready\n")
    (source / "install.sh").chmod(0o755)
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, max_cycles=1)

    first = run_deployment_agent(
        task,
        output,
        execute=True,
        decide=lambda _payload, _iteration: (
            _decision(
                "shell",
                "install sample",
                command=["./install.sh"],
                working_directory="repository-1:.",
            ),
            {},
        ),
    )
    assert first["status"] == "BUDGET_EXHAUSTED"

    _task(task, source, max_cycles=3)

    def resume(payload, iteration):
        assert payload["memory"] == MEMORY
        if iteration == 2:
            assert payload["latest_feedback"]["event_sequence"] > 0
            return _decision(
                "record",
                "sample is deployed",
                component="sample",
                deployed=True,
                evidence_events=[payload["latest_feedback"]["event_sequence"]],
            ), {}
        return _decision("finish", "deployment complete"), {}

    report = run_deployment_agent(task, output, execute=True, resume=True, decide=resume)

    assert report["status"] == "SUCCESS"
    assert report["cycles"] == 3
    assert len((output / "events.jsonl").read_text().splitlines()) > 1


def test_repeat_detection_uses_command_result_not_reason(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, max_cycles=4)
    seen_repeat_limit = False

    def decide(payload, iteration):
        nonlocal seen_repeat_limit
        if iteration == 4:
            seen_repeat_limit = "repeat_limit" in payload["latest_feedback"]
            return _decision("finish", "no deployment outcome was found"), {}
        return _decision(
            "shell",
            f"wording {iteration}",
            command=["test", "-f", "missing"],
            working_directory="repository-1:.",
        ), {}

    run_deployment_agent(task, output, execute=True, decide=decide)
    assert seen_repeat_limit


def test_agent_can_enter_a_directory_created_in_its_workspace(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, max_cycles=3)
    decisions = iter(
        [
            _decision(
                "shell",
                "create generated workspace",
                command=["mkdir", "generated"],
                working_directory="repository-1:.",
            ),
            _decision(
                "shell",
                "inspect generated workspace",
                command=["pwd"],
                working_directory="repository-1:generated",
            ),
            _decision("finish", "workspace behavior verified"),
        ]
    )

    report = run_deployment_agent(
        task,
        output,
        execute=True,
        decide=lambda _payload, _iteration: (next(decisions), {}),
    )

    assert report["latest_feedback"]["status"] == "succeeded"
    assert report["latest_feedback"]["execution"]["stdout"].rstrip().endswith("generated")


def test_only_boundary_crossing_commands_need_approval() -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "Deploy a system"},
            "sources": {"repositories": ["."]},
            "environment": {"type": "shell"},
        }
    )

    assert classify_command(["make", "start"], task) == "allowed"
    assert classify_command(["python3", "-c", "print('ok')"], task) == "allowed"
    assert classify_command(["aws", "sts", "get-caller-identity"], task) == "allowed"
    assert classify_command(["rm", "-rf", "data"], task).startswith("high_risk_approval")
    assert classify_command(["kubectl", "create", "ingress", "web"], task).startswith(
        "high_risk_approval"
    )
    assert classify_command(["sudo", "apt-get", "install", "curl"], task).startswith(
        "high_risk_approval"
    )

    kubernetes = DeploymentTask.model_validate(
        {
            "task": {"objective": "Deploy a system"},
            "sources": {"repositories": ["."]},
            "environment": {"type": "kubernetes", "context": "test"},
            "constraints": {"allowed_namespaces": ["application"]},
        }
    )
    assert classify_command(["./install.sh"], kubernetes).startswith("high_risk_approval")
