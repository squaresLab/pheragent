from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pheragent.deploy_agent import (
    Decision,
    DeploymentTask,
    _policy,
    _target_command,
    run_deployment_agent,
)


def _decision(kind: str, tool: str | None = None, **changes: object) -> Decision:
    payload = {
        "kind": kind,
        "tool": tool,
        "reason": "follow the local installer",
        "focus": "install application",
        "query": None,
        "source_path": None,
        "start_line": None,
        "end_line": None,
        "command": [],
        "working_directory": None,
        "evidence": [],
        "expected_change": None,
        "validation": [],
        "add_gaps": [],
        "resolve_gaps": [],
        "add_questions": [],
        "resolve_questions": [],
    }
    payload.update(changes)
    return Decision.model_validate(payload)


def _task(path: Path, source: Path, output: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "task": {"objective": "Install the sample application"},
                "sources": {"repositories": [str(source)]},
                "environment": {"type": "shell", "sandbox": True},
                "success_checks": [
                    {"command": ["ls", str(output / "workspace/repository-1/ready")]}
                ],
                "budgets": {
                    "max_cycles": 3,
                    "max_mutating_actions": 2,
                    "max_runtime_minutes": 2,
                    "max_read_actions_per_cycle": 3,
                },
            }
        )
    )


def test_agent_previews_then_validates_a_source_installer(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "install.sh").write_text("#!/bin/sh\ntouch ready\n")
    (source / "install.sh").chmod(0o755)
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, output)
    decisions = iter(
        [
            _decision("ACT", "search_sources", query="install sample"),
            _decision(
                "ACT",
                "execute",
                command=["./install.sh"],
                working_directory="repository-1:.",
                evidence=["repository-1:install.sh"],
                expected_change="sample installer created ready file",
                validation=[{"command": ["ls", str(output / "workspace/repository-1/ready")]}],
            ),
            _decision("DONE"),
        ]
    )

    def decide(_state, _observation, _last_result, _sources, _cycle):
        return next(decisions), {"requests": 1}

    report = run_deployment_agent(task, output, execute=True, decide=decide)
    assert report["status"] == "SUCCESS"
    assert report["mutating_actions"] == 1
    assert (output / "workspace/repository-1/ready").is_file()
    assert json.loads((output / "final-report.json").read_text())["status"] == "SUCCESS"
    assert (output / "actions.jsonl").is_file()
    assert (output / "trajectory.jsonl").is_file()


def test_preview_does_not_run_a_source_installer(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "install.sh").write_text("#!/bin/sh\ntouch ready\n")
    (source / "install.sh").chmod(0o755)
    task = tmp_path / "task.yaml"
    output = tmp_path / "preview"
    _task(task, source, output)

    def decide(_state, _observation, _last_result, _sources, _cycle):
        return _decision(
            "ACT",
            "execute",
            command=["./install.sh"],
            working_directory="repository-1",
            evidence=["repository-1:install.sh:1-2 — cited installer"],
            expected_change="ready file exists",
            validation=[{"command": ["ls", str(output / "workspace/repository-1/ready")]}],
        ), {}

    report = run_deployment_agent(task, output, decide=decide)
    assert report["status"] == "BLOCKED"
    assert not (output / "workspace").exists()


def test_failed_action_reaches_next_decision(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "install.sh").write_text("#!/bin/sh\nexit 1\n")
    (source / "install.sh").chmod(0o755)
    task = tmp_path / "task.yaml"
    output = tmp_path / "failed"
    _task(task, source, output)
    calls = []

    def decide(_state, _observation, last_result, _sources, _cycle):
        calls.append(last_result)
        if len(calls) == 1:
            return _decision(
                "ACT",
                "execute",
                command=["./install.sh"],
                working_directory="repository-1:.",
                evidence=["repository-1:install.sh"],
                expected_change="ready file exists",
                validation=[{"command": ["ls", str(output / "workspace/repository-1/ready")]}],
            ), {}
        return _decision("BLOCKED", reason="installer failed"), {}

    report = run_deployment_agent(task, output, execute=True, decide=decide)
    assert report["status"] == "BLOCKED"
    assert calls[1]["status"] == "failed_validation"
    assert calls[1]["execution"]["exit_code"] == 1


def test_policy_denies_direct_and_disguised_destructive_commands() -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "shell", "sandbox": True},
        }
    )
    assert _policy(["/bin/rm", "-rf", "/tmp/sample"], task, mutating=True).startswith("denied")
    assert _policy(["bash", "-c", "rm -rf /tmp/sample"], task, mutating=True).startswith("denied")


def test_kubernetes_actions_use_the_declared_context() -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "kubernetes", "context": "target-cluster"},
            "constraints": {"allowed_namespaces": ["*"]},
        }
    )
    assert _target_command(["kubectl", "get", "pods"], task) == [
        "kubectl",
        "--context",
        "target-cluster",
        "get",
        "pods",
    ]
    with pytest.raises(ValueError, match="omit --context"):
        _target_command(["kubectl", "--context", "other", "get", "pods"], task)


def test_missing_key_stops_before_source_acquisition(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, output)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        run_deployment_agent(task, output)
    assert not output.exists()
