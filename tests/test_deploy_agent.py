from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import yaml

from pheragent.deploy_agent import (
    _INSTRUCTIONS,
    Decision,
    DeploymentTask,
    _brief,
    _checks,
    _execute,
    _policy,
    _target_command,
    run_deployment_agent,
)
from pheragent.deploy_agent_cli import main as deploy_main


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
    assert calls[1]["status"] == "command_failed"
    assert calls[1]["execution"]["exit_code"] == 1


def test_route_choice_and_repeated_read_reach_next_decision(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Use ./install.sh for this target.\n")
    task = tmp_path / "task.yaml"
    _task(task, source, tmp_path / "run")
    seen = []
    decisions = iter(
        [
            _decision("ACT", "read_file", source_path="repository-1:README.md"),
            _decision(
                "ACT",
                "read_file",
                source_path="repository-1:README.md",
                selected_route="Use ./install.sh for this target",
                evidence=["repository-1:README.md"],
            ),
            _decision("BLOCKED", reason="prerequisite is missing"),
        ]
    )

    def decide(state, _observation, last_result, _sources, _cycle):
        seen.append((state.copy(), last_result.copy()))
        return next(decisions), {}

    report = run_deployment_agent(task, tmp_path / "run", decide=decide)
    assert report["status"] == "BLOCKED"
    assert seen[2][0]["selected_route"] == {
        "choice": "Use ./install.sh for this target",
        "evidence": ["repository-1:README.md"],
    }
    assert seen[2][1]["text"] == seen[1][1]["text"]
    assert seen[2][1]["repeated_result_count"] == 1


def test_agent_prompt_requires_route_choice_and_deployment_progress() -> None:
    assert "senior DevOps engineer" in _INSTRUCTIONS
    assert "step by step" in _INSTRUCTIONS
    assert "selected_route" in _INSTRUCTIONS


def test_agent_prompt_expands_missing_prerequisites_before_blocking() -> None:
    assert "missing prerequisite" in _INSTRUCTIONS
    assert "configured sources" in _INSTRUCTIONS
    assert "resume the original objective" in " ".join(_INSTRUCTIONS.split())
    assert "validation" in _INSTRUCTIONS


def test_agent_validates_prerequisite_then_resumes_original_goal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name, output_name in (("storage.sh", "storage-ready"), ("install.sh", "ready")):
        script = source / name
        script.write_text(f"#!/bin/sh\ntouch {output_name}\n")
        script.chmod(0o755)
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, output)
    decisions = iter(
        [
            _decision(
                "ACT",
                "execute",
                command=["./storage.sh"],
                working_directory="repository-1:.",
                evidence=["repository-1:storage.sh"],
                expected_change="storage ready",
                add_gaps=["storage missing"],
                validation=[
                    {"command": ["ls", str(output / "workspace/repository-1/storage-ready")]}
                ],
            ),
            _decision(
                "ACT",
                "execute",
                command=["./install.sh"],
                working_directory="repository-1:.",
                evidence=["repository-1:install.sh"],
                expected_change="application ready",
                resolve_gaps=["storage missing"],
                validation=[{"command": ["ls", str(output / "workspace/repository-1/ready")]}],
            ),
            _decision("DONE"),
        ]
    )
    states = []

    def decide(state, _observation, _last_result, _sources, _cycle):
        states.append((state["gaps"].copy(), state["milestones"].copy()))
        return next(decisions), {}

    report = run_deployment_agent(task, output, execute=True, decide=decide)
    assert report["status"] == "SUCCESS"
    assert report["mutating_actions"] == 2
    assert states[1] == (["storage missing"], ["storage ready"])
    assert report["state"]["gaps"] == []
    assert report["state"]["milestones"] == ["storage ready", "application ready"]


def test_brief_retains_a_long_result_start_and_end() -> None:
    result = "Default route is in this header.\n" + "x" * 7000 + "\nLast error is here."
    brief = _brief(result)
    assert "Default route" in brief
    assert "Last error" in brief
    assert len(brief) < len(result)


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
    with pytest.raises(ValueError, match="must match the declared target context"):
        _target_command(["kubectl", "--context", "other", "get", "pods"], task)


def test_matching_context_storage_check_is_read_only() -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "kubernetes", "context": "test-cluster"},
        }
    )
    command = ["kubectl", "--context", "test-cluster", "get", "storageclass"]
    assert _policy(command, task, mutating=False) == "allowed"
    assert _target_command(command, task) == command


def test_read_only_execute_is_observed_without_approval(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Sample deployment source\n")
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "task": {"objective": "Inspect storage"},
                "sources": {"repositories": [str(source)]},
                "environment": {"type": "kubernetes", "context": "target-cluster"},
            }
        )
    )
    monkeypatch.setattr("pheragent.deploy_agent._observe", lambda _task: {})
    observed = []

    def inspect(command, _task, **_kwargs):
        observed.append(command)
        return {"exit_code": 0, "stdout": "No resources found", "stderr": ""}

    monkeypatch.setattr("pheragent.deploy_agent._command", inspect)
    decisions = iter(
        [
            _decision(
                "ACT",
                "execute",
                command=["kubectl", "--context", "target-cluster", "get", "storageclass"],
            ),
            _decision("BLOCKED", reason="inspection complete"),
        ]
    )

    def decide(_state, _observation, _last_result, _sources, _cycle):
        return next(decisions), {}

    output = tmp_path / "preview"
    report = run_deployment_agent(task_path, output, decide=decide)
    assert report["status"] == "BLOCKED"
    assert observed == [["kubectl", "--context", "target-cluster", "get", "storageclass"]]
    assert report["mutating_actions"] == 0
    assert not (output / "actions.jsonl").exists()


def test_unfamiliar_mutation_requires_review_and_wrong_context_is_denied() -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "kubernetes", "context": "target-cluster"},
            "constraints": {"allowed_namespaces": ["postgres"]},
        }
    )
    assert _policy(
        ["kubectl", "label", "pod", "db", "tested=yes", "-n", "postgres"], task, mutating=True
    ).startswith("approval_required")
    assert _policy(
        ["kubectl", "--context", "other-cluster", "get", "pods"], task, mutating=False
    ).startswith("denied")


@pytest.mark.parametrize(("answer", "should_run"), [("n\n", False), ("y\n", True)])
def test_kubernetes_change_waits_for_terminal_approval(
    tmp_path: Path, monkeypatch, capsys, answer: str, should_run: bool
) -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "kubernetes", "context": "target-cluster"},
            "constraints": {"allowed_namespaces": ["postgres"]},
        }
    )
    decision = _decision(
        "ACT",
        "execute",
        command=["kubectl", "label", "pod", "db", "tested=yes", "-n", "postgres"],
        evidence=["repository-1:README.md"],
        expected_change="Pod has the label",
        validation=[{"command": ["kubectl", "get", "pod", "db", "-n", "postgres"]}],
    )

    class Evidence:
        def existing_refs(self, _references):
            return {"repository-1:README.md"}

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    commands = []

    def run_command(command, _task, **_kwargs):
        commands.append(command)

        if command[1:3] == ["config", "current-context"]:
            return {"exit_code": 0, "stdout": "target-cluster\n", "stderr": ""}
        if "get" in command and not any("label" in earlier for earlier in commands):
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": 'Error from server (NotFound): pods "db" not found',
            }
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("pheragent.deploy_agent.sys.stdin", Terminal(answer))
    monkeypatch.setattr("pheragent.deploy_agent._command", run_command)
    result = _execute(decision, task, Evidence(), tmp_path, enabled=True, approve=False, timeout=10)
    assert (result["status"] == "validated") is should_run
    prompt = capsys.readouterr().out
    assert "Change: Pod has the label" in prompt
    assert "Source: repository-1:README.md" in prompt
    assert "Check: kubectl get pod db -n postgres" in prompt
    assert any(command[3] == "label" for command in commands) is should_run


def test_cli_reports_usage_and_verified_changes(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("pheragent.deploy_agent_cli.load_dotenv", lambda _path: None)
    monkeypatch.setattr(
        "pheragent.deploy_agent_cli.run_deployment_agent",
        lambda *_args, **_kwargs: {
            "status": "SUCCESS",
            "reason": "verified",
            "usage": {
                "requests": 3,
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
            },
            "mutating_actions": 2,
            "state": {
                "milestones": ["storage ready"],
                "verified_outcomes": [{"id": "storage"}],
            },
        },
    )
    assert deploy_main(["run", "task.yaml", "--output", str(tmp_path / "run")]) == 0
    summary = capsys.readouterr().out
    assert "3 requests; 120 tokens (input 100, output 20)" in summary
    assert "Actions attempted: 2" in summary
    assert "Validated changes: storage ready" in summary
    assert "New verified outcomes: storage" in summary


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


def test_dynamic_goal_counts_only_new_outcomes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in ("prepare", "storage", "service"):
        script = source / f"{name}.sh"
        script.write_text(f"#!/bin/sh\ntouch {name}-ready\n")
        script.chmod(0o755)
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, output)
    document = yaml.safe_load(task.read_text())
    document["task"]["stop_after_verified_outcomes"] = 2
    document.pop("success_checks")
    document["budgets"].update(max_cycles=5, max_mutating_actions=4)
    task.write_text(yaml.safe_dump(document))

    def action(name: str, outcome_id: str | None = None) -> Decision:
        return _decision(
            "ACT",
            "execute",
            command=[f"./{name}.sh"],
            working_directory="repository-1:.",
            evidence=[f"repository-1:{name}.sh"],
            expected_change=f"{name} ready",
            outcome_id=outcome_id,
            validation=[
                {
                    "command": ["ls", str(output / "workspace/repository-1")],
                    "contains": f"{name}-ready",
                }
            ],
        )

    decisions = iter(
        [
            action("prepare"),
            action("storage", "storage"),
            action("storage", "storage"),
            action("service", "service"),
        ]
    )

    def decide(_state, _observation, _last_result, _sources, _cycle):
        return next(decisions), {}

    report = run_deployment_agent(task, output, execute=True, decide=decide)
    assert report["status"] == "SUCCESS"
    assert report["mutating_actions"] == 3
    assert [item["id"] for item in report["state"]["verified_outcomes"]] == ["storage", "service"]
    actions = [json.loads(line) for line in (output / "actions.jsonl").read_text().splitlines()]
    assert [item["result"]["status"] for item in actions] == [
        "validated",
        "validated",
        "already_satisfied",
        "validated",
    ]


@pytest.mark.parametrize("script_body", ["touch already-ready", "exit 1"])
def test_preexisting_state_does_not_count_as_new_outcome(tmp_path: Path, script_body: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    script = source / "install.sh"
    script.write_text(f"#!/bin/sh\n{script_body}\n")
    script.chmod(0o755)
    (source / "already-ready").touch()
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, output)
    document = yaml.safe_load(task.read_text())
    document["task"]["stop_after_verified_outcomes"] = 1
    document.pop("success_checks")
    task.write_text(yaml.safe_dump(document))
    decisions = iter(
        [
            _decision(
                "ACT",
                "execute",
                command=["./install.sh"],
                working_directory="repository-1:.",
                evidence=["repository-1:install.sh"],
                expected_change="already ready",
                outcome_id="existing",
                validation=[
                    {
                        "command": ["ls", str(output / "workspace/repository-1")],
                        "contains": "already-ready",
                    }
                ],
            ),
            _decision("BLOCKED", reason="nothing else can be deployed"),
        ]
    )

    def decide(_state, _observation, _last_result, _sources, _cycle):
        return next(decisions), {}

    report = run_deployment_agent(task, output, execute=True, decide=decide)
    assert report["status"] == "BLOCKED"
    assert report["mutating_actions"] == 0
    assert report["state"]["verified_outcomes"] == []


def test_agent_asks_human_to_choose_between_supported_routes(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Two supported installation routes.\n")
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, output)

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr("pheragent.deploy_agent.sys.stdin", Terminal("2\n"))
    decisions = iter(
        [
            _decision(
                "ASK_HUMAN",
                reason="Both routes fit",
                focus="storage route",
                options=["Use provider A", "Use provider B"],
                evidence=["repository-1:README.md"],
            ),
            _decision("BLOCKED", reason="choice recorded"),
        ]
    )

    def decide(_state, _observation, _last_result, _sources, _cycle):
        return next(decisions), {}

    report = run_deployment_agent(task, output, decide=decide)
    assert report["state"]["selected_route"]["choice"] == "Use provider B"
    assert report["mutating_actions"] == 0


def test_unknown_preflight_does_not_execute(tmp_path: Path, monkeypatch) -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "shell", "sandbox": True},
        }
    )
    decision = _decision(
        "ACT",
        "execute",
        command=["touch", str(tmp_path / "ready")],
        evidence=["repository-1:README.md"],
        expected_change="ready",
        validation=[{"command": ["ls", str(tmp_path / "ready")]}],
    )

    class Evidence:
        def existing_refs(self, _references):
            return {"repository-1:README.md"}

    calls = []

    def cannot_read(command, _task, **_kwargs):
        calls.append(command)
        return {"exit_code": 1, "stdout": "", "stderr": "Permission denied"}

    monkeypatch.setattr("pheragent.deploy_agent._source_grounded", lambda *_args: True)
    monkeypatch.setattr("pheragent.deploy_agent._command", cannot_read)
    result = _execute(decision, task, Evidence(), tmp_path, enabled=True, approve=False, timeout=10)
    assert result["status"] == "needs_better_check"
    assert calls == [["ls", str(tmp_path / "ready")]]


def test_successful_command_without_verified_change_is_failure(tmp_path: Path, monkeypatch) -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "shell", "sandbox": True},
        }
    )
    decision = _decision(
        "ACT",
        "execute",
        command=["touch", str(tmp_path / "wrong-file")],
        evidence=["repository-1:README.md"],
        expected_change="ready",
        validation=[{"command": ["ls", str(tmp_path / "ready")]}],
    )

    class Evidence:
        def existing_refs(self, _references):
            return {"repository-1:README.md"}

    def run_command(command, _task, **_kwargs):
        if command[0] == "ls":
            return {"exit_code": 1, "stdout": "", "stderr": "No such file or directory"}
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("pheragent.deploy_agent._source_grounded", lambda *_args: True)
    monkeypatch.setattr("pheragent.deploy_agent._command", run_command)
    result = _execute(decision, task, Evidence(), tmp_path, enabled=True, approve=False, timeout=0)
    assert result["status"] == "verification_failed"
    assert result["execution"]["exit_code"] == 0


def test_delayed_readiness_is_verified_before_next_action(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    script = source / "install.sh"
    script.write_text("#!/bin/sh\ntouch ready\n")
    script.chmod(0o755)
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, output)
    document = yaml.safe_load(task.read_text())
    document["task"]["stop_after_verified_outcomes"] = 1
    document.pop("success_checks")
    task.write_text(yaml.safe_dump(document))
    reads = 0

    def delayed_checks(task, checks):
        nonlocal reads
        reads += 1
        result = _checks(task, checks)
        if reads == 2:
            result[0]["passed"] = False
        return result

    monkeypatch.setattr("pheragent.deploy_agent.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("pheragent.deploy_agent._checks", delayed_checks)
    decision = _decision(
        "ACT",
        "execute",
        command=["./install.sh"],
        working_directory="repository-1:.",
        evidence=["repository-1:install.sh"],
        expected_change="service ready",
        outcome_id="service",
        validation=[{"command": ["ls", str(output / "workspace/repository-1/ready")]}],
    )

    def decide(_state, _observation, _last_result, _sources, _cycle):
        return decision, {}

    report = run_deployment_agent(task, output, execute=True, decide=decide)
    actions = [json.loads(line) for line in (output / "actions.jsonl").read_text().splitlines()]
    assert report["status"] == "SUCCESS"
    assert report["mutating_actions"] == 1
    assert [item["result"]["status"] for item in actions] == ["validated"]
    assert reads >= 3
    assert [item["id"] for item in report["state"]["verified_outcomes"]] == ["service"]
