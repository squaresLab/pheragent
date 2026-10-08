from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest
import yaml

from pheragent.cli import main as pheragent_main
from pheragent.deployment.agent import (
    _INNER_LOOP_INSTRUCTIONS,
    compact_context,
    run_deployment_agent,
)
from pheragent.deployment.models import SourceKind, SourcesConfig
from pheragent.deployment.runtime import (
    classify_command,
    execute_action,
    inspect_target,
    run_checks,
    target_command,
)
from pheragent.deployment.source_manager import SourceManager
from pheragent.deployment.task import Check, Decision, DeploymentTask


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
        "source": None,
        "required_inputs": [],
        "expected_change": None,
        "validation": [],
        "add_gaps": [],
        "resolve_gaps": [],
        "add_questions": [],
        "resolve_questions": [],
    }
    payload.update(changes)
    return Decision.model_validate(payload)


def _execution_events(output: Path) -> list[dict]:
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    return [event for event in events if event["event"] == "tool" and event["tool"] == "execute"]


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
                validation=[
                    {
                        "command": ["ls", str(output / "workspace/repository-1/ready")],
                        "unsatisfied_exit_codes": [2],
                    }
                ],
            ),
            _decision("DONE"),
        ]
    )

    def decide(_state, _observation, _last_result, _sources, _cycle):
        return next(decisions), {"requests": 1}

    report = run_deployment_agent(task, output, execute=True, approve=True, decide=decide)
    assert report["status"] == "SUCCESS"
    assert report["mutating_actions"] == 1
    assert (output / "workspace/repository-1/ready").is_file()
    assert json.loads((output / "final-report.json").read_text())["status"] == "SUCCESS"
    assert (output / "events.jsonl").is_file()
    assert (output / "run-summary.json").is_file()


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
            validation=[
                {
                    "command": ["ls", str(output / "workspace/repository-1/ready")],
                    "unsatisfied_exit_codes": [2],
                }
            ],
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
                validation=[
                    {
                        "command": ["ls", str(output / "workspace/repository-1/ready")],
                        "unsatisfied_exit_codes": [2],
                    }
                ],
            ), {}
        return _decision("BLOCKED", reason="installer failed"), {}

    report = run_deployment_agent(task, output, execute=True, approve=True, decide=decide)
    assert report["status"] == "BLOCKED"
    assert calls[1]["status"] == "command_failed"
    assert calls[1]["execution"]["exit_code"] == 1


def test_three_incomplete_actions_stop_the_agent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Run the installer.\n")
    task = tmp_path / "task.yaml"
    output = tmp_path / "incomplete"
    _task(task, source, output)
    calls = []

    def decide(_state, _observation, last_result, _sources, _cycle):
        calls.append(last_result)
        return _decision(
            "ACT",
            "execute",
            command=["touch", "ready"],
            evidence=["repository-1:README.md"],
            expected_change="ready file exists",
        ), {}

    report = run_deployment_agent(task, output, execute=True, approve=True, decide=decide)
    assert report["status"] == "BLOCKED"
    assert report["reason"] == "three incomplete action proposals"
    assert calls[1]["status"] == "needs_revision"
    assert calls[1]["revision_attempt"] == 1
    assert len(calls) == 3


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


def test_agent_waits_for_referenced_secret_and_resumes_same_run(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Deploy the sample.\n")
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, output)
    document = yaml.safe_load(task.read_text())
    document["inputs"] = {"service_token": {"from_env": "SAMPLE_SERVICE_TOKEN", "sensitive": True}}
    task.write_text(yaml.safe_dump(document))
    monkeypatch.delenv("SAMPLE_SERVICE_TOKEN", raising=False)

    def wait(_state, _observation, _last_result, _sources, _cycle):
        return _decision(
            "WAITING_FOR_INPUT",
            reason="service token is required",
            required_inputs=[
                {
                    "name": "service_token",
                    "prompt": "Service token",
                    "sensitive": True,
                }
            ],
        ), {}

    first = run_deployment_agent(task, output, decide=wait)
    assert first["status"] == "WAITING_FOR_INPUT"
    request = yaml.safe_load((output / "human-request.yaml").read_text())
    assert request["required_inputs"]["service_token"]["available"] is False
    assert request["required_inputs"]["service_token"]["prompt"] == "Service token"
    assert request["required_inputs"]["service_token"]["sensitive"] is True
    assert "SAMPLE_SERVICE_TOKEN" not in (output / "human-request.yaml").read_text() or (
        request["required_inputs"]["service_token"]["source"] == "env:SAMPLE_SERVICE_TOKEN"
    )

    monkeypatch.setenv("SAMPLE_SERVICE_TOKEN", "not-written-to-artifacts")

    def stop(state, _observation, _last_result, _sources, _cycle):
        assert state["inputs"]["service_token"]["available"] is True
        assert "not-written-to-artifacts" not in json.dumps(state)
        return _decision("BLOCKED", reason="resume verified"), {}

    second = run_deployment_agent(task, output, resume=True, decide=stop)
    assert second["reason"] == "resume verified"


def test_resume_uses_current_task_constraints(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Deploy the sample.\n")
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, output)

    def stop(state, _observation, _last_result, _sources, _cycle):
        assert state["constraints"]["allow_destructive_actions"] is False
        return _decision("BLOCKED", reason="approval required"), {}

    run_deployment_agent(task, output, decide=stop)
    document = yaml.safe_load(task.read_text())
    document["constraints"] = {"allow_destructive_actions": True}
    task.write_text(yaml.safe_dump(document))

    def resume(state, _observation, _last_result, _sources, _cycle):
        assert state["constraints"]["allow_destructive_actions"] is True
        return _decision("BLOCKED", reason="updated constraints loaded"), {}

    report = run_deployment_agent(task, output, resume=True, decide=resume)
    assert report["reason"] == "updated constraints loaded"


def test_agent_collects_and_persists_interactive_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Deploy the sample.\n")
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, output)

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr("pheragent.deployment.agent.sys.stdin", Terminal("iam.example.org\n"))
    decisions = iter(
        [
            _decision(
                "WAITING_FOR_INPUT",
                reason="IAM hostname is required",
                required_inputs=[
                    {
                        "name": "iam_hostname",
                        "prompt": "IAM hostname",
                        "sensitive": False,
                    }
                ],
            ),
            _decision("BLOCKED", reason="input received"),
        ]
    )

    def decide(state, _observation, _last_result, _sources, _cycle):
        decision = next(decisions)
        if decision.kind == "BLOCKED":
            assert state["inputs"]["iam_hostname"]["value"] == "iam.example.org"
        return decision, {}

    report = run_deployment_agent(task, output, decide=decide)
    effective_task = json.loads((output / "task.json").read_text())
    assert report["reason"] == "input received"
    assert effective_task["inputs"]["iam_hostname"]["value"] == "iam.example.org"


def test_agent_hides_and_never_persists_interactive_secrets(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Deploy the sample.\n")
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, output)
    secret = "not-written-anywhere"

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("IAM_ADMIN_PASSWORD", raising=False)
    monkeypatch.setattr("pheragent.deployment.agent.sys.stdin", Terminal())
    monkeypatch.setattr("pheragent.deployment.agent.getpass", lambda _prompt: secret)
    decisions = iter(
        [
            _decision(
                "WAITING_FOR_INPUT",
                reason="IAM password is required",
                required_inputs=[
                    {
                        "name": "IAM_ADMIN_PASSWORD",
                        "prompt": "IAM administrator password",
                        "sensitive": True,
                    }
                ],
            ),
            _decision("BLOCKED", reason="secret received"),
        ]
    )

    def decide(state, _observation, _last_result, _sources, _cycle):
        decision = next(decisions)
        if decision.kind == "BLOCKED":
            assert state["inputs"]["IAM_ADMIN_PASSWORD"]["available"] is True
            assert "value" not in state["inputs"]["IAM_ADMIN_PASSWORD"]
        return decision, {}

    report = run_deployment_agent(task, output, decide=decide)
    artifacts = "\n".join(
        path.read_text(errors="replace") for path in output.rglob("*") if path.is_file()
    )
    effective_task = json.loads((output / "task.json").read_text())
    assert report["reason"] == "secret received"
    assert os.environ["IAM_ADMIN_PASSWORD"] == secret
    assert secret not in artifacts
    assert effective_task["inputs"]["IAM_ADMIN_PASSWORD"] == {
        "from_env": "IAM_ADMIN_PASSWORD",
        "sensitive": True,
    }


def test_agent_requests_approval_for_a_linked_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text(
        "Continue with https://github.com/example/deployment-infra.\n"
    )
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, output)

    def decide(_state, _observation, _last_result, _sources, _cycle):
        return _decision(
            "ACT",
            "add_source",
            source={"location": "https://github.com/example/deployment-infra"},
            evidence=["repository-1:README.md"],
            reason="the active deployment guide delegates infrastructure setup",
        ), {}

    report = run_deployment_agent(task, output, decide=decide)
    request = yaml.safe_load((output / "human-request.yaml").read_text())
    assert report["status"] == "WAITING_FOR_INPUT"
    assert request["kind"] == "source_approval"
    assert request["source"]["location"] == "https://github.com/example/deployment-infra"


def test_agent_acquires_an_approved_linked_source(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text(
        "Continue with https://github.com/example/deployment-infra.\n"
    )
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "README.md").write_text("Infrastructure instructions.\n")
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, output)
    acquire = SourceManager.acquire

    def acquire_locally(manager, config):
        if len(config.sources) == 1:
            return acquire(manager, config)
        replacement = config.sources[-1].model_copy(
            update={"kind": SourceKind.LOCAL_DIRECTORY, "location": str(linked)}
        )
        return acquire(
            manager,
            SourcesConfig(system=config.system, sources=[*config.sources[:-1], replacement]),
        )

    monkeypatch.setattr(SourceManager, "acquire", acquire_locally)
    decisions = iter(
        [
            _decision(
                "ACT",
                "add_source",
                source={"location": "https://github.com/example/deployment-infra"},
                evidence=["repository-1:README.md"],
            ),
            _decision("BLOCKED", reason="source was available for the next decision"),
        ]
    )

    def decide(_state, _observation, _last_result, sources, _cycle):
        decision = next(decisions)
        if decision.kind == "BLOCKED":
            assert "repository-2:README.md" in sources.readable_paths
        return decision, {}

    report = run_deployment_agent(task, output, approve=True, decide=decide)
    effective_task = json.loads((output / "task.json").read_text())
    assert report["reason"] == "source was available for the next decision"
    assert effective_task["sources"]["repositories"][-1]["location"] == (
        "https://github.com/example/deployment-infra"
    )
    assert any(
        json.loads(line)["event"] == "source_approved"
        for line in (output / "events.jsonl").read_text().splitlines()
    )


def test_sensitive_input_cannot_be_stored_inline() -> None:
    with pytest.raises(ValueError, match="sensitive inputs cannot be stored inline"):
        DeploymentTask.model_validate(
            {
                "task": {"objective": "sample"},
                "sources": {"repositories": ["/tmp/sample"]},
                "environment": {"type": "shell"},
                "inputs": {"token": {"value": "secret", "sensitive": True}},
            }
        )


def test_agent_prompt_requires_route_choice_and_deployment_progress() -> None:
    assert "senior DevOps engineer" in _INNER_LOOP_INSTRUCTIONS
    assert "active_step" in _INNER_LOOP_INSTRUCTIONS
    assert "private chain-of-thought" in _INNER_LOOP_INSTRUCTIONS


def test_agent_prompt_expands_missing_prerequisites_before_blocking() -> None:
    assert "missing prerequisite" in _INNER_LOOP_INSTRUCTIONS
    assert "source-supported" in _INNER_LOOP_INSTRUCTIONS
    assert "return to the original" in " ".join(_INNER_LOOP_INSTRUCTIONS.split())
    assert "validation" in _INNER_LOOP_INSTRUCTIONS


def test_agent_prompt_requires_source_defined_validation_semantics() -> None:
    assert "zero exit code does not prove success" in " ".join(
        _INNER_LOOP_INSTRUCTIONS.split()
    ).casefold()
    assert "do not retry a failed command" in " ".join(
        _INNER_LOOP_INSTRUCTIONS.split()
    ).casefold()


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
                        {
                            "command": [
                                "ls",
                                str(output / "workspace/repository-1/storage-ready"),
                            ],
                            "unsatisfied_exit_codes": [2],
                        }
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
                    validation=[
                        {
                            "command": ["ls", str(output / "workspace/repository-1/ready")],
                            "unsatisfied_exit_codes": [2],
                        }
                    ],
            ),
            _decision("DONE"),
        ]
    )
    states = []

    def decide(state, _observation, _last_result, _sources, _cycle):
        states.append((state["gaps"].copy(), state["milestones"].copy()))
        return next(decisions), {}

    report = run_deployment_agent(task, output, execute=True, approve=True, decide=decide)
    assert report["status"] == "SUCCESS"
    assert report["mutating_actions"] == 2
    assert states[1] == (["storage missing"], ["storage ready"])
    assert report["state"]["gaps"] == []
    assert report["state"]["goal_stack"] == []
    assert report["state"]["milestones"] == ["storage ready", "application ready"]


def test_brief_retains_a_long_result_start_and_end() -> None:
    result = "Default route is in this header.\n" + "x" * 7000 + "\nLast error is here."
    brief = compact_context(result)
    assert "Default route" in brief
    assert "Last error" in brief
    assert len(brief) < len(result)


def test_brief_keeps_a_complete_selected_file_intact() -> None:
    text = "header\n" + "deployment step\n" * 600 + "make start\n"
    brief = compact_context({"complete": True, "text": text, "total_lines": 602})
    assert brief["text"] == text


def test_policy_denies_direct_and_disguised_destructive_commands() -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "shell", "sandbox": True},
        }
    )
    assert classify_command(["/bin/rm", "-rf", "/tmp/sample"], task, mutating=True).startswith(
        "denied"
    )
    assert classify_command(
        ["bash", "-c", "rm -rf /tmp/sample"], task, mutating=True
    ).startswith("denied")


def test_kubernetes_actions_use_the_declared_context() -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "kubernetes", "context": "target-cluster"},
            "constraints": {"allowed_namespaces": ["*"]},
        }
    )
    assert target_command(["kubectl", "get", "pods"], task) == [
        "kubectl",
        "--context",
        "target-cluster",
        "get",
        "pods",
    ]
    with pytest.raises(ValueError, match="must match the declared target context"):
        target_command(["kubectl", "--context", "other", "get", "pods"], task)


def test_matching_context_storage_check_is_read_only() -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "kubernetes", "context": "test-cluster"},
        }
    )
    command = ["kubectl", "--context", "test-cluster", "get", "storageclass"]
    assert classify_command(command, task, mutating=False) == "allowed"
    assert target_command(command, task) == command


def test_kubernetes_global_options_do_not_hide_a_read_only_verb() -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "kubernetes", "context": "test-cluster"},
        }
    )
    command = [
        "kubectl",
        "-n",
        "istio-system",
        "get",
        "deployment/istio-ingressgateway",
        "service/istio-ingressgateway",
        "-o",
        "wide",
    ]
    assert classify_command(command, task, mutating=False) == "allowed"


@pytest.mark.parametrize(
    "command",
    [
        ["helm", "repo", "list"],
        ["helm", "search", "repo", "postgresql"],
        ["helm", "history", "postgres"],
        ["helm", "lint", "chart"],
        ["helm", "template", "demo", "chart"],
        ["kubectl", "auth", "can-i", "get", "pods"],
        ["kubectl", "config", "current-context"],
        ["kubectl", "explain", "deployments"],
        ["kubectl", "events", "-A"],
        ["aws", "sts", "get-caller-identity"],
        ["aws", "ec2", "describe-instances"],
        ["aws", "eks", "describe-cluster", "--name", "example"],
        ["ansible", "--version"],
        ["ansible-playbook", "--syntax-check", "site.yaml"],
        ["bash", "-n", "install.sh"],
        ["git", "diff", "--stat"],
        ["grep", "-Fx", "  host: iam.example.org", "values.yaml"],
        ["stat", "install.sh"],
        ["docker", "ps", "--all"],
        ["docker", "images"],
        ["docker", "image", "ls"],
        ["docker", "inspect", "demo"],
        ["docker", "events", "--until", "1s"],
        ["docker", "logs", "demo"],
        ["docker", "compose", "version"],
        ["docker", "compose", "ps", "--all"],
        ["docker", "compose", "-f", "compose.yaml", "ps", "--all"],
        ["docker", "compose", "--env-file", ".env", "-f", "compose.yaml", "ps"],
        ["docker", "compose", "ls"],
        [
            "docker",
            "compose",
            "--env-file",
            ".env",
            "-f",
            "compose.yaml",
            "config",
            "--quiet",
        ],
        ["lsblk"],
        ["lscpu"],
        ["whoami"],
    ],
)
def test_supported_read_only_probes(command: list[str]) -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "kubernetes", "context": "target-cluster"},
        }
    )
    assert classify_command(command, task, mutating=False) == "allowed"


@pytest.mark.parametrize(
    "command",
    [
        ["helm", "repo", "update"],
        ["kubectl", "config", "use-context", "other-cluster"],
        ["aws", "ec2", "start-instances", "--instance-ids", "i-example"],
        ["aws", "ec2", "describe-instances", "--profile", "other-account"],
        ["aws", "ec2", "describe-instances", "--endpoint-url", "https://example.invalid"],
        ["ansible-playbook", "--syntax-check", "--flush-cache", "site.yaml"],
        ["ansible-playbook", "site.yaml"],
        ["bash", "-n", "-i", "install.sh"],
        ["bash", "install.sh"],
        ["docker", "compose", "config"],
        ["docker", "compose", "up", "-d"],
        ["docker", "compose", "-f", "compose.yaml", "up", "-d"],
        ["docker", "events"],
    ],
)
def test_commands_not_known_to_be_read_only_require_review(command: list[str]) -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "kubernetes", "context": "target-cluster"},
        }
    )
    assert classify_command(command, task, mutating=False).startswith(
        ("approval_required", "high_risk_approval", "denied")
    )
    if command[0] == "aws":
        assert classify_command(command, task, mutating=True).startswith("denied")


def test_policy_uses_review_as_the_default_for_unknown_mutations() -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "shell"},
        }
    )
    assert classify_command(["make", "start"], task, mutating=True).startswith(
        "approval_required"
    )


@pytest.mark.parametrize(
    "command",
    [
        ["sudo", "apt-get", "install", "open-iscsi"],
        ["kubectl", "delete", "pod", "db", "-n", "postgres"],
        ["helm", "uninstall", "postgres", "-n", "postgres"],
        ["terraform", "apply"],
        ["aws", "ec2", "start-instances", "--instance-ids", "i-example"],
    ],
)
def test_high_risk_changes_have_a_distinct_review_tier(command: list[str]) -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "kubernetes", "context": "target-cluster"},
            "constraints": {
                "allow_new_infrastructure": True,
                "allow_destructive_actions": True,
                "allowed_namespaces": ["*"],
            },
        }
    )
    assert classify_command(command, task, mutating=True).startswith("high_risk_approval")


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
    monkeypatch.setattr("pheragent.deployment.agent.observe_target", lambda _task: {})
    observed = []

    def inspect(command, _task, **_kwargs):
        observed.append(command)
        return {"exit_code": 0, "stdout": "No resources found", "stderr": ""}

    monkeypatch.setattr("pheragent.deployment.runtime._command", inspect)
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
    assert not _execution_events(output)


def test_observation_uses_selected_source_directory(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "compose.yaml").write_text("services: {}\n")
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, output)
    monkeypatch.setattr("pheragent.deployment.agent.observe_target", lambda _task: {})
    working_directories = []

    def inspect(_command, _task, **kwargs):
        working_directories.append(kwargs.get("cwd"))
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("pheragent.deployment.runtime._command", inspect)
    decisions = iter(
        [
            _decision(
                "ACT",
                "observe",
                command=["docker", "compose", "-f", "compose.yaml", "ps"],
                working_directory="repository-1:.",
            ),
            _decision("BLOCKED", reason="inspection complete"),
        ]
    )

    def decide(_state, _observation, _last_result, _sources, _cycle):
        return next(decisions), {}

    report = run_deployment_agent(task, output, decide=decide)
    assert report["status"] == "BLOCKED"
    assert working_directories == [output / "workspace/repository-1"]
    assert (working_directories[0] / "compose.yaml").is_file()


def test_unknown_observation_can_receive_one_time_human_approval(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "shell", "sandbox": True},
        }
    )
    decision = _decision(
        "ACT",
        "observe",
        command=["projectctl", "inspect"],
        reason="inspect the project without changing it",
    )

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    commands = []
    monkeypatch.setattr("pheragent.deployment.runtime.sys.stdin", Terminal("y\n"))
    monkeypatch.setattr(
        "pheragent.deployment.runtime._command",
        lambda command, _task, **_kwargs: commands.append(command)
        or {"exit_code": 0, "stdout": "ready", "stderr": ""},
    )
    result = inspect_target(
        decision,
        task,
        object(),
        tmp_path,
        enabled=True,
        approve=False,
    )
    assert result["exit_code"] == 0
    assert commands == [["projectctl", "inspect"]]
    assert "Approve unclassified observation?" in capsys.readouterr().out


def test_reworded_denied_observations_stop_early(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Sample deployment source\n")
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, output)
    monkeypatch.setattr("pheragent.deployment.agent.observe_target", lambda _task: {})
    commands = iter(
        [
            ["kubectl", "config", "use-context", "first"],
            ["kubectl", "config", "use-context", "second"],
            ["kubectl", "config", "use-context", "third"],
        ]
    )
    calls = 0

    def decide(_state, _observation, _last_result, _sources, _cycle):
        nonlocal calls
        calls += 1
        return _decision("ACT", "observe", command=next(commands)), {}

    report = run_deployment_agent(task, output, decide=decide)
    assert report["status"] == "BLOCKED"
    assert calls == 3
    assert "same policy denial repeated three times" in report["reason"]


def test_policy_denial_is_returned_to_the_agent_for_revision(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Run the installer.\n")
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, output)
    decisions = iter(
        [
            _decision(
                "ACT",
                "execute",
                command=["python3", "-c", "print('unsafe wrapper')"],
                evidence=["repository-1:README.md"],
                expected_change="application ready",
                validation=[{"command": ["ls", str(source / "ready")]}],
            ),
            _decision("BLOCKED", reason="request a supported alternative"),
        ]
    )
    calls = 0

    def decide(_state, _observation, last_result, _sources, _cycle):
        nonlocal calls
        if calls:
            assert last_result["status"] == "policy_denied"
            assert "inline interpreter" in last_result["reason"]
        calls += 1
        return next(decisions), {}

    report = run_deployment_agent(task, output, execute=True, decide=decide)
    assert report["status"] == "BLOCKED"
    assert report["reason"] == "request a supported alternative"
    assert calls == 2


def test_unavailable_stdin_reference_is_returned_for_revision(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("Run the installer.\n")
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, output)
    decisions = iter(
        [
            _decision(
                "ACT",
                "execute",
                command=["./install.sh"],
                stdin_input="INSTALL_CONFIRMATION",
                evidence=["repository-1:README.md"],
                expected_change="application ready",
                validation=[{"command": ["ls", str(source / "ready")]}],
            ),
            _decision("BLOCKED", reason="ask for the missing input"),
        ]
    )
    calls = 0

    def decide(_state, _observation, last_result, _sources, _cycle):
        nonlocal calls
        if calls:
            assert last_result["status"] == "needs_revision"
            assert "structured required_inputs" in last_result["reason"]
        calls += 1
        return next(decisions), {}

    report = run_deployment_agent(task, output, execute=True, decide=decide)
    assert report["reason"] == "ask for the missing input"
    assert calls == 2


def test_unfamiliar_mutation_requires_review_and_wrong_context_is_denied() -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "kubernetes", "context": "target-cluster"},
            "constraints": {"allowed_namespaces": ["postgres"]},
        }
    )
    assert classify_command(
        ["kubectl", "label", "pod", "db", "tested=yes", "-n", "postgres"], task, mutating=True
    ).startswith("approval_required")
    assert classify_command(
        ["kubectl", "--context", "other-cluster", "get", "pods"], task, mutating=False
    ).startswith("denied")


@pytest.mark.parametrize(
    ("command", "answer", "prompt_text", "should_run"),
    [
        (
            ["kubectl", "label", "pod", "db", "tested=yes", "-n", "postgres"],
            "n\n",
            "Approve deployment action",
            False,
        ),
        (
            ["kubectl", "label", "pod", "db", "tested=yes", "-n", "postgres"],
            "y\n",
            "Approve deployment action",
            True,
        ),
        (
            ["kubectl", "delete", "pod", "db", "-n", "postgres"],
            "y\n",
            "Approve HIGH-RISK deployment action",
            False,
        ),
        (
            ["kubectl", "delete", "pod", "db", "-n", "postgres"],
            "approve\n",
            "Approve HIGH-RISK deployment action",
            True,
        ),
    ],
)
def test_kubernetes_change_waits_for_terminal_approval(
    tmp_path: Path,
    monkeypatch,
    capsys,
    command: list[str],
    answer: str,
    prompt_text: str,
    should_run: bool,
) -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "kubernetes", "context": "target-cluster"},
            "constraints": {
                "allow_destructive_actions": True,
                "allowed_namespaces": ["postgres"],
            },
        }
    )
    decision = _decision(
        "ACT",
        "execute",
        command=command,
        evidence=["repository-1:README.md"],
        expected_change="Pod has the label",
        validation=[
            {
                "command": ["kubectl", "get", "pod", "db", "-n", "postgres"],
                "unsatisfied_exit_codes": [1],
            }
        ],
    )

    class Evidence:
        def existing_refs(self, _references):
            return {"repository-1:README.md"}

        def working_directory(self, _decision, _workspace):
            return None

        def grounds(self, _decision):
            return True

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    commands = []

    def run_command(command, _task, **_kwargs):
        commands.append(command)

        if command[1:3] == ["config", "current-context"]:
            return {"exit_code": 0, "stdout": "target-cluster\n", "stderr": ""}
        if "get" in command and not any(
            len(earlier) > 3 and earlier[3] in {"delete", "label"} for earlier in commands
        ):
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": 'Error from server (NotFound): pods "db" not found',
            }
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("pheragent.deployment.runtime.sys.stdin", Terminal(answer))
    monkeypatch.setattr("pheragent.deployment.runtime._command", run_command)
    monkeypatch.setattr("pheragent.deployment.runtime._stream_command", run_command)
    result = execute_action(
        decision, task, Evidence(), tmp_path, enabled=True, approve=False, timeout=10
    )
    assert (result["status"] == "validated") is should_run
    prompt = capsys.readouterr().out
    assert prompt_text in prompt
    assert "Change: Pod has the label" in prompt
    assert "Source: repository-1:README.md" in prompt
    assert "Check: kubectl get pod db -n postgres" in prompt
    assert any(item[3] == command[1] for item in commands if len(item) > 3) is should_run


def test_cli_reports_usage_and_verified_changes(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("pheragent.cli.load_dotenv", lambda _path: None)
    monkeypatch.setattr(
        "pheragent.deployment.cli.run_deployment_agent",
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
    assert (
        pheragent_main(["deployment", "run", "task.yaml", "--output", str(tmp_path / "run")]) == 0
    )
    summary = capsys.readouterr().out
    assert "3 requests; 120 tokens (input 100, output 20)" in summary
    assert "Actions attempted: 2" in summary
    assert "Validated changes: storage ready" in summary
    assert "New verified outcomes: storage" in summary


def test_deployment_exposes_only_progressive_run() -> None:
    from pheragent.cli import _build_parser

    parser = _build_parser()
    assert parser.parse_args(["deployment", "run", "task.yaml"]).task == Path("task.yaml")
    assert parser.parse_args(["deployment", "run", "task.yaml", "--resume"]).resume is True
    with pytest.raises(SystemExit):
        parser.parse_args(["deployment", "analyze"])


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

    report = run_deployment_agent(task, output, execute=True, approve=True, decide=decide)
    assert report["status"] == "SUCCESS"
    assert report["mutating_actions"] == 3
    assert [item["id"] for item in report["state"]["verified_outcomes"]] == ["storage", "service"]
    actions = _execution_events(output)
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

    monkeypatch.setattr("pheragent.deployment.agent.sys.stdin", Terminal("2\n"))
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

        def working_directory(self, _decision, _workspace):
            return None

        def grounds(self, _decision):
            return True

    calls = []

    def cannot_read(command, _task, **_kwargs):
        calls.append(command)
        return {"exit_code": 1, "stdout": "", "stderr": "Permission denied"}

    monkeypatch.setattr("pheragent.deployment.runtime._command", cannot_read)
    result = execute_action(
        decision, task, Evidence(), tmp_path, enabled=True, approve=False, timeout=10
    )
    assert result["status"] == "needs_better_check"
    assert calls == [["ls", str(tmp_path / "ready")]]


def test_execute_streams_output_and_supplies_referenced_stdin(tmp_path: Path, capsys) -> None:
    source = tmp_path / "source"
    source.mkdir()
    script = source / "install.sh"
    script.write_text(
        "#!/bin/sh\nread answer\nprintf 'received:%s\\n' \"$answer\"\ntouch ready\n"
    )
    script.chmod(0o755)
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": [str(source)]},
            "environment": {"type": "shell", "sandbox": True},
        }
    )
    decision = _decision(
        "ACT",
        "execute",
        command=["./install.sh"],
        working_directory="repository-1:.",
        evidence=["repository-1:install.sh"],
        expected_change="installer completes",
        required_inputs=[
            {
                "name": "INSTALL_CONFIRMATION",
                "prompt": "Confirm installation",
                "sensitive": True,
            }
        ],
        stdin_input="INSTALL_CONFIRMATION",
        validation=[{"command": ["ls", "ready"], "unsatisfied_exit_codes": [2]}],
    )

    class Evidence:
        def existing_refs(self, _references):
            return {"repository-1:install.sh"}

        def working_directory(self, _decision, _workspace):
            return source

        def grounds(self, _decision):
            return True

    log = tmp_path / "actions" / "0001.log"
    secret = "super-secret-value\n"
    result = execute_action(
        decision,
        task,
        Evidence(),
        tmp_path,
        enabled=True,
        approve=True,
        timeout=10,
        stdin_value=secret,
        stdin_sensitive=True,
        log_path=log,
    )
    assert result["status"] == "validated"
    assert "[REDACTED INPUT]" in result["execution"]["stdout"]
    assert "super-secret-value" not in result["execution"]["stdout"]
    assert "[REDACTED INPUT]" in log.read_text()
    assert "[REDACTED INPUT]" in capsys.readouterr().out


def test_execute_retains_partial_output_when_it_times_out(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    script = source / "install.sh"
    script.write_text("#!/bin/sh\nprintf 'started\\n'\nsleep 10\n")
    script.chmod(0o755)
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": [str(source)]},
            "environment": {"type": "shell", "sandbox": True},
        }
    )
    decision = _decision(
        "ACT",
        "execute",
        command=["./install.sh"],
        working_directory="repository-1:.",
        evidence=["repository-1:install.sh"],
        expected_change="installer completes",
        validation=[{"command": ["ls", "ready"], "unsatisfied_exit_codes": [2]}],
    )

    class Evidence:
        def existing_refs(self, _references):
            return {"repository-1:install.sh"}

        def working_directory(self, _decision, _workspace):
            return source

        def grounds(self, _decision):
            return True

    log = tmp_path / "actions" / "0001.log"
    result = execute_action(
        decision,
        task,
        Evidence(),
        tmp_path,
        enabled=True,
        approve=True,
        timeout=1,
        log_path=log,
    )
    assert result["status"] == "command_failed"
    assert result["execution"]["timed_out"] is True
    assert "started" in result["execution"]["stdout"]
    assert "timed out" in result["execution"]["stderr"]
    assert "started" in log.read_text()


def test_checks_use_declared_generic_outcomes(tmp_path: Path, monkeypatch) -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "sample"},
            "sources": {"repositories": ["/tmp/sample"]},
            "environment": {"type": "shell", "sandbox": True},
        }
    )
    results = iter(
        [
            {"exit_code": 0, "stdout": "demo\n", "stderr": ""},
            {"exit_code": 0, "stdout": "", "stderr": ""},
            {"exit_code": 1, "stdout": "", "stderr": "not found"},
            {"exit_code": 2, "stdout": "", "stderr": "unavailable"},
        ]
    )
    monkeypatch.setattr(
        "pheragent.deployment.runtime._command", lambda *_args, **_kwargs: next(results)
    )
    checks = [
        Check(command=["docker", "inspect", "demo"], contains="demo"),
        Check(command=["docker", "inspect", "demo"], contains="demo"),
        Check(command=["docker", "inspect", "demo"], unsatisfied_exit_codes=[1]),
        Check(command=["docker", "inspect", "demo"], unsatisfied_exit_codes=[1]),
    ]
    assert [item["status"] for item in run_checks(task, checks, tmp_path)] == [
        "satisfied",
        "unsatisfied",
        "unsatisfied",
        "unknown",
    ]


def test_declared_unsatisfied_preflight_allows_execution(tmp_path: Path, monkeypatch) -> None:
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
        expected_change="demo becomes available",
        validation=[
            {
                "command": ["docker", "inspect", "demo"],
                "unsatisfied_exit_codes": [1],
            }
        ],
    )

    class Evidence:
        def existing_refs(self, _references):
            return {"repository-1:README.md"}

        def working_directory(self, _decision, _workspace):
            return None

        def grounds(self, _decision):
            return True

    responses = iter(
        [
            {"exit_code": 1, "stdout": "", "stderr": "not found"},
            {"exit_code": 0, "stdout": "", "stderr": ""},
            {"exit_code": 0, "stdout": "demo", "stderr": ""},
        ]
    )
    monkeypatch.setattr(
        "pheragent.deployment.runtime._command", lambda *_args, **_kwargs: next(responses)
    )
    result = execute_action(
        decision, task, Evidence(), tmp_path, enabled=True, approve=True, timeout=10
    )
    assert result["status"] == "validated"
    assert result["before_unsatisfied"] is True


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
        validation=[
            {
                "command": ["ls", str(tmp_path / "ready")],
                "unsatisfied_exit_codes": [1],
            }
        ],
    )

    class Evidence:
        def existing_refs(self, _references):
            return {"repository-1:README.md"}

        def working_directory(self, _decision, _workspace):
            return None

        def grounds(self, _decision):
            return True

    def run_command(command, _task, **_kwargs):
        if command[0] == "ls":
            return {"exit_code": 1, "stdout": "", "stderr": "No such file or directory"}
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("pheragent.deployment.runtime._command", run_command)
    result = execute_action(
        decision, task, Evidence(), tmp_path, enabled=True, approve=True, timeout=0
    )
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

    def delayed_checks(task, checks, cwd=None):
        nonlocal reads
        reads += 1
        result = run_checks(task, checks, cwd)
        if reads == 2:
            result[0]["status"] = "unsatisfied"
        return result

    monkeypatch.setattr("pheragent.deployment.runtime.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("pheragent.deployment.runtime.run_checks", delayed_checks)
    decision = _decision(
        "ACT",
        "execute",
        command=["./install.sh"],
        working_directory="repository-1:.",
        evidence=["repository-1:install.sh"],
        expected_change="service ready",
        outcome_id="service",
        validation=[
            {
                "command": ["ls", str(output / "workspace/repository-1/ready")],
                "unsatisfied_exit_codes": [2],
            }
        ],
    )

    def decide(_state, _observation, _last_result, _sources, _cycle):
        return decision, {}

    report = run_deployment_agent(task, output, execute=True, approve=True, decide=decide)
    actions = _execution_events(output)
    assert report["status"] == "SUCCESS"
    assert report["mutating_actions"] == 1
    assert [item["result"]["status"] for item in actions] == ["validated"]
    assert reads >= 3
    assert [item["id"] for item in report["state"]["verified_outcomes"]] == ["service"]
