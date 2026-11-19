from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pheragent.deployment.agent import run_deployment_agent
from pheragent.deployment.llm import LLMClient
from pheragent.deployment.runtime import _stream_command, classify_command
from pheragent.deployment.sources import SourceWorkspace, _source_names
from pheragent.deployment.task import Decision, DeploymentTask, RunNote


def _task(
    path: Path, source: Path, *, max_cycles: int = 6, invariants: dict[str, str] | None = None
) -> None:
    if source.is_dir() and not (source / "README.md").exists():
        (source / "README.md").write_text("Deploy the sample application.\n")
    path.write_text(
        yaml.safe_dump(
            {
                "task": {
                    "objective": "Deploy the sample application",
                    "invariants": invariants or {},
                },
                "sources": {"repositories": [str(source)]},
                "environment": {"type": "shell", "sandbox": True},
                "budgets": {"max_cycles": max_cycles, "max_runtime_minutes": 2},
            }
        )
    )


def _decision(action: str, reason: str, **values) -> Decision:
    return Decision.model_validate({"action": action, "reason": reason, **values})


def test_llm_client_retries_a_malformed_parsed_response(monkeypatch) -> None:
    expected = _decision("finish", "deployment complete")
    attempts = 0

    class ParsedResponse(SimpleNamespace):
        def model_dump(self):
            raise AssertionError("the full parsed response must not be serialized")

    class Responses:
        def parse(self, **request):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ValueError("malformed structured response")
            assert request["text_format"] is Decision
            return ParsedResponse(
                output_parsed=expected,
                output_text=expected.model_dump_json() * 2,
                usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(
        "pheragent.deployment.llm._openai_client",
        lambda **_kwargs: SimpleNamespace(responses=Responses()),
    )
    decision, usage = LLMClient().complete(Decision, instructions="decide", payload={})

    assert decision == expected
    assert attempts == 2
    assert usage["requests"] == 2
    assert usage["total_tokens"] == 5


def test_sources_use_repository_names_and_owners_only_for_clashes(tmp_path: Path) -> None:
    assert _source_names(
        ["https://github.com/open-telemetry/opentelemetry-demo.git"], tmp_path
    ) == ["opentelemetry-demo"]
    assert _source_names(
        ["https://github.com/first/deploy.git", "https://github.com/second/deploy.git"],
        tmp_path,
    ) == ["first/deploy", "second/deploy"]
    assert _source_names(["."], tmp_path) == [tmp_path.name]


def test_source_workspace_exposes_files_without_preloading_them(tmp_path: Path) -> None:
    source = tmp_path / "demo"
    (source / "docs").mkdir(parents=True)
    (source / "README.md").write_text("Start with [deployment](docs/install.md).")
    (source / "docs/install.md").write_text("Run make start.")
    workspace = SourceWorkspace((SimpleNamespace(id="demo"),))

    assert workspace.working_directory("demo/docs", tmp_path) == source / "docs"


def test_source_workspace_rejects_paths_outside_it(tmp_path: Path) -> None:
    source = tmp_path / "demo"
    source.mkdir()
    workspace = SourceWorkspace((SimpleNamespace(id="demo"),))

    with pytest.raises(ValueError, match="escapes"):
        workspace.working_directory("../outside", tmp_path)


def test_long_command_output_shows_omitted_part_and_keeps_full_log(tmp_path: Path) -> None:
    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "Inspect a file"},
            "sources": {"repositories": ["."]},
            "environment": {"type": "shell"},
        }
    )
    result = _stream_command(
        [sys.executable, "-c", "print('HEAD'); print('x' * 200); print('TAIL')"],
        task,
        cwd=tmp_path,
        timeout=5,
        stdin_value=None,
        stdin_sensitive=False,
        log_path=tmp_path / "output.log",
        output_limit=100,
    )

    assert result["output_truncated"] is True
    assert result["output_lines"] == 3
    assert result["stdout"].startswith("HEAD")
    assert f"full output: {tmp_path / 'output.log'}" in result["stdout"]
    assert result["stdout"].endswith("TAIL\n")
    assert "x" * 200 in (tmp_path / "output.log").read_text()


def test_first_decision_can_read_a_large_source_file_in_preview(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("start here\n" + "x" * 20_000 + "\n")
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, max_cycles=3)

    def decide(payload, iteration):
        if iteration == 1:
            assert "overview" not in payload
            assert payload["task"]["sources"] == ["source"]
            return _decision(
                "shell",
                "read source",
                command=["sed", "-n", "1,2p", "README.md"],
                working_directory="source",
            ), {}
        feedback = payload["recent_history"][-1]["result"]
        assert feedback["execution"]["output_truncated"] is True
        assert feedback["state_changing"] is False
        return _decision("finish", "source inspected"), {}

    report = run_deployment_agent(task, output, decide=decide)
    assert report["status"] == "FAILED"
    assert (output / "actions/0001.log").read_text().startswith("start here")


def test_preview_does_not_run_a_change(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, max_cycles=2)

    def decide(payload, iteration):
        if iteration == 1:
            return _decision(
                "shell", "create marker", command=["touch", "marker"], working_directory="source"
            ), {}
        assert payload["recent_history"][-1]["result"]["status"] == "dry_run"
        return _decision("finish", "preview complete"), {}

    run_deployment_agent(task, output, decide=decide)
    assert not (output / "workspace/source/marker").exists()


def test_agent_deploys_verifies_and_records_a_component(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "install.sh").write_text("#!/bin/sh\ntouch ready\n")
    (source / "install.sh").chmod(0o755)
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, max_cycles=4, invariants={"ready-file": "Installed artifact exists"})
    evidence: list[int] = []

    def decide(payload, iteration):
        if iteration == 1:
            assert payload["task"]["objective"] == "Deploy the sample application"
            assert payload["task"]["sources"] == ["source"]
            assert "overview" not in payload
            assert payload["current_step"].startswith("Inspect source guidance")
            return _decision(
                "shell",
                "run the documented installer",
                command=["./install.sh"],
                working_directory="source",
                notes=[
                    RunNote(key="installer", text="Run source/install.sh"),
                    RunNote(key="credential", text="password: do-not-log"),
                    RunNote(key="current-step", text="Verify the sample installation"),
                    RunNote(
                        key="progress-summary", text="Installer ran; runtime not verified yet."
                    ),
                ],
            ), {"requests": 1}
        if iteration == 2:
            assert payload["current_step"] == "Verify the sample installation"
            assert payload["progress_summary"] == "Installer ran; runtime not verified yet."
            assert "progress-summary" not in payload["run_notes"]["index"]
            evidence.append(payload["recent_history"][-1]["result"]["event_sequence"])
            return _decision(
                "shell",
                "verify the installed artifact",
                command=["test", "-f", "ready"],
                working_directory="source",
            ), {"requests": 1}
        if iteration == 3:
            assert "Record the current outcome" in payload["finalization"]
            evidence.append(payload["recent_history"][-1]["result"]["event_sequence"])
            return _decision(
                "record",
                "the sample application is deployed",
                component="sample application",
                deployed=True,
                evidence_events=evidence,
                verification_events=[evidence[-1]],
                notes=[
                    RunNote(key="progress-summary", text="Sample installation verified."),
                    RunNote(
                        key="invariant-ready-file",
                        text="Installed artifact exists",
                        evidence_events=[evidence[-1]],
                    ),
                ],
            ), {"requests": 1}
        assert "Final request" in payload["finalization"]
        assert payload["progress_summary"] == "Sample installation verified."
        return _decision("finish", "all discovered outcomes are deployed"), {"requests": 1}

    report = run_deployment_agent(task, output, execute=True, decide=decide)

    assert report["status"] == "SUCCESS"
    assert "mutating_actions" not in report
    assert report["components"][0]["deployed"] is True
    assert (output / "workspace/source/ready").is_file()
    assert "./install.sh" in (output / "runbook.sh").read_text()
    assert "test -f ready" in (output / "runbook.sh").read_text()
    assert "INCOMPLETE" not in (output / "runbook.sh").read_text()
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events].count("overview_created") == 0
    assert [event["event"] for event in events].count("note_recorded") == 6
    assert "Run source/install.sh" in (output / "run_notes.md").read_text()
    assert "do-not-log" not in (output / "events.jsonl").read_text()
    assert "do-not-log" not in (output / "run_notes.md").read_text()
    assert not (output / "overview.md").exists()


def test_runbook_replays_a_recorded_repair_from_clean_sources(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "install.sh").write_text("#!/bin/sh\nexit 1\n")
    (source / "install.sh").chmod(0o755)
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, max_cycles=6, invariants={"ready": "The installed application is ready"})
    evidence: list[int] = []

    def decide(payload, iteration):
        if iteration == 1:
            return _decision(
                "shell", "try the installer", command=["./install.sh"], working_directory="source"
            ), {}
        if iteration == 2:
            assert payload["recent_history"][-1]["result"]["execution"]["exit_code"] == 1
            return _decision(
                "shell",
                "repair the installer",
                command=["sed", "-i", "s/exit 1/touch ready/", "install.sh"],
                working_directory="source",
            ), {}
        if iteration == 3:
            evidence.append(payload["recent_history"][-1]["result"]["event_sequence"])
            return _decision(
                "shell",
                "install after repair",
                command=["./install.sh"],
                working_directory="source",
            ), {}
        if iteration == 4:
            evidence.append(payload["recent_history"][-1]["result"]["event_sequence"])
            return _decision(
                "shell",
                "verify installation",
                command=["test", "-f", "ready"],
                working_directory="source",
            ), {}
        if iteration == 5:
            evidence.append(payload["recent_history"][-1]["result"]["event_sequence"])
            return _decision(
                "record",
                "application is ready",
                component="application",
                deployed=True,
                evidence_events=evidence,
                verification_events=[evidence[-1]],
                notes=[
                    RunNote(
                        key="invariant-ready",
                        text="The installed application is ready",
                        evidence_events=[evidence[-1]],
                    )
                ],
            ), {}
        return _decision("finish", "deployment complete"), {}

    report = run_deployment_agent(task, output, execute=True, decide=decide)
    assert report["status"] == "SUCCESS"
    runbook = (output / "runbook.sh").read_text()
    assert runbook.index("sed -i") < runbook.index("./install.sh") < runbook.index("test -f ready")
    assert "INCOMPLETE" not in runbook
    assert str(output) not in runbook

    replay = tmp_path / "replay"
    (replay / "workspace").mkdir(parents=True)
    shutil.copytree(source, replay / "workspace/source")
    shutil.copy2(output / "runbook.sh", replay / "runbook.sh")
    result = subprocess.run(["sh", str(replay / "runbook.sh")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (replay / "workspace/source/ready").is_file()


def test_agent_resumes_from_notes_and_recent_history(tmp_path: Path) -> None:
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
                working_directory="source",
                notes=[
                    RunNote(key="install-route", text="Use source/install.sh"),
                    RunNote(key="current-step", text="Verify sample"),
                    RunNote(key="progress-summary", text="Installer ran; verification remains."),
                ],
            ),
            {},
        ),
    )
    assert first["status"] == "BUDGET_EXHAUSTED"

    _task(task, source, max_cycles=4)
    evidence: list[int] = []

    def resume(payload, iteration):
        assert payload["current_step"] == "Verify sample"
        assert payload["progress_summary"] == "Installer ran; verification remains."
        assert payload["run_notes"]["index"] == {"install-route": "Use source/install.sh"}
        if iteration == 2:
            assert "latest_feedback" not in payload
            evidence.append(payload["recent_history"][-1]["result"]["event_sequence"])
            return _decision(
                "shell",
                "verify sample",
                command=["test", "-f", "ready"],
                working_directory="source",
            ), {}
        if iteration == 3:
            evidence.append(payload["recent_history"][-1]["result"]["event_sequence"])
            return _decision(
                "record",
                "sample is deployed",
                component="sample",
                deployed=True,
                evidence_events=evidence,
            ), {}
        return _decision("finish", "deployment complete"), {}

    report = run_deployment_agent(task, output, execute=True, resume=True, decide=resume)

    assert report["status"] == "SUCCESS"
    assert report["cycles"] == 4
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
            seen_repeat_limit = "repeat_limit" in payload["recent_history"][-1]["result"]
            return _decision("finish", "no deployment outcome was found"), {}
        return _decision(
            "shell",
            f"wording {iteration}",
            command=["test", "-f", "missing"],
            working_directory="source",
        ), {}

    run_deployment_agent(task, output, execute=True, decide=decide)
    assert seen_repeat_limit


def test_agent_can_enter_a_directory_created_in_its_workspace(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, max_cycles=5)
    decisions = iter(
        [
            _decision(
                "shell",
                "create generated workspace",
                command=["mkdir", "generated"],
                working_directory="source",
            ),
            _decision(
                "shell",
                "inspect generated workspace",
                command=["pwd"],
                working_directory="source/generated",
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


def test_command_without_working_directory_runs_at_workspace_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    task = tmp_path / "task.yaml"
    output = tmp_path / "run"
    _task(task, source, max_cycles=4)
    decisions = iter(
        [
            _decision("shell", "inspect the workspace", command=["pwd"]),
            _decision("finish", "workspace inspected"),
        ]
    )

    report = run_deployment_agent(
        task,
        output,
        execute=True,
        decide=lambda _payload, _iteration: (next(decisions), {}),
    )

    assert report["latest_feedback"]["execution"]["stdout"].strip() == str(
        (output / "workspace").resolve()
    )


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
    assert classify_command(["sed", "-n", "1,120p", "README.md"], task) == "allowed"
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
