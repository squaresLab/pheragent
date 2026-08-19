from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pheragent.cli import main
from pheragent.deployment.errors import (
    DeploymentInputError,
    WorkflowExecutionError,
    WorkflowNotExecutableError,
)
from pheragent.deployment.execution import prepare_execution


def _workflow_payload() -> dict[str, object]:
    coverage = {
        "primary_roots_traced": True,
        "deployment_actions_accounted": True,
        "independent_gap_search_complete": True,
        "docs_repo_disagreements_reviewed": True,
        "docs_repo_disagreements_resolved": True,
        "external_requirements_reviewed": True,
        "dynamic_deployments_resolved": True,
        "deployable_components_grounded": True,
        "semantic_coverage_complete": True,
        "validation_checks_grounded": True,
        "documentation_evidence_reviewed": True,
    }

    def step(
        step_id: str,
        component_id: str,
        command: str,
        *,
        after: list[str],
    ) -> dict[str, object]:
        return {
            "id": step_id,
            "kind": "component",
            "targets": [{"id": component_id, "name": component_id.split("_", 1)[1].title()}],
            "executor": "shell",
            "source_ref": {"repo_id": "fixture", "path": "deploy.sh"},
            "working_directory": ".",
            "command": command,
            "required_inputs": [],
            "after": after,
            "status": "ready",
            "blockers": [],
        }

    return {
        "version": "0.1",
        "system": "fixture",
        "ready_for_execution": True,
        "coverage": coverage,
        "provided_blocks": [],
        # Deliberately not in execution order: the runner must honor `after`.
        "steps": [
            step("S002", "C002_application", "printf application", after=["S001"]),
            step("S001", "C001_database", "printf database", after=[]),
            step("S003", "C003_validation", "printf validation", after=["S002"]),
        ],
        "external_requirements": [],
        "validation_checks": [],
        "disagreements": [],
        "unresolved": [],
    }


def _write_workflow(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    (source / "deploy.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    workflow = tmp_path / "deployment-workflow.yaml"
    workflow.write_text(yaml.safe_dump(_workflow_payload(), sort_keys=False), encoding="utf-8")
    return workflow, source


def test_dry_run_orders_commands_without_executing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, source = _write_workflow(tmp_path)
    monkeypatch.setattr(
        "pheragent.deployment.execution.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("dry-run executed a subprocess"),
    )

    prepared = prepare_execution(workflow, {"fixture": source})
    rendered = prepared.render()

    assert [operation.step.id for operation in prepared.operations] == ["S001", "S002", "S003"]
    assert rendered.index("printf database") < rendered.index("printf application")
    assert "No commands were executed." in rendered
    assert f"Approval token: {prepared.approval_token}" in rendered


def test_execution_requires_the_exact_dry_run_approval(tmp_path: Path) -> None:
    workflow, source = _write_workflow(tmp_path)
    prepared = prepare_execution(workflow, {"fixture": source})
    calls: list[str] = []

    with pytest.raises(DeploymentInputError, match="approval token does not match"):
        prepared.execute(
            approval_token="sha256:not-the-plan",
            timeout=10,
            command_runner=lambda command, _cwd, _timeout: calls.append(command) or 0,
        )

    assert calls == []


def test_execution_stops_at_first_failed_operation(tmp_path: Path) -> None:
    workflow, source = _write_workflow(tmp_path)
    prepared = prepare_execution(workflow, {"fixture": source})
    calls: list[str] = []

    def runner(command: str, _cwd: Path, _timeout: float) -> int:
        calls.append(command)
        return 9 if command == "printf application" else 0

    with pytest.raises(WorkflowExecutionError, match="S002 failed with exit code 9"):
        prepared.execute(
            approval_token=prepared.approval_token,
            timeout=10,
            command_runner=runner,
        )

    assert calls == ["printf database", "printf application"]


def test_trial_mode_executes_only_ready_dependency_closed_steps(tmp_path: Path) -> None:
    payload = _workflow_payload()
    payload["ready_for_execution"] = False
    coverage = payload["coverage"]
    steps = payload["steps"]
    assert isinstance(coverage, dict)
    assert isinstance(steps, list)
    coverage["semantic_coverage_complete"] = False
    blocked = next(step for step in steps if step["id"] == "S002")
    blocked["status"] = "blocked"
    blocked["blockers"] = ["synthetic unresolved installer"]

    source = tmp_path / "source"
    source.mkdir()
    workflow = tmp_path / "unready-workflow.yaml"
    workflow.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    prepared = prepare_execution(
        workflow,
        {"fixture": source},
        allow_unready=True,
    )
    calls: list[str] = []
    completed = prepared.execute(
        approval_token=prepared.approval_token,
        timeout=10,
        command_runner=lambda command, _cwd, _timeout: calls.append(command) or 0,
    )

    assert prepared.executable is True
    assert [operation.step.id for operation in prepared.operations] == ["S001"]
    assert [step.id for step in prepared.excluded_steps] == ["S002", "S003"]
    assert completed == ("S001",)
    assert calls == ["printf database"]


def test_legacy_duplicate_stack_commands_are_rejected(tmp_path: Path) -> None:
    payload = _workflow_payload()
    steps = payload["steps"]
    assert isinstance(steps, list)
    first = steps[0]
    second = steps[1]
    assert isinstance(first, dict)
    assert isinstance(second, dict)
    first["command"] = "docker compose up"
    second["command"] = "docker compose up"
    for step in (first, second):
        target = step.pop("targets")[0]
        step["component_id"] = target["id"]
        step["component_name"] = target["name"]

    source = tmp_path / "source"
    source.mkdir()
    workflow = tmp_path / "legacy-workflow.yaml"
    workflow.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(WorkflowNotExecutableError, match="repeat one grounded operation"):
        prepare_execution(workflow, {"fixture": source})


def test_cli_defaults_to_dry_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, source = _write_workflow(tmp_path)
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "deployment",
            "run",
            str(workflow),
            "--source-root",
            f"fixture={source}",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Deployment dry-run: fixture" in captured.out
    assert "No commands were executed." in captured.out
    assert captured.err == ""
