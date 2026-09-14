from __future__ import annotations

import json
from pathlib import Path
from threading import Event

import pytest
import yaml

from pheragent.cli import main
from pheragent.deployment.errors import (
    DeploymentInputError,
    WorkflowNotExecutableError,
)
from pheragent.deployment.execution import CommandOutcome, prepare_execution
from pheragent.deployment.recovery import (
    PatchValidation,
    RecoveryResolution,
    RecoveryStatus,
    RunWorkspace,
    ThreadedRecoveryQueue,
)


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


def _write_functional_blocks(tmp_path: Path) -> None:
    payload = {
        "blocks": [
            {
                "id": "B0",
                "name": "Runtime",
                "type": "runtime_environment",
                "subtype": "container_platform",
                "state": "provided",
                "components": [],
            },
            {
                "id": "B1",
                "name": "Data Services",
                "type": "shared_services",
                "subtype": "data_services",
                "after": ["B0"],
                "components": [
                    {
                        "id": "C001_database",
                        "name": "Database",
                        "deployable": True,
                        "external": False,
                    }
                ],
            },
            {
                "id": "B2",
                "name": "Application",
                "type": "application",
                "subtype": "core_application",
                "after": ["B1"],
                "components": [
                    {
                        "id": "C002_application",
                        "name": "Application",
                        "deployable": True,
                        "external": False,
                    },
                    {
                        "id": "C003_validation",
                        "name": "Validation",
                        "deployable": True,
                        "external": False,
                    },
                ],
            },
        ]
    }
    (tmp_path / "functional-blocks.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


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


def test_execution_skips_failed_descendants_and_continues_independent_branch(
    tmp_path: Path,
) -> None:
    payload = _workflow_payload()
    steps = payload["steps"]
    assert isinstance(steps, list)
    steps.extend(
        [
            {
                "id": "S004",
                "kind": "component",
                "targets": [{"id": "C004_cache", "name": "Cache"}],
                "executor": "shell",
                "source_ref": {"repo_id": "fixture", "path": "deploy.sh"},
                "working_directory": ".",
                "command": "printf cache",
                "required_inputs": [],
                "after": [],
                "status": "ready",
                "blockers": [],
            }
        ]
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "deploy.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    workflow = tmp_path / "deployment-workflow.yaml"
    workflow.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    prepared = prepare_execution(workflow, {"fixture": source})
    calls: list[str] = []

    def runner(command: str, _cwd: Path, _timeout: float) -> int:
        calls.append(command)
        return 9 if command == "printf application" else 0

    report = prepared.execute(
        approval_token=prepared.approval_token,
        timeout=10,
        command_runner=runner,
    )

    assert calls == ["printf database", "printf cache", "printf application"]
    assert report.completed == ("S001", "S004")
    assert [(issue.step_id, issue.reason) for issue in report.failed] == [("S002", "exit code 9")]
    assert [(issue.step_id, issue.reason) for issue in report.skipped] == [
        ("S003", "unsuccessful prerequisite(s): S002")
    ]
    assert report.successful is False


def test_execution_continues_a_successful_chain_after_an_independent_failure(
    tmp_path: Path,
) -> None:
    payload = _workflow_payload()
    steps = payload["steps"]
    assert isinstance(steps, list)
    by_id = {step["id"]: step for step in steps}
    by_id["S001"]["command"] = "install postgres"
    by_id["S002"]["command"] = "install minio"
    by_id["S002"]["after"] = []
    by_id["S003"]["command"] = "configure minio credentials"
    steps[:] = [by_id["S001"], by_id["S002"], by_id["S003"]]

    source = tmp_path / "source"
    source.mkdir()
    workflow = tmp_path / "deployment-workflow.yaml"
    workflow.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    prepared = prepare_execution(workflow, {"fixture": source})
    calls: list[str] = []

    def runner(command: str, _cwd: Path, _timeout: float) -> int:
        calls.append(command)
        return 1 if command == "install postgres" else 0

    report = prepared.execute(
        approval_token=prepared.approval_token,
        timeout=10,
        command_runner=runner,
    )

    assert calls == ["install postgres", "install minio", "configure minio credentials"]
    assert report.completed == ("S002", "S003")
    assert [issue.step_id for issue in report.failed] == ["S001"]
    assert report.skipped == ()


def test_execution_checkpoints_each_completed_step(tmp_path: Path) -> None:
    workflow, source = _write_workflow(tmp_path)
    prepared = prepare_execution(workflow, {"fixture": source})
    output = tmp_path / "execution"

    def runner(command: str, _cwd: Path, _timeout: float) -> int:
        if command == "printf application":
            checkpoint = json.loads((output / "execution.json").read_text(encoding="utf-8"))
            assert checkpoint["completed"] == ["S001"]
        return 0

    prepared.execute(
        approval_token=prepared.approval_token,
        timeout=10,
        output_directory=output,
        command_runner=runner,
    )


def test_execution_repairs_failure_while_independent_work_continues(tmp_path: Path) -> None:
    payload = _workflow_payload()
    steps = payload["steps"]
    assert isinstance(steps, list)
    by_id = {step["id"]: step for step in steps}
    by_id["S001"]["command"] = "install database"
    by_id["S002"]["command"] = "configure application"
    by_id["S003"]["command"] = "validate application"
    steps.append(
        {
            "id": "S004",
            "kind": "component",
            "targets": [{"id": "C004_cache", "name": "Cache"}],
            "executor": "shell",
            "source_ref": {"repo_id": "fixture", "path": "deploy.sh"},
            "working_directory": ".",
            "command": "install cache",
            "required_inputs": [],
            "after": [],
            "status": "ready",
            "blockers": [],
        }
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "deploy.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    workflow = tmp_path / "deployment-workflow.yaml"
    workflow.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    prepared = prepare_execution(workflow, {"fixture": source})
    recovery_started = Event()
    release_recovery = Event()
    calls: list[str] = []

    def resolve(failure):
        recovery_started.set()
        assert release_recovery.wait(timeout=2)
        return RecoveryResolution(
            failure_id=failure.id,
            step_id=failure.step_id,
            status=RecoveryStatus.RESOLVED,
            reason="sandbox validation passed",
        )

    def runner(command: str, _cwd: Path, _timeout: float) -> CommandOutcome:
        calls.append(command)
        if command == "install database" and calls.count(command) == 1:
            return CommandOutcome.failed(1, "repo bitnami not found")
        if command == "install cache":
            assert recovery_started.wait(timeout=2)
            release_recovery.set()
        return CommandOutcome.succeeded("ok")

    with ThreadedRecoveryQueue(resolve, workers=2) as recovery_queue:
        report = prepared.execute(
            approval_token=prepared.approval_token,
            timeout=10,
            command_runner=runner,
            recovery_queue=recovery_queue,
            max_repair_attempts=2,
        )

    assert calls == [
        "install cache",
        "install database",
        "install database",
        "configure application",
        "validate application",
    ] or calls == [
        "install database",
        "install cache",
        "install database",
        "configure application",
        "validate application",
    ]
    assert set(report.completed) == {"S001", "S002", "S003", "S004"}
    assert report.failed == ()
    assert report.skipped == ()
    assert len([attempt for attempt in report.attempts if attempt.step_id == "S001"]) == 2


def test_approved_patch_is_shared_with_retry_and_later_steps(tmp_path: Path) -> None:
    workflow, source = _write_workflow(tmp_path)
    prepared = prepare_execution(workflow, {"fixture": source})
    patch = """\
--- a/deploy.sh
+++ b/deploy.sh
@@ -1 +1,2 @@
 #!/bin/bash
+# repaired
"""
    attempts = 0

    def resolve(failure):
        return RecoveryResolution(
            failure_id=failure.id,
            step_id=failure.step_id,
            status=RecoveryStatus.NEEDS_HUMAN,
            reason="review the exact source patch",
            patch=patch,
            approval_items=("image: docker.io/example/minio@sha256:123",),
            validation=PatchValidation(True, ("git.apply", "shell.syntax")),
        )

    def runner(command: str, cwd: Path, _timeout: float) -> CommandOutcome:
        nonlocal attempts
        if command == "printf database":
            attempts += 1
            if attempts == 1:
                return CommandOutcome.failed(1, "image approval required")
        if attempts > 1:
            assert "# repaired" in cwd.joinpath("deploy.sh").read_text(encoding="utf-8")
        return CommandOutcome.succeeded()

    with (
        RunWorkspace({"fixture": source}, tmp_path / "workspace") as workspace,
        ThreadedRecoveryQueue(resolve, workers=1) as recovery_queue,
    ):
        report = prepared.execute(
            approval_token=prepared.approval_token,
            timeout=10,
            command_runner=runner,
            recovery_queue=recovery_queue,
            execution_roots=workspace.roots,
            promote_patch=workspace.apply,
            approve_repair=lambda resolution: resolution.approval_items == (
                "image: docker.io/example/minio@sha256:123",
            ),
        )

    assert report.successful
    assert attempts == 2
    assert "# repaired" not in source.joinpath("deploy.sh").read_text(encoding="utf-8")


def test_execution_keeps_exhausted_failure_and_skips_only_its_descendants(
    tmp_path: Path,
) -> None:
    payload = _workflow_payload()
    steps = payload["steps"]
    assert isinstance(steps, list)
    steps.append(
        {
            "id": "S004",
            "kind": "component",
            "targets": [{"id": "C004_cache", "name": "Cache"}],
            "executor": "shell",
            "source_ref": {"repo_id": "fixture", "path": "deploy.sh"},
            "working_directory": ".",
            "command": "install cache",
            "required_inputs": [],
            "after": [],
            "status": "ready",
            "blockers": [],
        },
    )
    source = tmp_path / "source"
    source.mkdir()
    workflow = tmp_path / "deployment-workflow.yaml"
    workflow.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    prepared = prepare_execution(workflow, {"fixture": source})

    def resolve(failure):
        return RecoveryResolution(
            failure_id=failure.id,
            step_id=failure.step_id,
            status=RecoveryStatus.EXHAUSTED,
            reason="repair budget exhausted",
        )

    with ThreadedRecoveryQueue(resolve, workers=2) as recovery_queue:
        report = prepared.execute(
            approval_token=prepared.approval_token,
            timeout=10,
            command_runner=lambda command, _cwd, _timeout: (
                CommandOutcome.failed(1, "still broken")
                if command == "printf database"
                else CommandOutcome.succeeded("ok")
            ),
            recovery_queue=recovery_queue,
            max_repair_attempts=1,
        )

    assert report.completed == ("S004",)
    assert [issue.step_id for issue in report.failed] == ["S001"]
    assert {issue.step_id for issue in report.skipped} == {"S002", "S003"}


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
    report = prepared.execute(
        approval_token=prepared.approval_token,
        timeout=10,
        command_runner=lambda command, _cwd, _timeout: calls.append(command) or 0,
    )

    assert prepared.executable is True
    assert [operation.step.id for operation in prepared.operations] == ["S001"]
    assert [step.id for step in prepared.excluded_steps] == ["S002", "S003"]
    assert report.completed == ("S001",)
    assert report.successful is True
    assert calls == ["printf database"]


def test_trial_mode_can_select_one_block_after_provided_blocks(tmp_path: Path) -> None:
    workflow, source = _write_workflow(tmp_path)
    _write_functional_blocks(tmp_path)

    prepared = prepare_execution(
        workflow,
        {"fixture": source},
        allow_unready=True,
        block_id="B1",
    )

    assert prepared.selected_block is not None
    assert prepared.selected_block.id == "B1"
    assert [operation.step.id for operation in prepared.operations] == ["S001"]
    assert prepared.excluded_steps == ()
    assert "Selected block: B1 (Data Services)" in prepared.render()

    blocks_path = tmp_path / "functional-blocks.yaml"
    blocks_payload = yaml.safe_load(blocks_path.read_text(encoding="utf-8"))
    blocks_payload["blocks"][2]["after"] = ["B0"]
    blocks_path.write_text(yaml.safe_dump(blocks_payload, sort_keys=False), encoding="utf-8")
    application_block = prepare_execution(
        workflow,
        {"fixture": source},
        allow_unready=True,
        block_id="B2",
    )
    assert "cross-block workflow order omitted: S001" in application_block.render()
    assert "after in selected scope: S002" in application_block.render()


def test_block_selection_rejects_non_provided_prerequisites(tmp_path: Path) -> None:
    workflow, source = _write_workflow(tmp_path)
    _write_functional_blocks(tmp_path)

    with pytest.raises(WorkflowNotExecutableError, match="non-provided blocks: B1"):
        prepare_execution(
            workflow,
            {"fixture": source},
            allow_unready=True,
            block_id="B2",
        )


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
