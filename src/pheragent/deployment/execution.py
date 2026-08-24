from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .analysis_models import (
    DeploymentWorkflow,
    DeploymentWorkflowStep,
    FunctionalBlock,
    WorkflowStepStatus,
)
from .errors import (
    DeploymentInputError,
    WorkflowExecutionError,
    WorkflowNotExecutableError,
)
from .serialization import load_yaml
from .workflow import operation_fingerprint, ordered_workflow_steps

CommandRunner = Callable[[str, Path, float], int]
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class ExecutionOperation:
    """One command with its source-root-constrained working directory."""

    step: DeploymentWorkflowStep
    working_directory: Path


@dataclass(frozen=True, slots=True)
class PreparedExecution:
    """A validated, ordered workflow whose dry-run text is also its approval boundary."""

    workflow_path: Path
    workflow: DeploymentWorkflow
    source_roots: Mapping[str, Path]
    operations: tuple[ExecutionOperation, ...]
    excluded_steps: tuple[DeploymentWorkflowStep, ...]
    selected_block: FunctionalBlock | None
    allow_unready: bool
    approval_token: str

    @property
    def executable(self) -> bool:
        return bool(self.operations) and (
            self.allow_unready
            or (
                self.workflow.ready_for_execution
                and not self.excluded_steps
                and all(
                    operation.step.status == WorkflowStepStatus.READY
                    and operation.step.command is not None
                    for operation in self.operations
                )
            )
        )

    def render(self) -> str:
        lines = [
            f"Deployment dry-run: {self.workflow.system}",
            f"Workflow: {self.workflow_path}",
            f"Workflow ready: {str(self.workflow.ready_for_execution).lower()}",
            f"Trial override: {str(self.allow_unready).lower()}",
            f"Selected block: {_block_label(self.selected_block)}",
            f"Selected plan executable: {str(self.executable).lower()}",
            f"Ordered operations: {len(self.operations)}",
            "",
        ]
        operation_ids = {operation.step.id for operation in self.operations}
        for index, operation in enumerate(self.operations, start=1):
            step = operation.step
            targets = ", ".join(f"{target.name} ({target.id})" for target in step.targets)
            selected_dependencies = [item for item in step.after if item in operation_ids]
            omitted_dependencies = [item for item in step.after if item not in operation_ids]
            lines.extend(
                [
                    f"{index}. {step.id} [{step.status.value}] {targets}",
                    "   after in selected scope: "
                    + (", ".join(selected_dependencies) if selected_dependencies else "-"),
                    f"   executor: {step.executor.value}",
                    f"   source: {_source_label(step)}",
                    f"   working directory: {operation.working_directory}",
                    f"   command: {step.command or '<missing>'}",
                ]
            )
            if omitted_dependencies:
                lines.append(
                    "   cross-block workflow order omitted: " + ", ".join(omitted_dependencies)
                )
            if step.required_inputs:
                lines.append(f"   required inputs: {', '.join(step.required_inputs)}")
            if step.blockers:
                lines.append(f"   blockers: {'; '.join(step.blockers)}")
            lines.append("")
        if self.excluded_steps:
            lines.append("Excluded operations:")
            for step in self.excluded_steps:
                reason = "; ".join(step.blockers) or "depends on an excluded operation"
                lines.append(f"- {step.id}: {reason}")
            lines.append("")
        if self.executable:
            lines.extend(
                [
                    f"Approval token: {self.approval_token}",
                    "No commands were executed.",
                    "Run the same command with --execute and --approve TOKEN to execute.",
                ]
            )
        else:
            lines.extend(
                [
                    "Approval token: unavailable",
                    "No commands were executed.",
                    "Resolve the workflow blockers before execution.",
                ]
            )
        return "\n".join(lines) + "\n"

    def execute(
        self,
        *,
        approval_token: str,
        timeout: float,
        progress: ProgressCallback | None = None,
        command_runner: CommandRunner | None = None,
    ) -> tuple[str, ...]:
        """Execute sequentially and stop at the first failed operation."""
        if not self.executable:
            raise WorkflowNotExecutableError("deployment workflow is not ready for execution")
        if approval_token != self.approval_token:
            raise DeploymentInputError(
                "approval token does not match this workflow, command order, and source mapping"
            )
        notify = progress or (lambda _message: None)
        runner = command_runner or _run_command
        completed: list[str] = []
        total = len(self.operations)
        for index, operation in enumerate(self.operations, start=1):
            step = operation.step
            command = step.command
            if command is None:  # Protected by executable; keeps the boundary explicit.
                raise WorkflowNotExecutableError(f"workflow step {step.id} has no command")
            notify(f"[{index}/{total}] {step.id}: {command}")
            return_code = runner(command, operation.working_directory, timeout)
            if return_code != 0:
                raise WorkflowExecutionError(
                    f"workflow step {step.id} failed with exit code {return_code}; "
                    "execution stopped"
                )
            completed.append(step.id)
        return tuple(completed)


def prepare_execution(
    workflow_path: Path,
    source_roots: Mapping[str, Path],
    *,
    allow_unready: bool = False,
    block_id: str | None = None,
) -> PreparedExecution:
    """Resolve a full workflow or one functional block without executing commands."""
    resolved_workflow = workflow_path.expanduser().resolve(strict=True)
    workflow = DeploymentWorkflow.model_validate(load_yaml(resolved_workflow))
    roots = _resolve_source_roots(source_roots)
    ordered_steps = ordered_workflow_steps(workflow.steps)
    _reject_duplicate_operations(ordered_steps)
    selected_block = _load_selected_block(resolved_workflow, block_id)
    selected_steps, excluded_steps = _select_execution_steps(
        ordered_steps,
        allow_unready=allow_unready,
        selected_block=selected_block,
    )
    operations = tuple(
        ExecutionOperation(
            step=step,
            working_directory=_resolve_working_directory(step, roots),
        )
        for step in selected_steps
    )
    token = _approval_token(
        workflow,
        roots,
        operations,
        allow_unready=allow_unready,
        selected_block=selected_block,
    )
    return PreparedExecution(
        workflow_path=resolved_workflow,
        workflow=workflow,
        source_roots=roots,
        operations=operations,
        excluded_steps=tuple(excluded_steps),
        selected_block=selected_block,
        allow_unready=allow_unready,
        approval_token=token,
    )


def _load_selected_block(workflow_path: Path, block_id: str | None) -> FunctionalBlock | None:
    if block_id is None:
        return None
    artifact_path = workflow_path.with_name("functional-blocks.yaml")
    if not artifact_path.is_file():
        raise DeploymentInputError(
            f"--block requires the adjacent functional block artifact: {artifact_path}"
        )
    payload = load_yaml(artifact_path)
    raw_blocks = payload.get("blocks") if isinstance(payload, dict) else None
    if not isinstance(raw_blocks, list):
        raise DeploymentInputError(f"functional block artifact has no blocks: {artifact_path}")
    blocks = [FunctionalBlock.model_validate(item) for item in raw_blocks]
    block_by_id = {block.id: block for block in blocks}
    try:
        selected = block_by_id[block_id]
    except KeyError as exc:
        available = ", ".join(block.id for block in blocks if block.state != "provided") or "none"
        raise DeploymentInputError(
            f"unknown or unavailable block {block_id}; discovered blocks: {available}"
        ) from exc
    if selected.state == "provided":
        raise DeploymentInputError(f"block {block_id} is already provided and has nothing to run")
    unavailable = [
        dependency
        for dependency in selected.after
        if dependency not in block_by_id or block_by_id[dependency].state != "provided"
    ]
    if unavailable:
        raise WorkflowNotExecutableError(
            f"block {block_id} depends on non-provided blocks: {', '.join(unavailable)}"
        )
    return selected


def _resolve_source_roots(source_roots: Mapping[str, Path]) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for source_id, path in source_roots.items():
        if not source_id:
            raise DeploymentInputError("source root ID cannot be empty")
        root = path.expanduser().resolve(strict=True)
        if not root.is_dir():
            raise DeploymentInputError(f"source root is not a directory: {root}")
        resolved[source_id] = root
    return resolved


def _resolve_working_directory(
    step: DeploymentWorkflowStep,
    source_roots: Mapping[str, Path],
) -> Path:
    source_ref = step.operation_source_ref or step.source_ref
    try:
        source_root = source_roots[source_ref.repo_id]
    except KeyError as exc:
        raise DeploymentInputError(
            f"workflow step {step.id} requires --source-root {source_ref.repo_id}=PATH"
        ) from exc
    working_directory = (source_root / (step.working_directory or ".")).resolve()
    if not working_directory.is_relative_to(source_root):
        raise DeploymentInputError(
            f"workflow step {step.id} working directory escapes its source root"
        )
    if not working_directory.is_dir():
        raise DeploymentInputError(
            f"workflow step {step.id} working directory does not exist: {working_directory}"
        )
    return working_directory


def _select_execution_steps(
    steps: list[DeploymentWorkflowStep],
    *,
    allow_unready: bool,
    selected_block: FunctionalBlock | None,
) -> tuple[list[DeploymentWorkflowStep], list[DeploymentWorkflowStep]]:
    scoped_steps = _steps_for_block(steps, selected_block)
    if not allow_unready:
        return scoped_steps, []
    scoped_ids = {step.id for step in scoped_steps}
    selected: list[DeploymentWorkflowStep] = []
    selected_ids: set[str] = set()
    excluded: list[DeploymentWorkflowStep] = []
    for step in scoped_steps:
        runnable = (
            step.status == WorkflowStepStatus.READY
            and step.command is not None
            and (set(step.after) & scoped_ids) <= selected_ids
        )
        if runnable:
            selected.append(step)
            selected_ids.add(step.id)
        else:
            excluded.append(step)
    return selected, excluded


def _steps_for_block(
    steps: list[DeploymentWorkflowStep],
    selected_block: FunctionalBlock | None,
) -> list[DeploymentWorkflowStep]:
    if selected_block is None:
        return steps
    component_ids = {component.id for component in selected_block.components}
    selected: list[DeploymentWorkflowStep] = []
    for step in steps:
        target_ids = {target.id for target in step.targets}
        if not target_ids & component_ids:
            continue
        outside_block = sorted(target_ids - component_ids)
        if outside_block:
            raise WorkflowNotExecutableError(
                f"workflow step {step.id} spans block {selected_block.id} and other components: "
                + ", ".join(outside_block)
            )
        selected.append(step)
    if not selected:
        raise WorkflowNotExecutableError(
            f"block {selected_block.id} has no grounded workflow operations"
        )
    return selected


def _reject_duplicate_operations(steps: list[DeploymentWorkflowStep]) -> None:
    """Prevent legacy per-component workflows from repeating one shared stack command."""
    seen: dict[tuple[object, ...], str] = {}
    for step in steps:
        if step.command is None or step.action_id is not None:
            continue
        source = step.operation_source_ref or step.source_ref
        key = operation_fingerprint(
            executor=step.executor.value,
            command=step.command,
            working_directory=step.working_directory,
            source_ref=source,
        )
        if previous := seen.get(key):
            raise WorkflowNotExecutableError(
                f"workflow steps {previous} and {step.id} repeat one grounded operation; "
                "regenerate the workflow with command deduplication"
            )
        seen[key] = step.id


def _approval_token(
    workflow: DeploymentWorkflow,
    source_roots: Mapping[str, Path],
    operations: tuple[ExecutionOperation, ...],
    *,
    allow_unready: bool,
    selected_block: FunctionalBlock | None,
) -> str:
    payload = {
        "workflow": workflow.model_dump(mode="json", exclude_none=True),
        "source_roots": {source_id: str(path) for source_id, path in sorted(source_roots.items())},
        "execution_order": [operation.step.id for operation in operations],
        "allow_unready": allow_unready,
        "selected_block": selected_block.id if selected_block else None,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(serialized.encode()).hexdigest()


def _block_label(block: FunctionalBlock | None) -> str:
    return f"{block.id} ({block.name})" if block else "all"


def _source_label(step: DeploymentWorkflowStep) -> str:
    source = step.operation_source_ref or step.source_ref
    line = f":{source.start_line}" if source.start_line else ""
    return f"{source.repo_id}:{source.path}{line}"


def _run_command(command: str, working_directory: Path, timeout: float) -> int:
    try:
        completed = subprocess.run(
            ["bash", "-o", "pipefail", "-c", command],
            cwd=working_directory,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkflowExecutionError(f"command timed out after {timeout:g} seconds") from exc
    return completed.returncode
