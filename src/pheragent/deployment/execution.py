from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from pheragent.utils import slugify

from .analysis_models import (
    DeploymentWorkflow,
    DeploymentWorkflowStep,
    FunctionalBlock,
    FunctionalBlocksDocument,
    FunctionalComponent,
    FunctionalDeployRef,
    WorkflowStepKind,
    WorkflowStepStatus,
    WorkflowTarget,
)
from .errors import (
    DeploymentInputError,
    WorkflowExecutionError,
    WorkflowNotExecutableError,
)
from .graph import topological_levels
from .recovery import (
    PlanUpdateKind,
    RecoveryFailure,
    RecoveryResolution,
    RecoveryStatus,
    WorkflowPlanUpdate,
)
from .redaction import redact_secrets
from .serialization import load_yaml, write_json, write_text
from .workflow import operation_fingerprint, ordered_workflow_steps

ProgressCallback = Callable[[str], None]
_FAILURE_POLICY = "continue-independent"
_CAPTURE_LIMIT = 1_000_000
_FAILURE_EXCERPT_LIMIT = 8_000


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    return_code: int | None
    output: str
    duration_seconds: float
    timed_out: bool = False

    @classmethod
    def succeeded(cls, output: str = "", *, duration_seconds: float = 0.0) -> CommandOutcome:
        return cls(0, output, duration_seconds)

    @classmethod
    def failed(
        cls,
        return_code: int,
        output: str,
        *,
        duration_seconds: float = 0.0,
    ) -> CommandOutcome:
        return cls(return_code, output, duration_seconds)


CommandRunner = Callable[[str, Path, float], CommandOutcome | int]
RepairApproval = Callable[[RecoveryResolution], bool]
PatchPromoter = Callable[[str, str], Path]


class RecoveryQueue(Protocol):
    @property
    def has_pending(self) -> bool: ...

    def submit(self, failure: RecoveryFailure) -> None: ...

    def poll(self) -> RecoveryResolution | None: ...

    def wait(self) -> RecoveryResolution: ...


@dataclass(frozen=True, slots=True)
class ExecutionOperation:
    """One command with its source-root-constrained working directory."""

    step: DeploymentWorkflowStep
    working_directory: Path


@dataclass(frozen=True, slots=True)
class ExecutionIssue:
    """One operation that failed or could not run."""

    step_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class ExecutionAttempt:
    step_id: str
    attempt: int
    status: str
    exit_code: int | None
    timed_out: bool
    duration_seconds: float
    output_path: str | None = None


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """Final outcome after every runnable dependency branch has been attempted."""

    completed: tuple[str, ...]
    failed: tuple[ExecutionIssue, ...]
    skipped: tuple[ExecutionIssue, ...]
    attempts: tuple[ExecutionAttempt, ...] = ()
    recoveries: tuple[RecoveryResolution, ...] = ()
    failures_seen: tuple[RecoveryFailure, ...] = ()
    workflow: DeploymentWorkflow | None = None
    functional_blocks: FunctionalBlocksDocument | None = None

    @property
    def successful(self) -> bool:
        return not self.failed and not self.skipped


@dataclass(frozen=True, slots=True)
class PreparedExecution:
    """A validated, ordered workflow whose dry-run text is also its approval boundary."""

    workflow_path: Path
    workflow: DeploymentWorkflow
    source_roots: Mapping[str, Path]
    operations: tuple[ExecutionOperation, ...]
    excluded_steps: tuple[DeploymentWorkflowStep, ...]
    selected_block: FunctionalBlock | None
    functional_blocks: FunctionalBlocksDocument | None
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
            "Failure policy: continue independent operations; skip failed descendants",
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
        recovery_queue: RecoveryQueue | None = None,
        max_repair_attempts: int = 2,
        output_directory: Path | None = None,
        execution_roots: Mapping[str, Path] | None = None,
        promote_patch: PatchPromoter | None = None,
        approve_repair: RepairApproval | None = None,
    ) -> ExecutionReport:
        """Run ready work while failed branches recover within a fixed budget."""
        if not self.executable:
            raise WorkflowNotExecutableError("deployment workflow is not ready for execution")
        if approval_token != self.approval_token:
            raise DeploymentInputError(
                "approval token does not match this workflow, command order, and source mapping"
            )
        if max_repair_attempts < 1:
            raise DeploymentInputError("max repair attempts must be greater than zero")
        output = output_directory.expanduser().resolve() if output_directory else None
        if output:
            output.mkdir(parents=True, exist_ok=True)
        report = _execute_operations(
            self,
            runner=command_runner or _run_command,
            recovery_queue=recovery_queue,
            max_repair_attempts=max_repair_attempts,
            timeout=timeout,
            output_directory=output,
            notify=progress or (lambda _message: None),
            execution_roots=execution_roots,
            promote_patch=promote_patch,
            approve_repair=approve_repair,
        )
        if output:
            _write_execution_artifacts(output, report)
        return report


def _execute_operations(
    prepared: PreparedExecution,
    *,
    runner: CommandRunner,
    recovery_queue: RecoveryQueue | None,
    max_repair_attempts: int,
    timeout: float,
    output_directory: Path | None,
    notify: ProgressCallback,
    execution_roots: Mapping[str, Path] | None,
    promote_patch: PatchPromoter | None,
    approve_repair: RepairApproval | None,
) -> ExecutionReport:
    operations = {operation.step.id: operation for operation in prepared.operations}
    ordered = list(prepared.operations)
    workflow = prepared.workflow
    functional_blocks = prepared.functional_blocks
    states = dict.fromkeys(operations, "pending")
    active_roots = _execution_roots(prepared.source_roots, execution_roots)
    source_roots = {
        step_id: active_roots[_step_repo_id(operation.step)]
        for step_id, operation in operations.items()
    }
    working_directories = {
        step_id: source_roots[step_id]
        / operation.working_directory.relative_to(
            prepared.source_roots[_step_repo_id(operation.step)]
        )
        for step_id, operation in operations.items()
    }
    attempt_counts = dict.fromkeys(operations, 0)
    repair_counts = dict.fromkeys(operations, 0)
    latest_failures: dict[str, RecoveryFailure] = {}
    completed: list[str] = []
    failed: dict[str, ExecutionIssue] = {}
    skipped: dict[str, ExecutionIssue] = {}
    attempts: list[ExecutionAttempt] = []
    recoveries: list[RecoveryResolution] = []
    failures_seen: list[RecoveryFailure] = []
    pending_approvals: deque[RecoveryResolution] = deque()

    def report() -> ExecutionReport:
        return ExecutionReport(
            tuple(completed),
            tuple(failed.values()),
            tuple(skipped.values()),
            tuple(attempts),
            tuple(recoveries),
            tuple(failures_seen),
            workflow,
            _mark_deployed_blocks(functional_blocks, workflow, set(completed)),
        )

    def checkpoint() -> None:
        if output_directory:
            _write_execution_artifacts(output_directory, report())

    def accept_resolution(resolution: RecoveryResolution) -> None:
        nonlocal workflow, functional_blocks, ordered
        recoveries.append(resolution)
        step_id = resolution.step_id
        failure = latest_failures[step_id]
        if resolution.status == RecoveryStatus.RESOLVED:
            try:
                if resolution.patch:
                    if promote_patch is None:
                        raise ValueError("repair workspace is unavailable")
                    promote_patch(failure.repo_id, resolution.patch)
                if resolution.plan_update:
                    if not resolution.validation or not resolution.validation.succeeded:
                        raise ValueError("plan update has not passed bounded validation")
                    workflow, functional_blocks, added, revisit = _revise_workflow(
                        workflow,
                        functional_blocks,
                        resolution.plan_update,
                        failure,
                        selected_block=prepared.selected_block,
                    )
                    revised = {step.id: step for step in workflow.steps}
                    for current_id, operation in tuple(operations.items()):
                        operations[current_id] = replace(
                            operation,
                            step=revised[current_id],
                        )
                    if added:
                        root = active_roots[_step_repo_id(added)]
                        operation = ExecutionOperation(
                            added,
                            _resolve_working_directory(added, active_roots),
                        )
                        operations[added.id] = operation
                        states[added.id] = "pending"
                        source_roots[added.id] = root
                        working_directories[added.id] = operation.working_directory
                        attempt_counts[added.id] = 0
                        repair_counts[added.id] = 0
                    if revisit:
                        if revisit not in operations:
                            raise ValueError(
                                f"cannot revisit {revisit} outside the selected execution scope"
                            )
                        _reopen_completed_branch(revisit, operations, states, completed)
                    ordered = [
                        operations[item.id]
                        for item in ordered_workflow_steps(
                            [operation.step for operation in operations.values()]
                        )
                    ]
                    notify(
                        "recovery: "
                        f"{step_id}: workflow revised ({resolution.plan_update.kind.value})"
                    )
            except ValueError as exc:
                states[step_id] = "failed"
                failed[step_id] = ExecutionIssue(step_id, str(exc))
                notify(f"recovery: {step_id}: plan update rejected; {exc}")
                return
            states[step_id] = "pending"
            failed.pop(step_id, None)
            skipped.pop(step_id, None)
            notify(f"recovery: {step_id}: validated fix ready; resuming plan")
            return
        if (
            resolution.status == RecoveryStatus.NEEDS_HUMAN
            and (resolution.patch or resolution.plan_update)
            and resolution.validation
            and resolution.validation.succeeded
        ):
            states[step_id] = "awaiting_approval"
            pending_approvals.append(resolution)
            notify(f"recovery: {step_id}: repair is ready for approval")
            return
        if (
            resolution.status == RecoveryStatus.RETRYABLE
            and repair_counts[step_id] < max_repair_attempts
        ):
            assert recovery_queue is not None
            repair_counts[step_id] += 1
            retried = replace(
                failure,
                id=f"{step_id}-recovery-{repair_counts[step_id]}",
                history=(*failure.history, resolution.reason),
            )
            latest_failures[step_id] = retried
            recovery_queue.submit(retried)
            notify(
                f"recovery: {step_id}: trying another repair "
                f"({repair_counts[step_id]}/{max_repair_attempts})"
            )
            return
        states[step_id] = "failed"
        failed[step_id] = ExecutionIssue(step_id, resolution.reason)
        notify(f"recovery: {step_id}: {resolution.status.value}; {resolution.reason}")

    while True:
        while recovery_queue and (resolution := recovery_queue.poll()) is not None:
            accept_resolution(resolution)
            checkpoint()

        _skip_failed_descendants(operations, states, skipped, notify)
        ready = _next_ready_operation(tuple(ordered), states)
        if ready is not None:
            step = ready.step
            command = step.command
            if command is None:
                raise WorkflowNotExecutableError(f"workflow step {step.id} has no command")
            attempt_counts[step.id] += 1
            attempt = attempt_counts[step.id]
            label = f"[{list(operations).index(step.id) + 1}/{len(operations)}] {step.id}"
            suffix = f" (attempt {attempt})" if attempt > 1 else ""
            notify(f"{label}{suffix}: {command}")
            outcome = _run_attempt(runner, command, working_directories[step.id], timeout)
            log_path = _write_attempt_log(output_directory, step.id, attempt, outcome.output)
            attempts.append(_attempt_record(step.id, attempt, outcome, log_path))
            if outcome.return_code == 0 and not outcome.timed_out:
                states[step.id] = "completed"
                completed.append(step.id)
                failed.pop(step.id, None)
                notify(f"{label}{suffix}: completed")
                checkpoint()
                continue
            failure = _build_failure(
                prepared,
                ready,
                workflow=workflow,
                functional_blocks=functional_blocks,
                source_root=source_roots[step.id],
                working_directory=working_directories[step.id],
                attempt=attempt,
                outcome=outcome,
                history=tuple(
                    resolution.reason
                    for resolution in recoveries
                    if resolution.step_id == step.id
                ),
            )
            failures_seen.append(failure)
            latest_failures[step.id] = failure
            reason = _failure_reason(outcome)
            notify(f"{label}{suffix}: failed; {reason}")
            if recovery_queue and repair_counts[step.id] < max_repair_attempts:
                repair_counts[step.id] += 1
                states[step.id] = "recovering"
                recovery_queue.submit(failure)
                notify(
                    f"{label}: queued for recovery "
                    f"({repair_counts[step.id]}/{max_repair_attempts})"
                )
            else:
                states[step.id] = "failed"
                failed[step.id] = ExecutionIssue(step.id, reason)
            checkpoint()
            continue

        if recovery_queue and recovery_queue.has_pending:
            resolution = recovery_queue.wait()
            accept_resolution(resolution)
            checkpoint()
            continue
        if pending_approvals:
            proposed = pending_approvals.popleft()
            if approve_repair and approve_repair(proposed):
                # Replace the pending decision so usage and repair metrics count one
                # recovery attempt, not the approval transition as a second attempt.
                recoveries.remove(proposed)
                approved = replace(
                    proposed,
                    status=RecoveryStatus.RESOLVED,
                    reason=f"human approved: {proposed.reason}",
                )
                accept_resolution(approved)
            else:
                states[proposed.step_id] = "failed"
                failed[proposed.step_id] = ExecutionIssue(
                    proposed.step_id,
                    "repair was not approved",
                )
                notify(f"recovery: {proposed.step_id}: repair was not approved")
            checkpoint()
            continue
        if all(state in {"completed", "failed", "skipped"} for state in states.values()):
            break
        _skip_unreachable_operations(states, skipped, notify)
        checkpoint()
        break

    return report()


def _execution_roots(
    source_roots: Mapping[str, Path],
    replacements: Mapping[str, Path] | None,
) -> dict[str, Path]:
    if replacements is None:
        return dict(source_roots)
    missing = set(source_roots) - set(replacements)
    if missing:
        raise DeploymentInputError(
            "execution roots are missing source IDs: " + ", ".join(sorted(missing))
        )
    return {
        source_id: replacements[source_id].expanduser().resolve(strict=True)
        for source_id in source_roots
    }


def _revise_workflow(
    workflow: DeploymentWorkflow,
    blocks: FunctionalBlocksDocument | None,
    update: WorkflowPlanUpdate,
    failure: RecoveryFailure,
    *,
    selected_block: FunctionalBlock | None,
) -> tuple[
    DeploymentWorkflow,
    FunctionalBlocksDocument | None,
    DeploymentWorkflowStep | None,
    str | None,
]:
    steps = {step.id: step for step in workflow.steps}
    try:
        target = steps[failure.step_id]
    except KeyError as exc:
        raise ValueError(f"failed step is no longer in the workflow: {failure.step_id}") from exc
    if update.kind == PlanUpdateKind.REVISIT:
        prerequisite = update.prerequisite_step_id
        if prerequisite not in steps or prerequisite == target.id:
            raise ValueError("plan update names an invalid prerequisite step")
        revised_target = target.model_copy(
            update={"after": list(dict.fromkeys([*target.after, prerequisite]))}
        )
        revised = workflow.model_copy(
            update={
                "steps": [
                    revised_target if step.id == target.id else step
                    for step in workflow.steps
                ]
            }
        )
        return (
            DeploymentWorkflow.model_validate(revised.model_dump(mode="python")),
            blocks,
            None,
            prerequisite,
        )

    assert update.executor is not None
    assert update.source_ref is not None
    assert update.command is not None
    if selected_block and update.block_id and update.block_id != selected_block.id:
        raise ValueError("plan update cannot leave the selected block")
    next_step_id = _next_identifier(steps, "S")
    target_ids = target.targets
    revised_blocks = blocks
    if update.component_name:
        if blocks is None:
            raise ValueError("a new component requires functional-blocks.yaml")
        block_id = update.block_id or failure.block_id
        block_by_id = {block.id: block for block in blocks.blocks}
        if block_id not in block_by_id or block_by_id[block_id].state == "provided":
            raise ValueError("new component must belong to an existing discovered block")
        component_id = _next_identifier(
            (component.id for block in blocks.blocks for component in block.components),
            "C",
            slug=slugify(update.component_name),
        )
        component = FunctionalComponent(
            id=component_id,
            name=update.component_name,
            implementation=update.component_name,
            deployable=True,
            external=False,
            deploy=FunctionalDeployRef(
                executor=update.executor,
                ref=update.source_ref.path,
                repo_id=update.source_ref.repo_id,
                required_inputs=update.required_inputs,
            ),
        )
        revised_block_values = []
        for block in blocks.blocks:
            changes: dict[str, Any] = {}
            if block.id == block_id:
                changes["components"] = [*block.components, component]
            if (
                failure.block_id
                and block.id == failure.block_id
                and block_id != failure.block_id
            ):
                changes["after"] = list(dict.fromkeys([*block.after, block_id]))
            revised_block_values.append(block.model_copy(update=changes) if changes else block)
        dependencies = {block.id: set(block.after) for block in revised_block_values}
        revised_blocks = blocks.model_copy(
            update={
                "blocks": revised_block_values,
                "levels": topological_levels(
                    [block.id for block in revised_block_values],
                    dependencies,
                    cycle_label="functional block plan update",
                ),
            }
        )
        revised_blocks = FunctionalBlocksDocument.model_validate(
            revised_blocks.model_dump(mode="python")
        )
        target_ids = [WorkflowTarget(id=component.id, name=component.name)]
    added = DeploymentWorkflowStep(
        id=next_step_id,
        kind=WorkflowStepKind.COMPONENT,
        targets=target_ids,
        executor=update.executor,
        source_ref=update.source_ref,
        working_directory=(
            update.working_directory or Path(update.source_ref.path).parent.as_posix()
        ),
        command=update.command,
        required_inputs=update.required_inputs,
        after=target.after,
        status=WorkflowStepStatus.READY,
    )
    revised_target = target.model_copy(update={"after": [added.id]})
    revised = workflow.model_copy(
        update={
            "steps": [
                revised_target if step.id == target.id else step for step in workflow.steps
            ]
            + [added]
        }
    )
    return (
        DeploymentWorkflow.model_validate(revised.model_dump(mode="python")),
        revised_blocks,
        added,
        None,
    )


def _next_identifier(
    identifiers: Iterable[str],
    prefix: str,
    *,
    slug: str | None = None,
) -> str:
    largest = max(
        (
            int(identifier[1:].split("_", 1)[0])
            for identifier in identifiers
            if isinstance(identifier, str) and identifier.startswith(prefix)
        ),
        default=0,
    )
    base = f"{prefix}{largest + 1:03d}"
    return f"{base}_{slug or 'component'}" if prefix == "C" else base


def _step_repo_id(step: DeploymentWorkflowStep) -> str:
    return (step.operation_source_ref or step.source_ref).repo_id


def _skip_failed_descendants(
    operations: dict[str, ExecutionOperation],
    states: dict[str, str],
    skipped: dict[str, ExecutionIssue],
    notify: ProgressCallback,
) -> None:
    changed = True
    while changed:
        changed = False
        for step_id, operation in operations.items():
            if states[step_id] != "pending":
                continue
            blocked_by = [
                dependency
                for dependency in operation.step.after
                if states.get(dependency) in {"failed", "skipped"}
            ]
            if not blocked_by:
                continue
            reason = "unsuccessful prerequisite(s): " + ", ".join(blocked_by)
            states[step_id] = "skipped"
            skipped[step_id] = ExecutionIssue(step_id, reason)
            notify(f"{step_id}: skipped; {reason}")
            changed = True


def _reopen_completed_branch(
    root: str,
    operations: Mapping[str, ExecutionOperation],
    states: dict[str, str],
    completed: list[str],
) -> None:
    """Invalidate only completed work downstream of a revisited prerequisite."""
    reopened = {root}
    changed = True
    while changed:
        changed = False
        for step_id, operation in operations.items():
            if step_id not in reopened and any(
                dependency in reopened for dependency in operation.step.after
            ):
                reopened.add(step_id)
                changed = True
    for step_id in reopened:
        if states[step_id] == "completed":
            states[step_id] = "pending"
            completed.remove(step_id)


def _next_ready_operation(
    ordered: tuple[ExecutionOperation, ...],
    states: dict[str, str],
) -> ExecutionOperation | None:
    for operation in ordered:
        if states[operation.step.id] != "pending":
            continue
        dependencies = [item for item in operation.step.after if item in states]
        if all(states[item] == "completed" for item in dependencies):
            return operation
    return None


def _skip_unreachable_operations(
    states: dict[str, str],
    skipped: dict[str, ExecutionIssue],
    notify: ProgressCallback,
) -> None:
    for step_id, state in states.items():
        if state != "pending":
            continue
        reason = "deployment made no progress"
        states[step_id] = "skipped"
        skipped[step_id] = ExecutionIssue(step_id, reason)
        notify(f"{step_id}: skipped; {reason}")


def _run_attempt(
    runner: CommandRunner,
    command: str,
    working_directory: Path,
    timeout: float,
) -> CommandOutcome:
    started = time.monotonic()
    try:
        result = runner(command, working_directory, timeout)
    except WorkflowExecutionError as exc:
        return CommandOutcome(None, str(exc), time.monotonic() - started, "timed out" in str(exc))
    if isinstance(result, CommandOutcome):
        return result
    return CommandOutcome(result, "", time.monotonic() - started)


def _build_failure(
    prepared: PreparedExecution,
    operation: ExecutionOperation,
    *,
    workflow: DeploymentWorkflow,
    functional_blocks: FunctionalBlocksDocument | None,
    source_root: Path,
    working_directory: Path,
    attempt: int,
    outcome: CommandOutcome,
    history: tuple[str, ...],
) -> RecoveryFailure:
    step = operation.step
    source = step.operation_source_ref or step.source_ref
    block_id = _component_block_id(functional_blocks, step)
    return RecoveryFailure(
        id=f"{step.id}-attempt-{attempt}",
        step_id=step.id,
        block_id=prepared.selected_block.id if prepared.selected_block else block_id,
        component_ids=tuple(target.id for target in step.targets),
        executor=step.executor.value,
        command=step.command or "",
        repo_id=source.repo_id,
        source_path=source.path,
        source_root=source_root,
        working_directory=working_directory,
        attempt=attempt,
        exit_code=outcome.return_code,
        timed_out=outcome.timed_out,
        duration_seconds=round(outcome.duration_seconds, 6),
        output_excerpt=_compact_output(redact_secrets(outcome.output), _FAILURE_EXCERPT_LIMIT),
        history=history,
        plan_context=_plan_context(workflow, functional_blocks, step, block_id),
    )


def _component_block_id(
    document: FunctionalBlocksDocument | None,
    step: DeploymentWorkflowStep,
) -> str | None:
    target_ids = {target.id for target in step.targets}
    if document:
        for block in document.blocks:
            if target_ids & {component.id for component in block.components}:
                return block.id
    return None


def _mark_deployed_blocks(
    document: FunctionalBlocksDocument | None,
    workflow: DeploymentWorkflow,
    completed: set[str],
) -> FunctionalBlocksDocument | None:
    if document is None:
        return None
    revised = []
    for block in document.blocks:
        component_ids = {
            component.id for component in block.components if component.deployable
        }
        steps = [
            step
            for step in workflow.steps
            if component_ids & {target.id for target in step.targets}
        ]
        covered = {target.id for step in steps for target in step.targets}
        deployed = component_ids and component_ids <= covered and all(
            step.id in completed for step in steps
        )
        revised.append(
            block.model_copy(update={"state": "deployed"})
            if block.state != "provided" and deployed
            else block
        )
    return document.model_copy(update={"blocks": revised})


def _plan_context(
    workflow: DeploymentWorkflow,
    document: FunctionalBlocksDocument | None,
    failed_step: DeploymentWorkflowStep,
    block_id: str | None,
) -> dict[str, Any]:
    by_id = {step.id: step for step in workflow.steps}
    relevant = {failed_step.id}
    frontier = list(failed_step.after)
    while frontier and len(relevant) < 20:
        step_id = frontier.pop()
        if step_id in relevant or step_id not in by_id:
            continue
        relevant.add(step_id)
        frontier.extend(by_id[step_id].after)
    return {
        "failed_block": block_id,
        "steps": [
            {
                "id": step.id,
                "components": [target.name for target in step.targets],
                "block": _component_block_id(document, step),
                "after": step.after,
                "status": step.status.value,
            }
            for step in workflow.steps
            if step.id in relevant
        ],
        "blocks": [
            {
                "id": block.id,
                "name": block.name,
                "state": block.state,
                "after": block.after,
            }
            for block in (document.blocks if document else [])
        ],
    }


def _failure_reason(outcome: CommandOutcome) -> str:
    if outcome.timed_out:
        return "command timed out"
    return f"exit code {outcome.return_code}"


def _write_attempt_log(
    output_directory: Path | None,
    step_id: str,
    attempt: int,
    output: str,
) -> str | None:
    if output_directory is None:
        return None
    relative = Path("logs") / f"{step_id}-attempt-{attempt}.log"
    write_text(output_directory / relative, redact_secrets(_compact_output(output, _CAPTURE_LIMIT)))
    return relative.as_posix()


def _attempt_record(
    step_id: str,
    attempt: int,
    outcome: CommandOutcome,
    output_path: str | None,
) -> ExecutionAttempt:
    return ExecutionAttempt(
        step_id=step_id,
        attempt=attempt,
        status="completed" if outcome.return_code == 0 and not outcome.timed_out else "failed",
        exit_code=outcome.return_code,
        timed_out=outcome.timed_out,
        duration_seconds=round(outcome.duration_seconds, 6),
        output_path=output_path,
    )


def _write_execution_artifacts(output: Path, report: ExecutionReport) -> None:
    write_json(
        output / "execution.json",
        {
            "completed": report.completed,
            "failed": [asdict(issue) for issue in report.failed],
            "skipped": [asdict(issue) for issue in report.skipped],
            "attempts": [asdict(attempt) for attempt in report.attempts],
            "recoveries": [_resolution_payload(item) for item in report.recoveries],
            "workflow": (
                report.workflow.model_dump(mode="json", exclude_none=True)
                if report.workflow
                else None
            ),
            "functional_blocks": (
                report.functional_blocks.model_dump(mode="json", exclude_none=True)
                if report.functional_blocks
                else None
            ),
        },
    )
    write_json(
        output / "failure-bundle.json",
        {"failures": [_failure_payload(item) for item in report.failures_seen]},
    )


def _failure_payload(failure: RecoveryFailure) -> dict[str, Any]:
    return {
        "id": failure.id,
        "step_id": failure.step_id,
        "block_id": failure.block_id,
        "component_ids": failure.component_ids,
        "executor": failure.executor,
        "command": redact_secrets(failure.command),
        "repo_id": failure.repo_id,
        "source_path": failure.source_path,
        "working_directory": failure.working_directory.relative_to(failure.source_root).as_posix(),
        "attempt": failure.attempt,
        "exit_code": failure.exit_code,
        "timed_out": failure.timed_out,
        "duration_seconds": failure.duration_seconds,
        "output_excerpt": failure.output_excerpt,
        "history": failure.history,
        "plan_context": failure.plan_context,
    }


def _resolution_payload(resolution: RecoveryResolution) -> dict[str, Any]:
    return {
        "failure_id": resolution.failure_id,
        "step_id": resolution.step_id,
        "status": resolution.status.value,
        "reason": resolution.reason,
        "failure_kind": resolution.failure_kind.value,
        "patch": resolution.patch,
        "plan_update": (
            resolution.plan_update.model_dump(mode="json", exclude_none=True)
            if resolution.plan_update
            else None
        ),
        "approval_items": resolution.approval_items,
        "probes": [
            {
                "request": probe.request.model_dump(mode="json", exclude_none=True),
                "succeeded": probe.succeeded,
                "output": probe.output,
                "error": probe.error,
            }
            for probe in resolution.probes
        ],
        "usage": resolution.usage,
        "validation": (
            {
                "succeeded": resolution.validation.succeeded,
                "checks": resolution.validation.checks,
                "error": resolution.validation.error,
            }
            if resolution.validation
            else None
        ),
        "model": resolution.model,
        "llm_calls": resolution.llm_calls,
    }


def _compact_output(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = min(1_000, limit // 4)
    marker = "\n...[truncated]...\n"
    return text[:head] + marker + text[-(limit - head - len(marker)) :]


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
    functional_blocks = _load_functional_blocks(resolved_workflow)
    selected_block = _select_block(functional_blocks, block_id)
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
        functional_blocks=functional_blocks,
        allow_unready=allow_unready,
        approval_token=token,
    )


def _load_functional_blocks(workflow_path: Path) -> FunctionalBlocksDocument | None:
    artifact_path = workflow_path.with_name("functional-blocks.yaml")
    if not artifact_path.is_file():
        return None
    return FunctionalBlocksDocument.model_validate(load_yaml(artifact_path))


def _select_block(
    document: FunctionalBlocksDocument | None,
    block_id: str | None,
) -> FunctionalBlock | None:
    if block_id is None:
        return None
    if document is None:
        raise DeploymentInputError(
            "--block requires functional-blocks.yaml beside the deployment workflow"
        )
    block_by_id = {block.id: block for block in document.blocks}
    try:
        selected = block_by_id[block_id]
    except KeyError as exc:
        available = (
            ", ".join(block.id for block in document.blocks if block.state != "provided") or "none"
        )
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
        "failure_policy": _FAILURE_POLICY,
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


def _run_command(command: str, working_directory: Path, timeout: float) -> CommandOutcome:
    started = time.monotonic()
    process = subprocess.Popen(
        ["bash", "-o", "pipefail", "-c", command],
        cwd=working_directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    captured = _OutputTail(_CAPTURE_LIMIT)

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            captured.append(line)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    timed_out = False
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            return_code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            return_code = process.wait()
    reader.join(timeout=5)
    return CommandOutcome(
        None if timed_out else return_code,
        captured.value,
        time.monotonic() - started,
        timed_out,
    )


class _OutputTail:
    """Keep a small beginning and bounded tail while output streams to the operator."""

    def __init__(self, limit: int) -> None:
        self._head_limit = min(32_000, limit // 4)
        self._tail_limit = limit - self._head_limit
        self._head = ""
        self._tail: deque[str] = deque()
        self._tail_size = 0
        self._truncated = False

    def append(self, text: str) -> None:
        if len(self._head) < self._head_limit:
            remaining = self._head_limit - len(self._head)
            self._head += text[:remaining]
            text = text[remaining:]
        if not text:
            return
        self._tail.append(text)
        self._tail_size += len(text)
        while self._tail_size > self._tail_limit and self._tail:
            removed = self._tail.popleft()
            self._tail_size -= len(removed)
            self._truncated = True

    @property
    def value(self) -> str:
        tail = "".join(self._tail)
        if not self._truncated:
            return self._head + tail
        return self._head + "\n...[output truncated]...\n" + tail
