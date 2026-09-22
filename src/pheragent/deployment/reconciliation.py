from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from pheragent.utils import slugify

from .analysis_models import (
    AnalysisExecutor,
    AnalysisSourceRef,
    DeploymentContext,
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
from .graph import topological_levels
from .models import ContractModel
from .runtime_context import RuntimeContextSnapshot


class CapabilityStatus(StrEnum):
    SATISFIED = "satisfied"
    MISSING = "missing"
    UNKNOWN = "unknown"


class CapabilityCheck(ContractModel):
    block_id: str
    capability: str
    status: CapabilityStatus
    reason: str
    evidence: list[str] = Field(default_factory=list)


class RuntimeReconciliation(ContractModel):
    captured_at: str
    checks: list[CapabilityCheck] = Field(default_factory=list)


class PlanUpdateKind(StrEnum):
    INSERT_BEFORE = "insert_before"
    REVISIT = "revisit"


class WorkflowPlanUpdate(ContractModel):
    """One bounded, source-grounded edit to a live deployment plan."""

    kind: PlanUpdateKind
    reason: str = Field(min_length=1, max_length=600)
    prerequisite_step_id: str | None = Field(default=None, pattern=r"^S[0-9]+$")
    component_name: str | None = Field(default=None, min_length=1, max_length=120)
    block_id: str | None = Field(default=None, pattern=r"^B[0-9]+$")
    executor: AnalysisExecutor | None = None
    source_ref: AnalysisSourceRef | None = None
    working_directory: str | None = None
    command: str | None = Field(default=None, min_length=1, max_length=2_000)
    required_inputs: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_shape(self) -> WorkflowPlanUpdate:
        if self.kind == PlanUpdateKind.REVISIT:
            if not self.prerequisite_step_id:
                raise ValueError("a revisit update requires prerequisite_step_id")
            return self
        if not all((self.executor, self.source_ref, self.command)):
            raise ValueError("an insert_before update requires executor, source_ref, and command")
        if self.component_name and not self.block_id:
            raise ValueError("a newly discovered component requires block_id")
        return self


@dataclass(frozen=True, slots=True)
class PlanRevision:
    workflow: DeploymentWorkflow
    blocks: FunctionalBlocksDocument | None
    added_step: DeploymentWorkflowStep | None = None
    revisit_step_id: str | None = None


CapabilityEvidence = tuple[CapabilityStatus, list[str]]
CapabilityRule = Callable[[RuntimeContextSnapshot], CapabilityEvidence]


def reconcile_provided_capabilities(
    context: DeploymentContext,
    snapshot: RuntimeContextSnapshot,
) -> RuntimeReconciliation:
    """Resolve provided claims from observations; failed or unsupported probes stay unknown."""
    return RuntimeReconciliation(
        captured_at=snapshot.captured_at,
        checks=[
            _check_capability(block.id, capability, snapshot)
            for block in context.provided_blocks
            for capability in block.provides
        ],
    )


def apply_runtime_reconciliation(
    context: DeploymentContext,
    reconciliation: RuntimeReconciliation,
) -> DeploymentContext:
    """Remove only context claims that successful observations proved false."""
    missing = {
        (check.block_id, check.capability)
        for check in reconciliation.checks
        if check.status == CapabilityStatus.MISSING
    }
    return context.model_copy(
        update={
            "provided_blocks": [
                block.model_copy(
                    update={
                        "provides": [
                            capability
                            for capability in block.provides
                            if (block.id, capability) not in missing
                        ]
                    }
                )
                for block in context.provided_blocks
            ]
        }
    )


def apply_plan_update(
    workflow: DeploymentWorkflow,
    blocks: FunctionalBlocksDocument | None,
    update: WorkflowPlanUpdate,
    *,
    failed_step_id: str,
    failed_block_id: str | None,
    selected_block_id: str | None = None,
) -> PlanRevision:
    """Validate one bounded plan change and return an acyclic revised plan."""
    steps = {step.id: step for step in workflow.steps}
    if failed_step_id not in steps:
        raise ValueError(f"failed step is no longer in the workflow: {failed_step_id}")
    target = steps[failed_step_id]
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
        return PlanRevision(
            DeploymentWorkflow.model_validate(revised.model_dump(mode="python")),
            blocks,
            revisit_step_id=prerequisite,
        )

    assert update.executor is not None
    assert update.source_ref is not None
    assert update.command is not None
    if selected_block_id and update.block_id and update.block_id != selected_block_id:
        raise ValueError("plan update cannot leave the selected block")
    target_ids = target.targets
    revised_blocks = blocks
    if update.component_name:
        if blocks is None:
            raise ValueError("a new component requires functional-blocks.yaml")
        block_id = update.block_id or failed_block_id
        block_by_id = {block.id: block for block in blocks.blocks}
        if block_id not in block_by_id or block_by_id[block_id].state == "provided":
            raise ValueError("new component must belong to an existing discovered block")
        component = _new_component(blocks, update)
        revised_blocks = _add_component(blocks, component, block_id, failed_block_id)
        target_ids = [WorkflowTarget(id=component.id, name=component.name)]

    added = DeploymentWorkflowStep(
        id=_next_identifier(steps, "S"),
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
    return PlanRevision(
        DeploymentWorkflow.model_validate(revised.model_dump(mode="python")),
        revised_blocks,
        added_step=added,
    )


def invalidated_completed_steps(
    workflow: DeploymentWorkflow,
    root: str,
    completed: set[str],
) -> set[str]:
    """Return completed work affected by revisiting one prerequisite."""
    reopened = {root}
    while newly_affected := {
        step.id
        for step in workflow.steps
        if step.id not in reopened and any(item in reopened for item in step.after)
    }:
        reopened.update(newly_affected)
    return reopened & completed


def _new_component(
    blocks: FunctionalBlocksDocument,
    update: WorkflowPlanUpdate,
) -> FunctionalComponent:
    assert update.component_name is not None
    assert update.executor is not None
    assert update.source_ref is not None
    return FunctionalComponent(
        id=_next_identifier(
            (component.id for block in blocks.blocks for component in block.components),
            "C",
            slug=slugify(update.component_name),
        ),
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


def _add_component(
    document: FunctionalBlocksDocument,
    component: FunctionalComponent,
    block_id: str,
    failed_block_id: str | None,
) -> FunctionalBlocksDocument:
    blocks: list[FunctionalBlock] = []
    for block in document.blocks:
        changes: dict[str, Any] = {}
        if block.id == block_id:
            changes["components"] = [*block.components, component]
        if failed_block_id and block.id == failed_block_id and block_id != failed_block_id:
            changes["after"] = list(dict.fromkeys([*block.after, block_id]))
        blocks.append(block.model_copy(update=changes) if changes else block)
    dependencies = {block.id: set(block.after) for block in blocks}
    revised = document.model_copy(
        update={
            "blocks": blocks,
            "levels": topological_levels(
                [block.id for block in blocks],
                dependencies,
                cycle_label="functional block plan update",
            ),
        }
    )
    return FunctionalBlocksDocument.model_validate(revised.model_dump(mode="python"))


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
            if identifier.startswith(prefix)
        ),
        default=0,
    )
    base = f"{prefix}{largest + 1:03d}"
    return f"{base}_{slug or 'component'}" if prefix == "C" else base


def _check_capability(
    block_id: str,
    capability: str,
    snapshot: RuntimeContextSnapshot,
) -> CapabilityCheck:
    lowered = capability.casefold()
    normalized = lowered.replace("_", "-")
    rule = _RULES.get(normalized)
    if lowered.startswith("helm-release:"):
        status, evidence = _helm_release(snapshot, lowered.partition(":")[2])
    else:
        status, evidence = rule(snapshot) if rule else (CapabilityStatus.UNKNOWN, [])
    return CapabilityCheck(
        block_id=block_id,
        capability=capability,
        status=status,
        reason={
            CapabilityStatus.SATISFIED: "runtime evidence supports the provided capability",
            CapabilityStatus.MISSING: "successful runtime probes did not find the capability",
            CapabilityStatus.UNKNOWN: "runtime probes do not prove whether the capability exists",
        }[status],
        evidence=evidence,
    )


def _probe(
    snapshot: RuntimeContextSnapshot,
    provider: str,
    name: str,
    present: bool,
) -> CapabilityEvidence:
    if not _probe_succeeded(snapshot, provider, name):
        return CapabilityStatus.UNKNOWN, []
    return (
        CapabilityStatus.SATISFIED if present else CapabilityStatus.MISSING,
        [f"runtime:{provider}/{name}"],
    )


def _ingress(snapshot: RuntimeContextSnapshot) -> CapabilityEvidence:
    kubernetes = snapshot.kubernetes
    present = bool(kubernetes.ingress_classes) or any(
        (item.ready or 0) > 0
        and any(term in item.name.casefold() for term in ("ingress", "gateway"))
        for item in kubernetes.workloads
    )
    probes = ("ingress_classes", "workloads")
    if not all(_probe_succeeded(snapshot, "kubernetes", name) for name in probes):
        return CapabilityStatus.UNKNOWN, []
    return (
        CapabilityStatus.SATISFIED if present else CapabilityStatus.MISSING,
        [f"runtime:kubernetes/{name}" for name in probes],
    )


def _csi_provisioning(snapshot: RuntimeContextSnapshot) -> CapabilityEvidence:
    probes = ("storage_classes", "csi_drivers")
    if not all(_probe_succeeded(snapshot, "kubernetes", name) for name in probes):
        return CapabilityStatus.UNKNOWN, []
    drivers = {driver.name.casefold() for driver in snapshot.kubernetes.csi_drivers}
    present = any(
        storage_class.state and storage_class.state.casefold() in drivers
        for storage_class in snapshot.kubernetes.storage_classes
    )
    return (
        CapabilityStatus.SATISFIED if present else CapabilityStatus.MISSING,
        [f"runtime:kubernetes/{name}" for name in probes],
    )


def _helm_release(snapshot: RuntimeContextSnapshot, release: str) -> CapabilityEvidence:
    namespace, separator, name = release.rpartition("/")
    present = any(
        item.name.casefold() == name
        and (not separator or (item.namespace or "").casefold() == namespace)
        and (item.state or "").casefold() == "deployed"
        for item in snapshot.kubernetes.helm_releases
    )
    return _probe(snapshot, "kubernetes", "helm_releases", present)


def _probe_succeeded(snapshot: RuntimeContextSnapshot, provider: str, name: str) -> bool:
    return any(
        probe.provider == provider and probe.name == name and probe.succeeded
        for probe in snapshot.probes
    )


_RULES: dict[str, CapabilityRule] = {
    "aws-account": lambda snapshot: _probe(
        snapshot,
        "aws",
        "caller_identity",
        bool(snapshot.aws.account and snapshot.aws.principal_arn),
    ),
    "aws-api-access": lambda snapshot: _probe(
        snapshot,
        "aws",
        "caller_identity",
        bool(snapshot.aws.account and snapshot.aws.principal_arn),
    ),
    "linux-host": lambda snapshot: _probe(
        snapshot,
        "host",
        "system",
        snapshot.host.system.casefold() == "linux",
    ),
    "kubernetes-api": lambda snapshot: _probe(
        snapshot,
        "kubernetes",
        "version",
        snapshot.kubernetes.available,
    ),
    "workload-scheduling": lambda snapshot: _probe(
        snapshot,
        "kubernetes",
        "nodes",
        any((node.ready or 0) > 0 for node in snapshot.kubernetes.nodes),
    ),
    "persistent-storage": lambda snapshot: _probe(
        snapshot,
        "kubernetes",
        "storage_classes",
        any(item.is_default for item in snapshot.kubernetes.storage_classes),
    ),
    "csi-provisioning": _csi_provisioning,
    "ingress": _ingress,
}
