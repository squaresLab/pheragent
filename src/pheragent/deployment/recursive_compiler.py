from __future__ import annotations

from pheragent.utils import slugify

from .analysis_models import (
    AnalysisBlockType,
    AnalysisExecutor,
    AnalysisQuestion,
    ComponentDisposition,
    DeploymentContext,
    DeploymentWorkflow,
    DeploymentWorkflowStep,
    FunctionalBlock,
    FunctionalBlocksDocument,
    FunctionalComponent,
    FunctionalDeployRef,
    ReadinessCoverage,
    ValidationSignal,
    WorkflowStepKind,
    WorkflowStepStatus,
    WorkflowTarget,
)
from .graph import topological_levels
from .recursive_plan import PlanNode, RecursivePlan, StepState


def compile_recursive_plan(
    plan: RecursivePlan,
    context: DeploymentContext,
) -> tuple[FunctionalBlocksDocument, DeploymentWorkflow]:
    """Translate a grounded recursive tree into the stable product artifacts."""
    stages = [child for root in plan.roots for child in (root.children or [root])]
    blocks = [_provided_block(item) for item in context.provided_blocks]
    previous_blocks = _provided_sinks(context)
    steps: list[DeploymentWorkflowStep] = []
    validations: list[ValidationSignal] = []
    unresolved: list[AnalysisQuestion] = []
    previous_step: str | None = None
    blocked_by: str | None = None
    component_number = 0

    for block_number, stage in enumerate(stages, start=_next_block_number(blocks)):
        components: list[FunctionalComponent] = []
        leaves = list(_leaves(stage))
        for node in leaves:
            if node.state not in {StepState.EXECUTABLE, StepState.SATISFIED}:
                unresolved.append(
                    AnalysisQuestion(
                        question=node.goal,
                        reason=node.issue or f"deployment outcome is {node.state.value}",
                    )
                )
                blocked_by = blocked_by or f"earlier deployment outcome {node.id} is unresolved"
                continue
            component_number += 1
            component_id = f"C{component_number:03d}_{slugify(node.title)}"
            source = node.operation_source_ref or next(iter(node.source_refs), None)
            deploy = (
                FunctionalDeployRef(
                    executor=_executor(node.command or ""),
                    ref=source.path,
                    repo_id=source.repo_id,
                )
                if node.state == StepState.EXECUTABLE and source
                else None
            )
            components.append(
                FunctionalComponent(
                    id=component_id,
                    name=node.title,
                    implementation=node.title,
                    deployable=node.state == StepState.EXECUTABLE,
                    external=False,
                    disposition=(
                        ComponentDisposition.DEPLOYMENT_COMPONENT
                        if deploy
                        else ComponentDisposition.PROVIDED_PREREQUISITE
                    ),
                    deploy=deploy,
                )
            )
            if not deploy or not node.command or source is None:
                continue
            step_id = f"S{len(steps) + 1:03d}"
            blockers = [blocked_by] if blocked_by else []
            steps.append(
                DeploymentWorkflowStep(
                    id=step_id,
                    kind=WorkflowStepKind.COMPONENT,
                    targets=[WorkflowTarget(id=component_id, name=node.title)],
                    executor=deploy.executor,
                    source_ref=source,
                    operation_source_ref=source,
                    working_directory=node.working_directory,
                    command=node.command,
                    after=[previous_step] if previous_step else [],
                    status=(
                        WorkflowStepStatus.BLOCKED if blockers else WorkflowStepStatus.READY
                    ),
                    blockers=blockers,
                )
            )
            previous_step = step_id
            validations.append(
                ValidationSignal(
                    subject=component_id,
                    check=node.success_check or "deployment command completes successfully",
                    source_ref=source,
                )
            )
        block_id = f"B{block_number}"
        blocks.append(
            FunctionalBlock(
                id=block_id,
                name=stage.title,
                type=AnalysisBlockType.UNKNOWN,
                subtype=slugify(stage.title),
                state=(
                    "provided"
                    if leaves and all(node.state == StepState.SATISFIED for node in leaves)
                    else "discovered"
                ),
                after=previous_blocks,
                components=components,
            )
        )
        previous_blocks = [block_id]

    complete = plan.deployment_ready and not unresolved and bool(steps)
    coverage = ReadinessCoverage(
        **{name: complete for name in ReadinessCoverage.model_fields}
    )
    document = FunctionalBlocksDocument(
        system=context.system,
        deployment=context.deployment,
        blocks=blocks,
        levels=topological_levels(
            [block.id for block in blocks],
            {block.id: set(block.after) for block in blocks},
            cycle_label="recursive functional block graph",
        ),
        unresolved=unresolved,
    )
    workflow = DeploymentWorkflow(
        system=context.system,
        ready_for_execution=complete,
        coverage=coverage,
        provided_blocks=context.provided_blocks,
        steps=steps,
        validation_checks=validations,
        unresolved=unresolved,
    )
    return document, workflow


def _leaves(node: PlanNode):
    if not node.children:
        yield node
    for child in node.children:
        yield from _leaves(child)


def _provided_block(block) -> FunctionalBlock:
    return FunctionalBlock(
        id=block.id,
        name=block.subtype.replace("-", " ").replace("_", " ").title(),
        type=block.type,
        subtype=block.subtype,
        implementation=block.implementation,
        state=block.state,
        after=block.after,
        provides=block.provides,
    )


def _provided_sinks(context: DeploymentContext) -> list[str]:
    prerequisites = {item for block in context.provided_blocks for item in block.after}
    return [block.id for block in context.provided_blocks if block.id not in prerequisites]


def _next_block_number(blocks: list[FunctionalBlock]) -> int:
    numbers = [int(block.id[1:]) for block in blocks if block.id[1:].isdigit()]
    return max(numbers, default=-1) + 1


def _executor(command: str) -> AnalysisExecutor:
    normalized = command.lstrip().casefold()
    for prefix, executor in (
        ("terraform ", AnalysisExecutor.TERRAFORM),
        ("ansible-playbook ", AnalysisExecutor.ANSIBLE),
        ("helm ", AnalysisExecutor.HELM),
        ("kubectl ", AnalysisExecutor.KUBERNETES),
        ("kustomize ", AnalysisExecutor.KUSTOMIZE),
        ("docker compose ", AnalysisExecutor.DOCKER_COMPOSE),
        ("docker-compose ", AnalysisExecutor.DOCKER_COMPOSE),
    ):
        if normalized.startswith(prefix):
            return executor
    return AnalysisExecutor.SHELL
