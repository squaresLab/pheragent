from __future__ import annotations

from collections.abc import Iterable

from .analysis_models import (
    AnalysisBlockType,
    AnalysisExecutor,
    AnalysisQuestion,
    AnalysisRelationType,
    AnalysisSourceRef,
    CandidateComponent,
    ClaimSource,
    ComponentDeployment,
    ComponentDisposition,
    DeploymentAction,
    DeploymentContext,
    DeploymentSignalBundle,
    DeploymentWorkflow,
    DeploymentWorkflowStep,
    DisagreementResolution,
    ReadinessCoverage,
    SourceDisagreement,
    WorkflowStepKind,
    WorkflowStepStatus,
    WorkflowTarget,
)
from .graph import add_dependency_if_acyclic, topological_order
from .investigation_models import (
    EvidenceKind,
    EvidenceObservation,
    InvestigationSynthesis,
    SourcePurpose,
)
from .redaction import redact_secrets


def operation_fingerprint(
    *,
    executor: str,
    command: str,
    working_directory: str | None,
    source_ref: AnalysisSourceRef,
) -> tuple[object, ...]:
    """Identify one grounded operation independently of its component labels."""
    return (
        executor,
        command.strip(),
        working_directory or ".",
        source_ref.repo_id,
        source_ref.path,
        source_ref.start_line,
        source_ref.end_line,
    )


def ordered_workflow_steps(
    steps: Iterable[DeploymentWorkflowStep],
) -> list[DeploymentWorkflowStep]:
    """Order workflow steps by their declared prerequisites."""
    materialized = list(steps)
    by_id = {step.id: step for step in materialized}
    dependencies = {step.id: set(step.after) for step in materialized}
    ordered_ids = topological_order(
        by_id,
        dependencies,
        cycle_label="deployment workflow",
    )
    return [by_id[step_id] for step_id in ordered_ids]


def _grounded_operation(
    deployment: ComponentDeployment | None,
) -> tuple[str | None, str | None]:
    if deployment is None or deployment.command is None:
        return None, None
    if redact_secrets(deployment.command) != deployment.command:
        return None, deployment.working_directory
    return deployment.command, deployment.working_directory


type _WorkflowNode = tuple[str, CandidateComponent, DeploymentAction | None]


def _group_workflow_operations(
    nodes: list[_WorkflowNode],
) -> tuple[list[tuple[str, list[_WorkflowNode]]], dict[str, str]]:
    """Collapse components materialized by the same grounded command into one operation."""
    groups: dict[tuple[object, ...], list[_WorkflowNode]] = {}
    for node in nodes:
        groups.setdefault(_workflow_operation_key(node), []).append(node)

    operations: list[tuple[str, list[_WorkflowNode]]] = []
    operation_by_node: dict[str, str] = {}
    for members in groups.values():
        operation_id = members[0][0]
        operations.append((operation_id, members))
        operation_by_node.update(
            (node_id, operation_id) for node_id, _component, _action in members
        )
    return operations, operation_by_node


def _workflow_operation_key(node: _WorkflowNode) -> tuple[object, ...]:
    node_id, component, action = node
    deployment = action.deployment if action else component.deployment
    command, working_directory = _grounded_operation(deployment)
    if action is not None or deployment is None or command is None:
        return ("node", node_id)
    source_ref = deployment.operation_source_ref or component.source_ref
    return ("grounded-operation",) + operation_fingerprint(
        executor=deployment.executor.value,
        command=command,
        working_directory=working_directory,
        source_ref=source_ref,
    )


def build_deployment_workflow(
    *,
    context: DeploymentContext,
    signals: DeploymentSignalBundle,
    observations: tuple[EvidenceObservation, ...],
    synthesis: InvestigationSynthesis | None,
    llm_completed: bool,
    mandatory_probes: tuple[str, ...],
    documentation_expected: bool,
) -> DeploymentWorkflow:
    disagreements, invalid_disagreements = _partition_disagreements(
        synthesis.disagreements if synthesis else [],
        observations,
    )
    executable_components = [
        component
        for component in signals.candidate_components
        if component.disposition
        in {ComponentDisposition.DEPLOYMENT_COMPONENT, ComponentDisposition.UNCERTAIN}
        and component.installed_by is None
    ]
    component_by_id = {component.id: component for component in signals.candidate_components}
    action_by_candidate = {
        action.source_candidate_id: action for action in signals.deployment_actions
    }
    raw_workflow_nodes: list[_WorkflowNode] = []
    executable_ids = {component.id for component in executable_components}
    for candidate in signals.candidate_components:
        action = action_by_candidate.get(candidate.id)
        if action is not None:
            raw_workflow_nodes.append((action.id, candidate, action))
        elif candidate.id in executable_ids:
            raw_workflow_nodes.append((candidate.id, candidate, None))
    workflow_operations, operation_by_node = _group_workflow_operations(raw_workflow_nodes)
    step_id_by_node = {
        node_id: f"S{index:03d}"
        for index, (node_id, _members) in enumerate(workflow_operations, start=1)
    }
    relation_node = {}
    for component in signals.candidate_components:
        raw_node = (
            action_by_candidate[component.id].id
            if component.id in action_by_candidate
            else component.installed_by
            if component.installed_by is not None
            else component.id
        )
        relation_node[component.id] = operation_by_node.get(raw_node, raw_node)
    after_by_node: dict[str, set[str]] = {node_id: set() for node_id in step_id_by_node}
    relation_conflicts: set[str] = set()
    for relation in signals.relations:
        source_node = relation_node.get(relation.source, relation.source)
        target_node = relation_node.get(relation.target, relation.target)
        if relation.relation == AnalysisRelationType.ORDERED_BEFORE:
            if target_node in after_by_node and source_node in step_id_by_node:
                if target_node == source_node:
                    continue
                if not add_dependency_if_acyclic(
                    after_by_node,
                    dependent=target_node,
                    prerequisite=source_node,
                ):
                    relation_conflicts.add(target_node)
        elif (
            relation.relation
            in {
                AnalysisRelationType.REQUIRES,
                AnalysisRelationType.HEALTH_GATED_BY,
            }
            and source_node in after_by_node
            and target_node in step_id_by_node
        ):
            if source_node == target_node:
                continue
            if not add_dependency_if_acyclic(
                after_by_node,
                dependent=source_node,
                prerequisite=target_node,
            ):
                relation_conflicts.add(source_node)

    binding_conflicts: set[str] = set()
    for action in signals.deployment_actions:
        action_node = operation_by_node.get(action.id, action.id)
        owner_node = operation_by_node.get(action.owner_component_id, action.owner_component_id)
        if (
            action_node in after_by_node
            and owner_node in step_id_by_node
            and not add_dependency_if_acyclic(
                after_by_node,
                dependent=action_node,
                prerequisite=owner_node,
            )
        ):
            binding_conflicts.add(action_node)

    # Preserve the same conservative layer order as the functional-block DAG.
    # Explicit component relations remain the finer-grained source of truth.
    by_type: dict[AnalysisBlockType, list[str]] = {}
    for component in executable_components:
        operation_id = operation_by_node.get(component.id, component.id)
        operations = by_type.setdefault(component.classification.block_type, [])
        if operation_id not in operations:
            operations.append(operation_id)
    for component in executable_components:
        operation_id = operation_by_node.get(component.id, component.id)
        prerequisites: list[str] = []
        if component.classification.block_type == AnalysisBlockType.RUNTIME_ENVIRONMENT:
            prerequisites = by_type.get(AnalysisBlockType.BASE_INFRASTRUCTURE, [])
        elif component.classification.block_type in {
            AnalysisBlockType.SHARED_SERVICES,
            AnalysisBlockType.OPERATIONS,
        }:
            prerequisites = by_type.get(AnalysisBlockType.RUNTIME_ENVIRONMENT, [])
        elif component.classification.block_type == AnalysisBlockType.APPLICATION:
            prerequisites = (
                by_type.get(AnalysisBlockType.SHARED_SERVICES)
                or by_type.get(AnalysisBlockType.RUNTIME_ENVIRONMENT)
                or by_type.get(AnalysisBlockType.BASE_INFRASTRUCTURE, [])
            )
        for prerequisite in prerequisites:
            if prerequisite in step_id_by_node and prerequisite != operation_id:
                add_dependency_if_acyclic(
                    after_by_node,
                    dependent=operation_id,
                    prerequisite=prerequisite,
                )

    workflow_paths = {
        (
            (action.source_ref.repo_id, action.source_ref.path)
            if action
            else (component.source_ref.repo_id, component.source_ref.path)
        )
        for _node_id, component, action in raw_workflow_nodes
    }
    workflow_paths.update(
        (
            deployment.operation_source_ref.repo_id,
            deployment.operation_source_ref.path,
        )
        for _node_id, component, action in raw_workflow_nodes
        if (deployment := action.deployment if action else component.deployment)
        and deployment.operation_source_ref
    )
    dynamic_paths = {
        (item.source_id, item.path)
        for item in observations
        if item.requires_resolution
        and (item.blocks_execution or (item.source_id, item.path) in workflow_paths)
    }
    steps = []
    for node_id, members in workflow_operations:
        _representative_id, component, action = members[0]
        blockers: list[str] = []
        deployment = action.deployment if action else component.deployment
        command, working_directory = _grounded_operation(deployment)
        source_ref = action.source_ref if action else component.source_ref
        if deployment is None:
            blockers.append("no grounded deployment entrypoint")
        elif command is None:
            blockers.append(f"executor {deployment.executor} requires a grounded invocation")
        operation_path = (
            (
                deployment.operation_source_ref.repo_id,
                deployment.operation_source_ref.path,
            )
            if deployment and deployment.operation_source_ref
            else None
        )
        if (
            source_ref.repo_id,
            source_ref.path,
        ) in dynamic_paths or operation_path in dynamic_paths:
            blockers.append("entrypoint contains unresolved dynamic deployment behavior")
        if action is None and any(
            member_component.disposition == ComponentDisposition.UNCERTAIN
            for _member_id, member_component, _member_action in members
        ):
            blockers.append("component disposition is uncertain")
        if node_id in binding_conflicts:
            blockers.append("action ownership conflicts with source-derived execution order")
        if node_id in relation_conflicts:
            blockers.append("source-derived component dependencies contain a cycle")
        owner = component_by_id[action.owner_component_id] if action else component
        targets = (
            [WorkflowTarget(id=owner.id, name=owner.name)]
            if action
            else [
                WorkflowTarget(id=member_component.id, name=member_component.name)
                for _member_id, member_component, _member_action in members
            ]
        )
        required_inputs = sorted(
            {
                required_input
                for _member_id, member_component, member_action in members
                for required_input in (
                    member_action.deployment.required_inputs
                    if member_action
                    else member_component.deployment.required_inputs
                    if member_component.deployment
                    else []
                )
            }
        )
        steps.append(
            DeploymentWorkflowStep(
                id=step_id_by_node[node_id],
                kind=(WorkflowStepKind.ACTION if action else WorkflowStepKind.COMPONENT),
                targets=targets,
                action_id=action.id if action else None,
                action_name=action.name if action else None,
                executor=(deployment.executor if deployment else AnalysisExecutor.UNKNOWN),
                source_ref=source_ref,
                operation_source_ref=(deployment.operation_source_ref if deployment else None),
                working_directory=working_directory,
                command=command,
                required_inputs=required_inputs,
                after=sorted(step_id_by_node[item] for item in after_by_node[node_id]),
                status=(WorkflowStepStatus.BLOCKED if blockers else WorkflowStepStatus.READY),
                blockers=blockers,
            )
        )

    root_paths = {
        (root.source_ref.repo_id, root.source_ref.path) for root in signals.deployment_roots
    }
    observed_roots = {
        (item.source_id, item.path)
        for item in observations
        if item.kind == EvidenceKind.PRIMARY_ROOT
    }
    stage_components = {
        component_id for stage in signals.deployment_stages for component_id in stage.component_ids
    }
    accounted = {
        component.id
        for component in signals.candidate_components
        if component.disposition != ComponentDisposition.IMPLEMENTATION_DETAIL
    } | {action.source_candidate_id for action in signals.deployment_actions}
    required_probes = {
        "primary_roots",
        "component_entrypoints",
        "unreferenced_installers",
        "documentation_requirements",
        "configuration_endpoints",
        "dynamic_deployments",
        "profile_conflicts",
    }
    unresolved_disagreements = any(
        item.resolution == DisagreementResolution.UNRESOLVED for item in disagreements
    )
    disagreements_reviewed = bool(llm_completed and synthesis and not invalid_disagreements)
    external_requirements_reviewed = bool(llm_completed and synthesis)
    semantic_complete = bool(
        llm_completed
        and synthesis
        and not synthesis.unresolved
        and not invalid_disagreements
        and not unresolved_disagreements
        and stage_components <= accounted
        and required_probes <= set(mandatory_probes)
    )
    coverage = ReadinessCoverage(
        primary_roots_traced=root_paths <= observed_roots,
        deployment_actions_accounted=stage_components <= accounted,
        independent_gap_search_complete=required_probes <= set(mandatory_probes),
        docs_repo_disagreements_reviewed=disagreements_reviewed,
        docs_repo_disagreements_resolved=(disagreements_reviewed and not unresolved_disagreements),
        external_requirements_reviewed=external_requirements_reviewed,
        dynamic_deployments_resolved=not dynamic_paths,
        deployable_components_grounded=all(
            step.status == WorkflowStepStatus.READY for step in steps
        ),
        semantic_coverage_complete=semantic_complete,
        validation_checks_grounded=bool(signals.validation_signals),
        documentation_evidence_reviewed=(
            not documentation_expected
            or any(item.source_purpose == SourcePurpose.DOCUMENTATION for item in observations)
        ),
    )
    unresolved = list(signals.unresolved)
    unresolved.extend(invalid_disagreements)
    if not llm_completed:
        unresolved.append(
            AnalysisQuestion(
                question="Has the evidence-guided deployment investigation completed?",
                reason="The semantic investigation did not complete successfully.",
            )
        )
    if dynamic_paths:
        unresolved.append(
            AnalysisQuestion(
                question="How should the dynamic deployment operations be resolved safely?",
                reason="Dynamic operations were detected and were deliberately not executed.",
            )
        )
    if unresolved_disagreements:
        unresolved.append(
            AnalysisQuestion(
                question="Which source should resolve the deployment disagreement?",
                reason="A repository/documentation disagreement remains unresolved.",
            )
        )
    if relation_conflicts:
        unresolved.append(
            AnalysisQuestion(
                question="Which component dependency should resolve the workflow cycle?",
                reason=(
                    "Conflicting component relationships were preserved as a blocked workflow "
                    "decision instead of producing an invalid dependency graph."
                ),
            )
        )
    if not signals.validation_signals:
        unresolved.append(
            AnalysisQuestion(
                question="How will deployed blocks be validated before dependents run?",
                reason="No source-grounded readiness or validation operation was found.",
            )
        )
    ready = llm_completed and all(coverage.model_dump(mode="python").values())
    return DeploymentWorkflow(
        system=context.system,
        ready_for_execution=ready,
        coverage=coverage,
        provided_blocks=context.provided_blocks,
        steps=steps,
        external_requirements=signals.external_requirements,
        validation_checks=signals.validation_signals,
        disagreements=disagreements,
        unresolved=unresolved,
    )


def _partition_disagreements(
    disagreements: list[SourceDisagreement],
    observations: tuple[EvidenceObservation, ...],
) -> tuple[list[SourceDisagreement], list[AnalysisQuestion]]:
    """Quarantine malformed optional conflicts without losing valid synthesis work."""
    purpose_by_id = {item.id: item.source_purpose.value for item in observations}
    valid: list[SourceDisagreement] = []
    invalid: list[AnalysisQuestion] = []
    for disagreement in disagreements:
        if disagreement.left_source == disagreement.right_source:
            invalid.append(
                AnalysisQuestion(
                    question=(f"What distinct sources disagree about {disagreement.subject}?"),
                    reason=(
                        "The model compared a source with itself. The invalid disagreement was "
                        "quarantined; other grounded synthesis decisions were retained."
                    ),
                )
            )
            continue
        evidence_purposes = {
            purpose_by_id[evidence_id]
            for evidence_id in disagreement.evidence_ids
            if evidence_id in purpose_by_id
        }
        required_purposes = {
            source.value
            for source in (disagreement.left_source, disagreement.right_source)
            if source != ClaimSource.CONTEXT
        }
        if required_purposes <= evidence_purposes:
            valid.append(disagreement)
            continue
        invalid.append(
            AnalysisQuestion(
                question=f"What evidence establishes the conflict about {disagreement.subject}?",
                reason=(
                    "The model reported a source disagreement without evidence from every "
                    "repository/documentation claim source. The conflict was quarantined; other "
                    "valid synthesis decisions were retained."
                ),
            )
        )
    return valid, invalid
