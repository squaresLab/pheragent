from __future__ import annotations

import re
from collections import defaultdict

import yaml

from .analysis_models import (
    AnalysisBlockType,
    AnalysisRelationType,
    CandidateComponent,
    ComponentDisposition,
    DeploymentContext,
    DeploymentSignalBundle,
    FunctionalBlock,
    FunctionalBlocksDocument,
    FunctionalComponent,
    FunctionalDeployRef,
    GoldDefinition,
)
from .evaluation import evaluate_functional_blocks
from .graph import add_dependency_if_acyclic, topological_levels
from .source_manager import AcquiredSource


def build_functional_blocks(
    context: DeploymentContext,
    signals: DeploymentSignalBundle,
    sources: dict[str, AcquiredSource],
    gold: GoldDefinition | None,
    *,
    llm_usage: dict[str, int],
    llm_stage_statuses: dict[str, str],
) -> FunctionalBlocksDocument:
    """Build and score the compact functional-block artifact from discovery signals."""
    blocks = _provided_blocks(context)
    next_id = _next_block_number(blocks)
    for name, block_type, subtype, components in _component_groups(signals):
        blocks.append(_functional_block(next_id, name, block_type, subtype, components))
        next_id += 1
    _attach_dependencies(blocks, signals)
    levels = _block_levels(blocks)
    evaluation = evaluate_functional_blocks(blocks, signals, sources, gold).model_copy(
        update={
            "llm_input_tokens": int(llm_usage.get("input_tokens", 0)),
            "llm_output_tokens": int(llm_usage.get("output_tokens", 0)),
            "llm_requests": int(llm_usage.get("requests", 0)),
            "llm_status": "; ".join(
                f"{stage}={status}" for stage, status in llm_stage_statuses.items()
            ),
            "llm_stages": llm_stage_statuses,
        }
    )
    document = FunctionalBlocksDocument(
        system=context.system,
        deployment=context.deployment,
        blocks=blocks,
        levels=levels,
        unresolved=signals.unresolved,
        evaluation=evaluation,
    )
    line_count = len(
        yaml.safe_dump(
            document.model_dump(mode="json", exclude_none=True),
            sort_keys=False,
            allow_unicode=True,
        ).splitlines()
    )
    return document.model_copy(
        update={"evaluation": evaluation.model_copy(update={"artifact_line_count": line_count})}
    )


def _provided_blocks(context: DeploymentContext) -> list[FunctionalBlock]:
    return [
        FunctionalBlock(
            id=item.id,
            name=_display_name(item.subtype),
            type=item.type,
            subtype=item.subtype,
            implementation=item.implementation,
            state=item.state,
            after=item.after,
            provides=item.provides,
        )
        for item in context.provided_blocks
    ]


def _next_block_number(blocks: list[FunctionalBlock]) -> int:
    return (
        max(
            (
                int(match.group(1))
                for block in blocks
                if (match := re.fullmatch(r"B(\d+)", block.id))
            ),
            default=-1,
        )
        + 1
    )


def _component_groups(
    signals: DeploymentSignalBundle,
) -> list[tuple[str, AnalysisBlockType, str, list[CandidateComponent]]]:
    grouped: dict[tuple[AnalysisBlockType, str, str | None], list[CandidateComponent]] = (
        defaultdict(list)
    )
    for component in signals.candidate_components:
        if component.disposition in {
            ComponentDisposition.IMPLEMENTATION_DETAIL,
            ComponentDisposition.PROVIDED_PREREQUISITE,
        }:
            continue
        classification = component.classification
        grouped[(classification.block_type, classification.subtype, classification.domain)].append(
            component
        )
    return [
        (_display_name(domain or subtype), block_type, subtype, components)
        for (block_type, subtype, domain), components in sorted(
            grouped.items(),
            key=lambda item: (item[0][0].value, item[0][1], item[0][2] or ""),
        )
    ]


def _functional_block(
    number: int,
    name: str,
    block_type: AnalysisBlockType,
    subtype: str,
    components: list[CandidateComponent],
) -> FunctionalBlock:
    return FunctionalBlock(
        id=f"B{number}",
        name=name,
        type=block_type,
        subtype=subtype,
        provides=sorted(
            {capability for component in components for capability in component.capabilities}
        ),
        components=[_functional_component(component) for component in components],
    )


def _functional_component(component: CandidateComponent) -> FunctionalComponent:
    deployment = component.deployment
    deploy_ref = (
        FunctionalDeployRef(
            executor=deployment.executor,
            ref=deployment.entrypoint,
            repo_id=component.source_ref.repo_id,
            required_inputs=deployment.required_inputs,
        )
        if deployment
        else None
    )
    return FunctionalComponent(
        id=component.id,
        name=component.name,
        implementation=component.implementation,
        deployable=component.deployable,
        external=component.external,
        disposition=component.disposition,
        installed_by=component.installed_by,
        deploy=deploy_ref,
    )


def _attach_dependencies(
    blocks: list[FunctionalBlock],
    signals: DeploymentSignalBundle,
) -> None:
    dependencies = {block.id: set(block.after) for block in blocks}
    component_blocks = {
        component.id: block.id for block in blocks for component in block.components
    }
    capability_blocks: dict[str, set[str]] = defaultdict(set)
    for block in blocks:
        for capability in block.provides:
            capability_blocks[capability].add(block.id)

    for relation in sorted(
        signals.relations,
        key=lambda item: (item.relation.value, item.source, item.target),
    ):
        endpoints = _dependency_endpoints(relation.relation, relation.source, relation.target)
        if endpoints is None:
            continue
        dependent, prerequisite = (
            _endpoint_block(endpoint, component_blocks, capability_blocks)
            for endpoint in endpoints
        )
        if dependent and prerequisite:
            add_dependency_if_acyclic(
                dependencies,
                dependent=dependent,
                prerequisite=prerequisite,
            )

    by_type: dict[AnalysisBlockType, list[FunctionalBlock]] = defaultdict(list)
    for block in blocks:
        by_type[block.type].append(block)
    base_ids = [item.id for item in by_type[AnalysisBlockType.BASE_INFRASTRUCTURE]]
    runtime_ids = [item.id for item in by_type[AnalysisBlockType.RUNTIME_ENVIRONMENT]]
    shared_ids = [item.id for item in by_type[AnalysisBlockType.SHARED_SERVICES]]
    for block in blocks:
        if block.type == AnalysisBlockType.RUNTIME_ENVIRONMENT:
            fallback = base_ids
        elif block.type in {AnalysisBlockType.SHARED_SERVICES, AnalysisBlockType.OPERATIONS}:
            fallback = runtime_ids or base_ids
        elif block.type == AnalysisBlockType.APPLICATION:
            fallback = shared_ids or runtime_ids or base_ids
        else:
            fallback = []
        for prerequisite in fallback:
            add_dependency_if_acyclic(
                dependencies,
                dependent=block.id,
                prerequisite=prerequisite,
            )
    for block in blocks:
        block.after = sorted(dependencies[block.id])


def _dependency_endpoints(
    relation: AnalysisRelationType,
    source: str,
    target: str,
) -> tuple[str, str] | None:
    if relation == AnalysisRelationType.ORDERED_BEFORE:
        return target, source
    if relation in {AnalysisRelationType.REQUIRES, AnalysisRelationType.HEALTH_GATED_BY}:
        return source, target
    return None


def _endpoint_block(
    endpoint: str,
    component_blocks: dict[str, str],
    capability_blocks: dict[str, set[str]],
) -> str | None:
    if endpoint in component_blocks:
        return component_blocks[endpoint]
    providers = capability_blocks.get(endpoint, set())
    return next(iter(providers)) if len(providers) == 1 else None


def _block_levels(blocks: list[FunctionalBlock]) -> list[list[str]]:
    block_ids = {block.id for block in blocks}
    dependencies = {
        block.id: {item for item in block.after if item in block_ids} for block in blocks
    }
    return topological_levels(
        block_ids,
        dependencies,
        cycle_label="functional block graph",
    )


def _display_name(value: str) -> str:
    special = {
        "postgresql": "PostgreSQL",
        "postgres": "PostgreSQL",
        "minio": "MinIO",
        "activemq": "ActiveMQ",
        "clamav": "ClamAV",
        "softhsm": "SoftHSM",
        "rke2": "RKE2",
        "iam": "Keycloak",
        "smtp": "SMTP",
    }
    return special.get(value.casefold(), value.replace("_", " ").replace("-", " ").title())
