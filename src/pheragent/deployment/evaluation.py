from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from .analysis_models import (
    AnalysisEvaluation,
    CandidateComponent,
    ComponentDisposition,
    DeploymentSignalBundle,
    FunctionalBlock,
    GoldDefinition,
    SignalStrength,
)
from .source_manager import AcquiredSource

SourceExists = Callable[[str, str], bool]

_FORBIDDEN_COMPONENT = re.compile(
    r"(?:^|\b)(?:command|step\s*\d*|kubectl|configmap|namespace|secret|configuration|"
    r"deployment)(?:$|\b)|\.(?:sh|ya?ml|json|ini)$|[/\\]",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DiscoveryClosure:
    component_count: int
    executable_component_count: int
    routed_component_count: int
    grounded_component_count: int
    orphan_component_ids: tuple[str, ...]

    @property
    def route_coverage(self) -> float:
        if not self.executable_component_count:
            return 1.0
        return self.routed_component_count / self.executable_component_count

    @property
    def closure(self) -> float:
        if not self.component_count:
            return 1.0
        closed = self.grounded_component_count - len(self.orphan_component_ids)
        return max(0.0, closed / self.component_count)

    @property
    def independently_grounded(self) -> bool:
        return (
            self.grounded_component_count == self.component_count and not self.orphan_component_ids
        )


def evaluate_discovery_closure(
    components: list[CandidateComponent],
    *,
    source_exists: SourceExists,
) -> DiscoveryClosure:
    relevant = [
        component
        for component in components
        if component.disposition
        not in {
            ComponentDisposition.IMPLEMENTATION_DETAIL,
            ComponentDisposition.PROVIDED_PREREQUISITE,
        }
    ]
    executable = [component for component in relevant if component.deployable]
    by_id = {component.id: component for component in components}
    routed = [
        component
        for component in executable
        if _has_route(component, by_id, source_exists=source_exists, visited=set())
    ]
    routed_ids = {component.id for component in routed}
    grounded = [
        component
        for component in relevant
        if source_exists(component.source_ref.repo_id, component.source_ref.path)
    ]
    return DiscoveryClosure(
        component_count=len(relevant),
        executable_component_count=len(executable),
        routed_component_count=len(routed),
        grounded_component_count=len(grounded),
        orphan_component_ids=tuple(
            sorted(component.id for component in executable if component.id not in routed_ids)
        ),
    )


def evaluate_functional_blocks(
    blocks: list[FunctionalBlock],
    signals: DeploymentSignalBundle,
    sources: dict[str, AcquiredSource],
    gold: GoldDefinition | None,
) -> AnalysisEvaluation:
    """Evaluate a generated block graph without exposing scoring details to its caller."""
    components = [component for block in blocks for component in block.components]

    def source_exists(repo_id: str, path: str) -> bool:
        return repo_id in sources and sources[repo_id].contains_file(path)

    closure = evaluate_discovery_closure(
        signals.candidate_components,
        source_exists=source_exists,
    )
    deployable = [component for component in components if component.deployable]
    grounded = [
        component
        for component in components
        if component.deploy is not None
        and source_exists(component.deploy.repo_id, component.deploy.ref)
    ]
    forbidden = [
        component for component in components if _FORBIDDEN_COMPONENT.search(component.name)
    ]
    total = len(components)
    evaluation = AnalysisEvaluation(
        candidate_component_count=len(signals.candidate_components),
        component_count=total,
        implementation_detail_count=sum(
            component.disposition == ComponentDisposition.IMPLEMENTATION_DETAIL
            for component in signals.candidate_components
        ),
        uncertain_component_count=sum(
            component.disposition == ComponentDisposition.UNCERTAIN
            for component in signals.candidate_components
        ),
        deployable_component_count=len(deployable),
        deployability_coverage=closure.route_coverage,
        grounded_component_rate=len(grounded) / total if total else 1.0,
        forbidden_component_count=len(forbidden),
        relation_count=len(signals.relations),
        source_derived_relation_count=sum(
            relation.strength != SignalStrength.LLM_INFERRED for relation in signals.relations
        ),
        hallucination_rate=1 - (len(grounded) / total if total else 1.0),
        discovery_closure=closure.closure,
        executable_route_coverage=closure.route_coverage,
        orphan_component_count=len(closure.orphan_component_ids),
        independently_grounded=closure.independently_grounded and not forbidden,
    )
    return _apply_gold_metrics(evaluation, blocks, gold) if gold else evaluation


def _apply_gold_metrics(
    evaluation: AnalysisEvaluation,
    blocks: list[FunctionalBlock],
    gold: GoldDefinition,
) -> AnalysisEvaluation:
    components = [component for block in blocks for component in block.components]
    predicted = {component.name.casefold(): component for component in components}
    expected = {name.casefold() for name in gold.expected_components}
    forbidden_names = {name.casefold() for name in gold.forbidden_components}
    correct = len(set(predicted) & expected)
    false_positive = len(set(predicted) - expected - forbidden_names) + len(
        set(predicted) & forbidden_names
    )
    classifications = [
        (
            (component := predicted.get(name.casefold())) is not None
            and (block := _component_block(blocks, component.id)).type
            == expected_classification.block_type
            and block.subtype == expected_classification.subtype
        )
        for name, expected_classification in gold.expected_classifications.items()
    ]
    entrypoints = [
        (
            (component := predicted.get(name.casefold())) is not None
            and component.deploy is not None
            and component.deploy.ref == expected_path
        )
        for name, expected_path in gold.expected_entrypoints.items()
    ]
    edge_precision, edge_recall = _edge_metrics(blocks, gold)
    return evaluation.model_copy(
        update={
            "component_precision": (
                correct / (correct + false_positive) if correct + false_positive else 1.0
            ),
            "component_recall": correct / len(expected) if expected else 1.0,
            "classification_accuracy": (
                sum(classifications) / len(classifications) if classifications else None
            ),
            "edge_precision": edge_precision,
            "edge_recall": edge_recall,
            "entrypoint_accuracy": (sum(entrypoints) / len(entrypoints) if entrypoints else None),
            "grouping_f1": _grouping_f1(blocks, gold.expected_groups),
        }
    )


def _edge_metrics(
    blocks: list[FunctionalBlock], gold: GoldDefinition
) -> tuple[float | None, float | None]:
    if not gold.expected_major_edges:
        return None, None
    block_by_id = {block.id: block for block in blocks}
    predicted = {
        (block_by_id[parent].type.value, block.type.value)
        for block in blocks
        for parent in block.after
        if parent in block_by_id
    }
    expected = {(edge.source, edge.target) for edge in gold.expected_major_edges}
    correct = len(predicted & expected)
    return (
        correct / len(predicted) if predicted else 0.0,
        correct / len(expected) if expected else 1.0,
    )


def _grouping_f1(blocks: list[FunctionalBlock], expected_groups: list[list[str]]) -> float | None:
    if not expected_groups:
        return None
    universe = {name.casefold() for group in expected_groups for name in group}
    expected_pairs = _pairs(expected_groups)
    predicted_pairs = _pairs(
        [
            [item.name for item in block.components if item.name.casefold() in universe]
            for block in blocks
        ]
    )
    true_positive = len(expected_pairs & predicted_pairs)
    precision = true_positive / len(predicted_pairs) if predicted_pairs else 0.0
    recall = true_positive / len(expected_pairs) if expected_pairs else 1.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _pairs(groups: list[list[str]]) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for group in groups:
        normalized = sorted(item.casefold() for item in group)
        result.update(
            (left, right)
            for index, left in enumerate(normalized)
            for right in normalized[index + 1 :]
        )
    return result


def _component_block(blocks: list[FunctionalBlock], component_id: str) -> FunctionalBlock:
    for block in blocks:
        if any(component.id == component_id for component in block.components):
            return block
    raise ValueError(f"component is not assigned to a block: {component_id}")


def _has_route(
    component: CandidateComponent,
    components: dict[str, CandidateComponent],
    *,
    source_exists: SourceExists,
    visited: set[str],
) -> bool:
    if component.id in visited:
        return False
    visited.add(component.id)
    if component.installed_by:
        owner = components.get(component.installed_by)
        return bool(
            owner and _has_route(owner, components, source_exists=source_exists, visited=visited)
        )
    deployment = component.deployment
    return bool(
        deployment
        and deployment.command
        and source_exists(component.source_ref.repo_id, deployment.entrypoint)
    )
