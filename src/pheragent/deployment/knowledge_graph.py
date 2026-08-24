from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import PurePosixPath

from pydantic import Field, model_validator

from .analysis_models import (
    AnalysisRelation,
    AnalysisRelationType,
    AnalysisSourceRef,
    CandidateComponent,
    DeploymentSignalBundle,
    ReferenceNode,
    ValidationSignal,
)
from .investigation_models import (
    InvestigationPurpose,
    InvestigationQuery,
    SourceScope,
)
from .models import ContractModel
from .retrieval import is_installer_path


class KnowledgeNodeKind(StrEnum):
    ARTIFACT = "artifact"
    COMPONENT = "component"
    OPERATION = "operation"
    CAPABILITY = "capability"
    VALIDATION = "validation"


class KnowledgeEdgeKind(StrEnum):
    SUPPORTED_BY = "supported_by"
    REFERENCES = "references"
    INVOKES = "invokes"
    INSTALLS = "installs"
    PROVIDES = "provides"
    REQUIRES = "requires"
    VALIDATES = "validates"


class KnowledgeGapKind(StrEnum):
    MISSING_OPERATION = "missing_operation"
    UNOWNED_ARTIFACT = "unowned_artifact"
    MISSING_PROVIDER = "missing_provider"
    MISSING_VALIDATION = "missing_validation"


class KnowledgeNode(ContractModel):
    id: str = Field(min_length=1)
    kind: KnowledgeNodeKind
    label: str = Field(min_length=1)
    source_refs: list[AnalysisSourceRef] = Field(default_factory=list)


class KnowledgeEdge(ContractModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    kind: KnowledgeEdgeKind
    evidence: list[AnalysisSourceRef] = Field(default_factory=list)


class KnowledgeGap(ContractModel):
    id: str = Field(min_length=1)
    kind: KnowledgeGapKind
    subject: str = Field(min_length=1)
    component_id: str | None = None
    terms: list[str] = Field(min_length=1, max_length=5)
    source_scope: SourceScope
    path_prefix: str | None = None
    evidence: list[AnalysisSourceRef] = Field(default_factory=list)


class DeploymentKnowledgeGraph(ContractModel):
    """A deterministic deployment graph whose claims retain their source evidence."""

    version: str = "0.1"
    projection_stage: str = "pre_retrieval"
    nodes: list[KnowledgeNode] = Field(default_factory=list)
    edges: list[KnowledgeEdge] = Field(default_factory=list)
    gaps: list[KnowledgeGap] = Field(default_factory=list)

    @model_validator(mode="after")
    def edges_reference_known_nodes(self) -> DeploymentKnowledgeGraph:
        known = {node.id for node in self.nodes}
        unknown = sorted(
            endpoint
            for edge in self.edges
            for endpoint in (edge.source, edge.target)
            if endpoint not in known
        )
        if unknown:
            raise ValueError("knowledge graph edge references unknown nodes: " + ", ".join(unknown))
        return self


def project_deployment_knowledge(
    signals: DeploymentSignalBundle,
    reference_nodes: list[ReferenceNode],
    reference_relations: list[AnalysisRelation],
) -> DeploymentKnowledgeGraph:
    """Project discovered facts without reading files, executing code, or calling an LLM."""
    builder = _GraphBuilder(reference_nodes)
    builder.add_components(signals)
    builder.add_relations([*signals.relations, *reference_relations])
    return builder.build()


def graph_gap_queries(
    graph: DeploymentKnowledgeGraph,
    *,
    limit: int = 2,
) -> tuple[InvestigationQuery, ...]:
    """Turn high-priority graph gaps into queries understood by the shared retriever."""
    return tuple(
        InvestigationQuery(
            id=f"graph-query-{index:02d}",
            purpose=_query_purpose(gap.kind),
            terms=gap.terms,
            source_scope=gap.source_scope,
            path_prefix=gap.path_prefix,
            component_id=gap.component_id,
            reason_code=f"GRAPH_{gap.kind.value.upper()}",
        )
        for index, gap in enumerate(graph.gaps[: max(limit, 0)], start=1)
    )


class _GraphBuilder:
    def __init__(self, references: list[ReferenceNode]) -> None:
        self.nodes: dict[str, KnowledgeNode] = {}
        self.edges: dict[tuple[str, str, KnowledgeEdgeKind], KnowledgeEdge] = {}
        self.gaps: list[KnowledgeGap] = []
        self.artifact_ids: dict[tuple[str, str], str] = {}
        self.reference_ids: dict[str, str] = {}
        self.component_ids: dict[str, str] = {}
        self.owned_artifacts: set[str] = set()
        for reference in sorted(references, key=lambda item: (item.repo_id, item.path)):
            source_ref = AnalysisSourceRef(repo_id=reference.repo_id, path=reference.path)
            artifact_id = self._artifact_node(source_ref)
            self.reference_ids[reference.id] = artifact_id
            self.reference_ids[f"{reference.repo_id}:{reference.path}"] = artifact_id

    def add_components(self, signals: DeploymentSignalBundle) -> None:
        for component in sorted(signals.candidate_components, key=lambda item: item.id):
            self._add_component(component)

    def add_relations(self, relations: list[AnalysisRelation]) -> None:
        provided_capabilities = {
            node.label.casefold()
            for node in self.nodes.values()
            if node.kind == KnowledgeNodeKind.CAPABILITY
            and any(
                edge.target == node.id and edge.kind == KnowledgeEdgeKind.PROVIDES
                for edge in self.edges.values()
            )
        }
        for relation in sorted(
            relations,
            key=lambda item: (item.source, item.target, item.relation.value),
        ):
            edge_kind = _edge_kind(relation.relation)
            if edge_kind is None:
                continue
            source = self._relation_endpoint(relation.source, relation.evidence)
            target = self._relation_endpoint(relation.target, relation.evidence)
            self._add_edge(source, target, edge_kind, relation.evidence)
            target_node = self.nodes[target]
            if (
                edge_kind == KnowledgeEdgeKind.REQUIRES
                and target_node.kind == KnowledgeNodeKind.CAPABILITY
                and target_node.label.casefold() not in provided_capabilities
            ):
                self._add_provider_gap(target_node, relation.evidence)

    def build(self) -> DeploymentKnowledgeGraph:
        self._add_unowned_artifact_gaps()
        return DeploymentKnowledgeGraph(
            nodes=sorted(self.nodes.values(), key=lambda item: item.id),
            edges=sorted(
                self.edges.values(),
                key=lambda item: (item.source, item.target, item.kind.value),
            ),
            gaps=sorted(
                self.gaps,
                key=lambda item: (_gap_priority(item.kind), item.subject.casefold(), item.id),
            ),
        )

    def _add_component(self, component: CandidateComponent) -> None:
        component_node = _stable_id("component", component.id)
        self.component_ids[component.id] = component_node
        self._add_node(
            component_node,
            KnowledgeNodeKind.COMPONENT,
            component.name,
            [component.source_ref],
        )
        artifact_node = self._artifact_node(component.source_ref)
        self.owned_artifacts.add(artifact_node)
        self._add_edge(
            component_node,
            artifact_node,
            KnowledgeEdgeKind.SUPPORTED_BY,
            [component.source_ref],
        )
        self._add_operation(component_node, component)
        self._add_capabilities(component_node, component)
        self._add_validations(component_node, component.validation_candidates)
        if component.deployable and not component.validation_candidates:
            self._add_component_gap(component, KnowledgeGapKind.MISSING_VALIDATION)

    def _add_operation(self, component_node: str, component: CandidateComponent) -> None:
        deployment = component.deployment
        if deployment is None:
            self._add_component_gap(component, KnowledgeGapKind.MISSING_OPERATION)
            return
        operation_ref = deployment.operation_source_ref or component.source_ref
        operation_id = _stable_id(
            "operation",
            f"{operation_ref.repo_id}:{operation_ref.path}:{deployment.command or ''}",
        )
        self._add_node(
            operation_id,
            KnowledgeNodeKind.OPERATION,
            deployment.command or deployment.entrypoint,
            [operation_ref],
        )
        self._add_edge(operation_id, component_node, KnowledgeEdgeKind.INSTALLS, [operation_ref])
        self.owned_artifacts.add(self._artifact_node(operation_ref))
        if not deployment.command:
            self._add_component_gap(component, KnowledgeGapKind.MISSING_OPERATION)

    def _add_capabilities(self, component_node: str, component: CandidateComponent) -> None:
        for capability in sorted(set(component.capabilities)):
            capability_node = _stable_id("capability", capability)
            self._add_node(
                capability_node,
                KnowledgeNodeKind.CAPABILITY,
                capability,
                [component.source_ref],
            )
            self._add_edge(
                component_node,
                capability_node,
                KnowledgeEdgeKind.PROVIDES,
                [component.source_ref],
            )

    def _add_validations(
        self,
        component_node: str,
        validations: list[ValidationSignal],
    ) -> None:
        for validation in validations:
            validation_id = _stable_id(
                "validation",
                f"{validation.source_ref.repo_id}:{validation.source_ref.path}:{validation.check}",
            )
            self._add_node(
                validation_id,
                KnowledgeNodeKind.VALIDATION,
                validation.check,
                [validation.source_ref],
            )
            self._add_edge(
                validation_id,
                component_node,
                KnowledgeEdgeKind.VALIDATES,
                [validation.source_ref],
            )

    def _add_component_gap(
        self,
        component: CandidateComponent,
        kind: KnowledgeGapKind,
    ) -> None:
        parent = PurePosixPath(component.source_ref.path).parent.as_posix()
        terms = list(
            dict.fromkeys(
                [
                    component.name,
                    *component.aliases,
                    *component.materialized_names,
                    "install deploy",
                ]
            )
        )[:5]
        self.gaps.append(
            KnowledgeGap(
                id=_stable_id("gap", f"{kind.value}:{component.id}"),
                kind=kind,
                subject=component.name,
                component_id=component.id,
                terms=terms,
                source_scope=SourceScope.BOTH,
                path_prefix=(
                    None
                    if kind == KnowledgeGapKind.MISSING_OPERATION or parent == "."
                    else parent
                ),
                evidence=[component.source_ref],
            )
        )

    def _add_provider_gap(
        self,
        capability: KnowledgeNode,
        evidence: list[AnalysisSourceRef],
    ) -> None:
        gap_id = _stable_id("gap", f"provider:{capability.label}")
        if any(gap.id == gap_id for gap in self.gaps):
            return
        self.gaps.append(
            KnowledgeGap(
                id=gap_id,
                kind=KnowledgeGapKind.MISSING_PROVIDER,
                subject=capability.label,
                terms=[capability.label, "service", "dependency"],
                source_scope=SourceScope.BOTH,
                evidence=evidence,
            )
        )

    def _add_unowned_artifact_gaps(self) -> None:
        for node in self.nodes.values():
            if node.kind != KnowledgeNodeKind.ARTIFACT or node.id in self.owned_artifacts:
                continue
            path = PurePosixPath(node.label)
            if not is_installer_path(node.label):
                continue
            self.gaps.append(
                KnowledgeGap(
                    id=_stable_id("gap", f"owner:{node.id}"),
                    kind=KnowledgeGapKind.UNOWNED_ARTIFACT,
                    subject=node.label,
                    terms=[path.parent.name or path.stem, path.name],
                    source_scope=SourceScope.REPOSITORY,
                    path_prefix=None if path.parent.as_posix() == "." else path.parent.as_posix(),
                    evidence=node.source_refs,
                )
            )

    def _artifact_node(self, reference: AnalysisSourceRef) -> str:
        key = (reference.repo_id, reference.path)
        node_id = self.artifact_ids.get(key)
        if node_id is None:
            node_id = _stable_id("artifact", f"{reference.repo_id}:{reference.path}")
            self.artifact_ids[key] = node_id
            self._add_node(node_id, KnowledgeNodeKind.ARTIFACT, reference.path, [reference])
        return node_id

    def _relation_endpoint(self, value: str, evidence: list[AnalysisSourceRef]) -> str:
        if value in self.component_ids:
            return self.component_ids[value]
        if value in self.reference_ids:
            return self.reference_ids[value]
        node_id = _stable_id("capability", value)
        self._add_node(node_id, KnowledgeNodeKind.CAPABILITY, value, evidence)
        return node_id

    def _add_node(
        self,
        node_id: str,
        kind: KnowledgeNodeKind,
        label: str,
        source_refs: list[AnalysisSourceRef],
    ) -> None:
        existing = self.nodes.get(node_id)
        refs = _unique_refs([*(existing.source_refs if existing else []), *source_refs])
        self.nodes[node_id] = KnowledgeNode(
            id=node_id,
            kind=kind,
            label=label,
            source_refs=refs,
        )

    def _add_edge(
        self,
        source: str,
        target: str,
        kind: KnowledgeEdgeKind,
        evidence: list[AnalysisSourceRef],
    ) -> None:
        key = (source, target, kind)
        existing = self.edges.get(key)
        grounded_evidence = evidence or [
            *self.nodes[source].source_refs,
            *self.nodes[target].source_refs,
        ]
        self.edges[key] = KnowledgeEdge(
            source=source,
            target=target,
            kind=kind,
            evidence=_unique_refs(
                [*(existing.evidence if existing else []), *grounded_evidence]
            ),
        )


def _edge_kind(relation: AnalysisRelationType) -> KnowledgeEdgeKind | None:
    return {
        AnalysisRelationType.REFERENCES: KnowledgeEdgeKind.REFERENCES,
        AnalysisRelationType.INVOKES: KnowledgeEdgeKind.INVOKES,
        AnalysisRelationType.REQUIRES: KnowledgeEdgeKind.REQUIRES,
        AnalysisRelationType.HEALTH_GATED_BY: KnowledgeEdgeKind.REQUIRES,
    }.get(relation)


def _query_purpose(kind: KnowledgeGapKind) -> InvestigationPurpose:
    return {
        KnowledgeGapKind.MISSING_OPERATION: InvestigationPurpose.DEPLOYMENT_ENTRYPOINTS,
        KnowledgeGapKind.UNOWNED_ARTIFACT: InvestigationPurpose.MISSING_COMPONENTS,
        KnowledgeGapKind.MISSING_PROVIDER: InvestigationPurpose.MISSING_COMPONENTS,
        KnowledgeGapKind.MISSING_VALIDATION: InvestigationPurpose.VALIDATION,
    }[kind]


def _gap_priority(kind: KnowledgeGapKind) -> int:
    return {
        KnowledgeGapKind.MISSING_OPERATION: 0,
        KnowledgeGapKind.UNOWNED_ARTIFACT: 1,
        KnowledgeGapKind.MISSING_PROVIDER: 2,
        KnowledgeGapKind.MISSING_VALIDATION: 3,
    }[kind]


def _stable_id(prefix: str, identity: str) -> str:
    return f"{prefix}-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"


def _unique_refs(references: list[AnalysisSourceRef]) -> list[AnalysisSourceRef]:
    by_key = {
        (ref.repo_id, ref.path, ref.start_line, ref.end_line): ref for ref in references
    }
    return [
        by_key[key]
        for key in sorted(by_key, key=lambda item: (item[0], item[1], item[2] or 0, item[3] or 0))
    ]
