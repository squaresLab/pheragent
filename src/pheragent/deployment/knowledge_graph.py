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
)
from .investigation_models import (
    InvestigationPurpose,
    InvestigationQuery,
    SourceScope,
)
from .models import ContractModel


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
        self.endpoint_ids: dict[str, str] = {}
        self.component_ids: dict[str, str] = {}
        for reference in sorted(references, key=lambda item: (item.repo_id, item.path)):
            source_ref = AnalysisSourceRef(repo_id=reference.repo_id, path=reference.path)
            artifact_id = self._artifact(source_ref)
            self.endpoint_ids.update(
                {reference.id: artifact_id, f"{reference.repo_id}:{reference.path}": artifact_id}
            )

    def add_components(self, signals: DeploymentSignalBundle) -> None:
        for component in sorted(signals.candidate_components, key=lambda item: item.id):
            self._component(component)

    def add_relations(self, relations: list[AnalysisRelation]) -> None:
        provided = {
            edge.target for edge in self.edges.values() if edge.kind == KnowledgeEdgeKind.PROVIDES
        }
        for relation in sorted(
            relations,
            key=lambda item: (item.source, item.target, item.relation.value),
        ):
            kind = _edge_kind(relation.relation)
            if kind is None:
                continue
            source = self._endpoint(relation.source, relation.evidence)
            target = self._endpoint(relation.target, relation.evidence)
            self._edge(source, target, kind, relation.evidence)
            if kind == KnowledgeEdgeKind.REQUIRES and target not in provided:
                node = self.nodes[target]
                if node.kind == KnowledgeNodeKind.CAPABILITY:
                    self._gap(
                        KnowledgeGapKind.MISSING_PROVIDER,
                        node.label,
                        [node.label, "service", "dependency"],
                        relation.evidence,
                    )

    def build(self) -> DeploymentKnowledgeGraph:
        return DeploymentKnowledgeGraph(
            nodes=sorted(self.nodes.values(), key=lambda item: item.id),
            edges=sorted(
                self.edges.values(),
                key=lambda item: (item.source, item.target, item.kind.value),
            ),
            gaps=sorted(
                self.gaps,
                key=lambda item: (
                    item.kind != KnowledgeGapKind.MISSING_OPERATION,
                    item.subject.casefold(),
                    item.id,
                ),
            ),
        )

    def _component(self, component: CandidateComponent) -> None:
        component_id = _stable_id("component", component.id)
        self.component_ids[component.id] = component_id
        self._node(
            component_id,
            KnowledgeNodeKind.COMPONENT,
            component.name,
            [component.source_ref],
        )
        artifact_id = self._artifact(component.source_ref)
        self._edge(
            component_id,
            artifact_id,
            KnowledgeEdgeKind.SUPPORTED_BY,
            [component.source_ref],
        )
        deployment = component.deployment
        if deployment is None:
            self._component_gap(component, KnowledgeGapKind.MISSING_OPERATION)
        else:
            reference = deployment.operation_source_ref or component.source_ref
            operation_id = _stable_id(
                "operation",
                f"{reference.repo_id}:{reference.path}:{deployment.command or ''}",
            )
            self._node(
                operation_id,
                KnowledgeNodeKind.OPERATION,
                deployment.command or deployment.entrypoint,
                [reference],
            )
            self._edge(operation_id, component_id, KnowledgeEdgeKind.INSTALLS, [reference])
            self._artifact(reference)
            if not deployment.command:
                self._component_gap(component, KnowledgeGapKind.MISSING_OPERATION)
        for capability in sorted(set(component.capabilities)):
            capability_id = _stable_id("capability", capability)
            self._node(
                capability_id, KnowledgeNodeKind.CAPABILITY, capability, [component.source_ref]
            )
            self._edge(
                component_id, capability_id, KnowledgeEdgeKind.PROVIDES, [component.source_ref]
            )
        for validation in component.validation_candidates:
            validation_id = _stable_id(
                "validation",
                f"{validation.source_ref.repo_id}:{validation.source_ref.path}:{validation.check}",
            )
            self._node(
                validation_id,
                KnowledgeNodeKind.VALIDATION,
                validation.check,
                [validation.source_ref],
            )
            self._edge(
                validation_id,
                component_id,
                KnowledgeEdgeKind.VALIDATES,
                [validation.source_ref],
            )
        if component.deployable and not component.validation_candidates:
            self._component_gap(component, KnowledgeGapKind.MISSING_VALIDATION)

    def _component_gap(
        self,
        component: CandidateComponent,
        kind: KnowledgeGapKind,
    ) -> None:
        parent = PurePosixPath(component.source_ref.path).parent.as_posix()
        self._gap(
            kind,
            component.name,
            list(
                dict.fromkeys(
                    [
                        component.name,
                        *component.aliases,
                        *component.materialized_names,
                        "install deploy",
                    ]
                )
            )[:5],
            [component.source_ref],
            component_id=component.id,
            path_prefix=(
                None
                if kind == KnowledgeGapKind.MISSING_OPERATION or parent == "."
                else parent
            ),
        )

    def _gap(
        self,
        kind: KnowledgeGapKind,
        subject: str,
        terms: list[str],
        evidence: list[AnalysisSourceRef],
        *,
        component_id: str | None = None,
        path_prefix: str | None = None,
    ) -> None:
        gap_id = _stable_id("gap", f"{kind.value}:{component_id or subject}")
        if any(gap.id == gap_id for gap in self.gaps):
            return
        self.gaps.append(
            KnowledgeGap(
                id=gap_id,
                kind=kind,
                subject=subject,
                component_id=component_id,
                terms=terms[:5],
                source_scope=SourceScope.BOTH,
                path_prefix=path_prefix,
                evidence=evidence,
            )
        )

    def _artifact(self, reference: AnalysisSourceRef) -> str:
        key = (reference.repo_id, reference.path)
        node_id = self.artifact_ids.get(key)
        if node_id is None:
            node_id = _stable_id("artifact", f"{reference.repo_id}:{reference.path}")
            self.artifact_ids[key] = node_id
            self._node(node_id, KnowledgeNodeKind.ARTIFACT, reference.path, [reference])
        return node_id

    def _endpoint(self, value: str, evidence: list[AnalysisSourceRef]) -> str:
        if value in self.component_ids:
            return self.component_ids[value]
        if value in self.endpoint_ids:
            return self.endpoint_ids[value]
        node_id = _stable_id("capability", value)
        self._node(node_id, KnowledgeNodeKind.CAPABILITY, value, evidence)
        return node_id

    def _node(
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

    def _edge(
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
        KnowledgeGapKind.MISSING_PROVIDER: InvestigationPurpose.MISSING_COMPONENTS,
        KnowledgeGapKind.MISSING_VALIDATION: InvestigationPurpose.VALIDATION,
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
