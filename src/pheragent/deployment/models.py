from __future__ import annotations

from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import (
    BlockType,
    ComponentRelationType,
    Confidence,
    DeterministicFindingKind,
    Executor,
    FactPredicate,
    FindingSeverity,
    InventoryCategory,
    OperationPhase,
    ProvenanceOrigin,
    ReviewStatus,
    SourceKind,
    TargetKind,
    ValidationLevel,
)

IDENTIFIER_PATTERN = r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provenance(ContractModel):
    origin: ProvenanceOrigin
    confidence: Confidence
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_evidence_for_extracted_claims(self) -> Self:
        if self.origin == ProvenanceOrigin.EXTRACTED and not self.evidence_refs:
            raise ValueError("extracted provenance requires at least one evidence reference")
        return self


class SourceSpec(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    kind: SourceKind
    location: str = Field(min_length=1)
    revision: str | None = None
    root_path: str = "."
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)


class SourcesConfig(ContractModel):
    system: str = Field(min_length=1)
    sources: list[SourceSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_source_ids(self) -> Self:
        _require_unique((source.id for source in self.sources), "source id")
        return self


class SourceRecord(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    kind: SourceKind
    repository: str = Field(min_length=1)
    revision: str | None = None
    root_path: str = "."
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class SourceManifestEntry(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    kind: SourceKind
    location: str = Field(min_length=1)
    requested_revision: str | None = None
    resolved_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{40,64}$")
    root_path: str = "."
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def require_revision_for_git(self) -> Self:
        if self.kind == SourceKind.GIT and self.resolved_revision is None:
            raise ValueError("Git manifest entries require a resolved revision")
        if self.kind != SourceKind.GIT and self.resolved_revision is not None:
            raise ValueError("only Git manifest entries may have a resolved revision")
        return self


class SourceManifest(ContractModel):
    manifest_version: str = Field(pattern=r"^0\.1$")
    system: str = Field(min_length=1)
    sources: list[SourceManifestEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_source_ids(self) -> Self:
        _require_unique((source.id for source in self.sources), "source id")
        return self


class EvidenceRecord(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_id: str = Field(pattern=IDENTIFIER_PATTERN)
    path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    heading: str | None = None
    excerpt_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    excerpt: str | None = None

    @model_validator(mode="after")
    def validate_line_range(self) -> Self:
        if self.end_line < self.start_line:
            raise ValueError("evidence end_line must be greater than or equal to start_line")
        return self


class ArtifactMetadata(ContractModel):
    artifact_id: str = Field(pattern=IDENTIFIER_PATTERN)
    system_name: str = Field(min_length=1)
    system_version: str = "unknown"
    profile: str = "unknown"
    generated_at: datetime
    generator_version: str = Field(min_length=1)


class ScopeClaim(ContractModel):
    description: str = Field(min_length=1)
    provenance: Provenance


class DeploymentScope(ContractModel):
    includes: list[ScopeClaim] = Field(default_factory=list)
    excludes: list[ScopeClaim] = Field(default_factory=list)


class Target(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    kind: TargetKind
    description: str = Field(min_length=1)
    established_by: str | None = None
    provenance: Provenance


class SourceArtifactReference(ContractModel):
    source_id: str = Field(pattern=IDENTIFIER_PATTERN)
    path: str = Field(min_length=1)


class Requirement(ContractModel):
    capability: str = Field(min_length=1)
    provider_block: str | None = None
    mandatory: bool = True
    reference: str | None = None
    provenance: Provenance


class Capability(ContractModel):
    capability: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance


class Operation(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    phase: OperationPhase
    executor: Executor
    component: str | None = None
    source_artifact: SourceArtifactReference | None = None
    command: str | None = None
    internal_requires: list[str] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    provenance: Provenance

    @model_validator(mode="after")
    def validate_command_provenance(self) -> Self:
        if self.command and self.provenance.origin != ProvenanceOrigin.EXTRACTED:
            raise ValueError("operation commands may only use extracted provenance")
        if self.command and self.source_artifact is None:
            raise ValueError("operation commands require a source_artifact")
        return self


class DeploymentValidation(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    level: ValidationLevel
    description: str = Field(min_length=1)
    blocking: bool = True
    executor: Executor
    source_artifact: SourceArtifactReference | None = None
    provenance: Provenance
    review_status: ReviewStatus | None = None

    @model_validator(mode="after")
    def validate_synthesized_review_status(self) -> Self:
        if (
            self.provenance.origin == ProvenanceOrigin.SYNTHESIZED
            and self.review_status != ReviewStatus.PROPOSED
        ):
            raise ValueError("synthesized validations must have review_status=proposed")
        return self


class Component(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    implementation: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    operations: list[str] = Field(default_factory=list)
    provenance: Provenance


class ComponentRelation(ContractModel):
    source: str = Field(pattern=IDENTIFIER_PATTERN)
    relation: ComponentRelationType
    target: str = Field(pattern=IDENTIFIER_PATTERN)
    provenance: Provenance


class RecoveryBoundary(ContractModel):
    local_components: list[str] = Field(default_factory=list)
    upstream_capabilities: list[str] = Field(default_factory=list)


class DeploymentBlock(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    name: str = Field(min_length=1)
    type: BlockType
    target: str | None = None
    purpose: str = Field(min_length=1)
    grouping_rationale: str = Field(min_length=1)
    grouping_provenance: Provenance
    components: list[Component] = Field(default_factory=list)
    component_relations: list[ComponentRelation] = Field(default_factory=list)
    requires: list[Requirement] = Field(default_factory=list)
    provides: list[Capability] = Field(default_factory=list)
    operations: list[Operation] = Field(default_factory=list)
    validations: list[DeploymentValidation] = Field(default_factory=list)
    recovery_boundary: RecoveryBoundary = Field(default_factory=RecoveryBoundary)
    provenance: Provenance

    @model_validator(mode="after")
    def validate_internal_references(self) -> Self:
        component_ids = _require_unique(
            (component.id for component in self.components),
            f"component id in block {self.id}",
        )
        operation_ids = _require_unique(
            (operation.id for operation in self.operations),
            f"operation id in block {self.id}",
        )
        _require_unique(
            (validation.id for validation in self.validations),
            f"validation id in block {self.id}",
        )
        for component in self.components:
            _require_known(component.operations, operation_ids, "operation", component.id)
        for operation in self.operations:
            if operation.component and operation.component not in component_ids:
                raise ValueError(
                    f"operation {operation.id} references unknown component {operation.component}"
                )
            _require_known(operation.internal_requires, operation_ids, "operation", operation.id)
        for relation in self.component_relations:
            _require_known((relation.source, relation.target), component_ids, "component", self.id)
        _require_known(
            self.recovery_boundary.local_components,
            component_ids,
            "component",
            f"recovery boundary for {self.id}",
        )
        return self


class ArtifactPolicies(ContractModel):
    commands_are_declarative: bool = True
    secret_values_forbidden: bool = True


class UnresolvedQuestion(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    question: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    related_block_ids: list[str] = Field(default_factory=list)
    related_subjects: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class UnresolvedQuestionsDocument(ContractModel):
    questions_version: str = Field(default="0.1", pattern=r"^0\.1$")
    questions: list[UnresolvedQuestion] = Field(default_factory=list)


class DeploymentArtifact(ContractModel):
    artifact_version: str = Field(pattern=r"^0\.1$")
    metadata: ArtifactMetadata
    sources: list[SourceRecord] = Field(min_length=1)
    scope: DeploymentScope
    targets: list[Target] = Field(default_factory=list)
    blocks: list[DeploymentBlock] = Field(default_factory=list)
    policies: ArtifactPolicies = Field(default_factory=ArtifactPolicies)
    unresolved_questions: list[UnresolvedQuestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_cross_references(self) -> Self:
        source_ids = _require_unique((source.id for source in self.sources), "source id")
        target_ids = _require_unique((target.id for target in self.targets), "target id")
        block_ids = _require_unique((block.id for block in self.blocks), "block id")
        _require_unique(
            (question.id for question in self.unresolved_questions), "unresolved question id"
        )

        for target in self.targets:
            if target.established_by and target.established_by not in block_ids:
                raise ValueError(
                    f"target {target.id} references unknown establishing block "
                    f"{target.established_by}"
                )
        for block in self.blocks:
            if block.target and block.target not in target_ids:
                raise ValueError(f"block {block.id} references unknown target {block.target}")
            for requirement in block.requires:
                if requirement.provider_block and requirement.provider_block not in block_ids:
                    raise ValueError(
                        f"block {block.id} references unknown provider block "
                        f"{requirement.provider_block}"
                    )
            for item in (*block.operations, *block.validations):
                if item.source_artifact and item.source_artifact.source_id not in source_ids:
                    raise ValueError(
                        f"{type(item).__name__.lower()} {item.id} references unknown source "
                        f"{item.source_artifact.source_id}"
                    )
        for question in self.unresolved_questions:
            _require_known(question.related_block_ids, block_ids, "block", question.id)
        return self


class DeploymentFact(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    subject: str = Field(min_length=1)
    subject_aliases: list[str] | None = None
    predicate: FactPredicate
    object: str = Field(min_length=1)
    object_aliases: list[str] | None = None
    provenance: Provenance


class ExtractedFactClaim(ContractModel):
    subject: str = Field(min_length=1)
    predicate: FactPredicate
    object: str = Field(min_length=1)
    confidence: Confidence
    evidence_refs: list[str] = Field(min_length=1)


class ExtractedQuestionClaim(ContractModel):
    question: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    related_subjects: list[str]
    evidence_refs: list[str] = Field(min_length=1)


class FactExtractionResponse(ContractModel):
    facts: list[ExtractedFactClaim]
    unresolved_questions: list[ExtractedQuestionClaim]


class FactExtractionReport(ContractModel):
    extraction_version: str = Field(default="0.1", pattern=r"^0\.1$")
    requested_extractor: str = Field(pattern=r"^(auto|deterministic|llm)$")
    used_extractor: str = Field(pattern=r"^(deterministic|deterministic\+llm)$")
    model: str | None = None
    chunk_count: int = Field(ge=0)
    selected_chunk_count: int = Field(ge=0)
    skipped_chunk_count: int = Field(ge=0)
    max_llm_requests: int = Field(ge=0)
    llm_requests_made: int = Field(ge=0)
    fact_count: int = Field(ge=0)
    unresolved_question_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)


class DependencyGraphEdge(ContractModel):
    provider_block: str = Field(pattern=IDENTIFIER_PATTERN)
    consumer_block: str = Field(pattern=IDENTIFIER_PATTERN)
    capability: str = Field(min_length=1)
    mandatory: bool = True


class DependencyGraph(ContractModel):
    graph_version: str = Field(default="0.1", pattern=r"^0\.1$")
    nodes: list[str] = Field(default_factory=list)
    edges: list[DependencyGraphEdge] = Field(default_factory=list)
    unmatched_capabilities: list[str] = Field(default_factory=list)
    cycle: list[str] = Field(default_factory=list)


class InventoryEntry(ContractModel):
    source_id: str = Field(pattern=IDENTIFIER_PATTERN)
    path: str = Field(min_length=1)
    category: InventoryCategory
    size_bytes: int = Field(ge=0)
    selected: bool = False
    inspected: bool = False
    skip_reason: str | None = None
    relevance_score: float = Field(default=0.0, ge=0.0)
    parser: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_skip_reason(self) -> Self:
        if not self.selected and not self.skip_reason:
            raise ValueError("unselected inventory entries require skip_reason")
        if self.selected and self.skip_reason:
            raise ValueError("selected inventory entries must not have skip_reason")
        if self.inspected and not self.selected:
            raise ValueError("inspected inventory entries must be selected")
        if self.parser and not self.inspected:
            raise ValueError("inventory parser requires inspected=true")
        return self


class RepositoryInventory(ContractModel):
    inventory_version: str = Field(default="0.1", pattern=r"^0\.1$")
    detected_technologies: list[str] = Field(default_factory=list)
    entries: list[InventoryEntry] = Field(default_factory=list)


class DeterministicFinding(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_id: str = Field(pattern=IDENTIFIER_PATTERN)
    path: str = Field(min_length=1)
    category: InventoryCategory
    kind: DeterministicFindingKind
    name: str = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(min_length=1)


class ExtractionChunk(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_id: str = Field(pattern=IDENTIFIER_PATTERN)
    path: str = Field(min_length=1)
    finding_ids: list[str] = Field(min_length=1)
    evidence: list[EvidenceRecord] = Field(min_length=1)
    findings: list[DeterministicFinding] = Field(min_length=1)


class ValidationFinding(ContractModel):
    severity: FindingSeverity
    code: str = Field(pattern=r"^[a-z0-9_]+$")
    message: str = Field(min_length=1)
    location: str | None = None


class ArtifactValidationResult(ContractModel):
    valid: bool
    findings: list[ValidationFinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def align_valid_with_findings(self) -> Self:
        has_error = any(finding.severity == FindingSeverity.ERROR for finding in self.findings)
        if self.valid == has_error:
            raise ValueError("valid must be false exactly when error findings are present")
        return self


def _require_unique(values: Any, label: str) -> set[str]:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {label}: {value}")
        seen.add(value)
    return seen


def _require_known(values: Any, known: set[str], label: str, owner: str) -> None:
    for value in values:
        if value not in known:
            raise ValueError(f"{owner} references unknown {label} {value}")
