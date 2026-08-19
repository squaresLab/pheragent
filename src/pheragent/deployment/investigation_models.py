from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from .analysis_models import (
    COMPONENT_IDENTIFIER_PATTERN,
    AnalysisBlockType,
    AnalysisExecutor,
    AnalysisQuestion,
    ComponentDisposition,
    DeploymentActionType,
    SourceDisagreement,
)
from .models import IDENTIFIER_PATTERN, ContractModel


class SourcePurpose(StrEnum):
    REPOSITORY = "repository"
    DOCUMENTATION = "documentation"


class InvestigationPurpose(StrEnum):
    MISSING_COMPONENTS = "missing_components"
    DEPLOYMENT_ENTRYPOINTS = "deployment_entrypoints"
    DEPENDENCIES = "dependencies"
    EXTERNAL_REQUIREMENTS = "external_requirements"
    ALIASES = "aliases"
    VALIDATION = "validation"
    PROFILE_CONFLICTS = "profile_conflicts"
    DYNAMIC_BEHAVIOR = "dynamic_behavior"


class SourceScope(StrEnum):
    REPOSITORY = "repository"
    DOCUMENTATION = "documentation"
    BOTH = "both"


class EvidenceKind(StrEnum):
    PRIMARY_ROOT = "primary_root"
    COMPONENT_ENTRYPOINT = "component_entrypoint"
    SEARCH_RESULT = "search_result"
    UNREFERENCED_INSTALLER = "unreferenced_installer"
    CONFIGURATION_REFERENCE = "configuration_reference"
    DOCUMENTATION_REQUIREMENT = "documentation_requirement"
    DYNAMIC_DEPLOYMENT = "dynamic_deployment"
    INSTALLATION_ROUTE = "installation_route"


class EvidenceStrength(StrEnum):
    AUTHORITATIVE = "authoritative"
    STRONG = "strong"
    SUPPORTING = "supporting"
    WEAK = "weak"


class FactPredicate(StrEnum):
    DEPLOYS = "deploys"
    REQUIRES = "requires"
    ORDERED_BEFORE = "ordered_before"
    PROVIDES = "provides"
    CONSUMES = "consumes"
    CONFIGURES = "configures"
    INITIALIZES = "initializes"
    VALIDATES = "validates"
    RUNS_ON = "runs_on"
    ALIAS_OF = "alias_of"
    APPLIES_TO_PROFILE = "applies_to_profile"


class DeploymentClass(StrEnum):
    BASE_CLOUD = "base_infrastructure.cloud_infrastructure"
    BASE_ON_PREM = "base_infrastructure.on_prem_infrastructure"
    RUNTIME_CONTAINER = "runtime_environment.container_platform"
    RUNTIME_PLATFORM = "runtime_environment.platform_services"
    SHARED_DATA = "shared_services.data_services"
    SHARED_MESSAGING = "shared_services.messaging_integration"
    SHARED_IDENTITY = "shared_services.identity_security"
    SHARED_EXTERNAL = "shared_services.external_integration"
    APPLICATION_CORE = "application.core_application"
    APPLICATION_DOMAIN = "application.domain_service"
    OPERATIONS_OBSERVABILITY = "operations.observability"
    OPERATIONS_BACKUP = "operations.backup_recovery"
    OPERATIONS_GENERAL = "operations.operations"
    UNKNOWN = "unknown.unknown"

    @property
    def block_type(self) -> AnalysisBlockType:
        return AnalysisBlockType(self.value.split(".", 1)[0])

    @property
    def subtype(self) -> str:
        return self.value.split(".", 1)[1]


class ArtifactOutline(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_purpose: SourcePurpose
    path: str = Field(min_length=1)
    file_type: str = Field(min_length=1)
    roles: list[str] = Field(default_factory=list)
    references: int = Field(default=0, ge=0)
    referenced_by: int = Field(default=0, ge=0)
    materialized_names: list[str] = Field(default_factory=list, max_length=20)
    validation_count: int = Field(default=0, ge=0)
    context_hint: bool = False
    untrusted: bool = True


class InvestigationQuery(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    purpose: InvestigationPurpose
    terms: list[str] = Field(min_length=1, max_length=5)
    source_scope: SourceScope
    path_prefix: str | None = None
    component_id: str | None = Field(default=None, pattern=COMPONENT_IDENTIFIER_PATTERN)
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")

    @model_validator(mode="after")
    def validate_literal_terms(self) -> InvestigationQuery:
        if any(not term.strip() or len(term) > 80 for term in self.terms):
            raise ValueError("investigation terms must be non-empty and at most 80 characters")
        return self


class InvestigationPlan(ContractModel):
    queries: list[InvestigationQuery] = Field(default_factory=list, max_length=8)
    hypotheses: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def query_ids_are_unique(self) -> InvestigationPlan:
        ids = [query.id for query in self.queries]
        if len(ids) != len(set(ids)):
            raise ValueError("investigation query IDs must be unique")
        return self


class EvidenceObservation(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_id: str = Field(pattern=IDENTIFIER_PATTERN)
    source_purpose: SourcePurpose
    path: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    kind: EvidenceKind
    strength: EvidenceStrength
    summary: str = Field(min_length=1, max_length=500)
    excerpt: str | None = None
    excerpt_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    query_id: str | None = None
    prompt_injection_detected: bool = False
    dynamic_deployment: bool = False
    requires_resolution: bool = False
    blocks_execution: bool = False
    untrusted: bool = True

    @model_validator(mode="after")
    def validate_lines(self) -> EvidenceObservation:
        if self.end_line < self.start_line:
            raise ValueError("evidence end line must not precede start line")
        return self


class InvestigationFact(ContractModel):
    subject: str = Field(min_length=1, max_length=120)
    predicate: FactPredicate
    object: str = Field(min_length=1, max_length=240)
    evidence_ids: list[str] = Field(min_length=1, max_length=6)
    confidence: float = Field(ge=0.0, le=1.0)


class ComponentClassificationGroup(ContractModel):
    component_ids: list[str] = Field(min_length=1, max_length=80)
    disposition: ComponentDisposition
    classification: DeploymentClass
    domain: str | None = None
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    confidence: float = Field(ge=0.0, le=1.0)
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$", max_length=96)


class ComponentRename(ContractModel):
    component_id: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1, max_length=120)
    aliases: list[str] = Field(default_factory=list, max_length=8)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)


class DeploymentActionBinding(ContractModel):
    candidate_id: str = Field(min_length=1)
    owner_component_id: str = Field(min_length=1)
    action_type: DeploymentActionType
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    confidence: float = Field(ge=0.0, le=1.0)
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$", max_length=96)


class ImpliedEntity(ContractModel):
    canonical_name: str = Field(min_length=1, max_length=120)
    aliases: list[str] = Field(default_factory=list, max_length=8)
    disposition: ComponentDisposition
    classification: DeploymentClass
    capabilities: list[str] = Field(default_factory=list, max_length=8)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    entrypoint_evidence_id: str | None = None
    executor: AnalysisExecutor = AnalysisExecutor.UNKNOWN
    required_for_initial_deployment: bool
    confidence: float = Field(ge=0.0, le=1.0)


class InvestigationSynthesis(ContractModel):
    classification_groups: list[ComponentClassificationGroup] = Field(
        default_factory=list,
        max_length=30,
    )
    deployment_actions: list[DeploymentActionBinding] = Field(
        default_factory=list,
        max_length=12,
    )
    renames: list[ComponentRename] = Field(default_factory=list, max_length=12)
    implied_entities: list[ImpliedEntity] = Field(default_factory=list, max_length=8)
    facts: list[InvestigationFact] = Field(default_factory=list, max_length=8)
    disagreements: list[SourceDisagreement] = Field(default_factory=list, max_length=6)
    unresolved: list[AnalysisQuestion] = Field(default_factory=list, max_length=8)


def default_investigation_plan() -> InvestigationPlan:
    """Provide diverse fallback searches; these also prevent known-name search bias."""
    return InvestigationPlan(
        queries=[
            InvestigationQuery(
                id="query-requirements",
                purpose=InvestigationPurpose.EXTERNAL_REQUIREMENTS,
                terms=["prerequisite", "require", "external service"],
                source_scope=SourceScope.DOCUMENTATION,
                reason_code="INDEPENDENT_REQUIREMENT_SWEEP",
            ),
            InvestigationQuery(
                id="query-dependencies",
                purpose=InvestigationPurpose.DEPENDENCIES,
                terms=["depends_on", "dependsOn", "required_by"],
                source_scope=SourceScope.BOTH,
                reason_code="INDEPENDENT_DEPENDENCY_SWEEP",
            ),
            InvestigationQuery(
                id="query-endpoints",
                purpose=InvestigationPurpose.MISSING_COMPONENTS,
                terms=["_HOST", "_URL", "ENDPOINT", "BOOTSTRAP_SERVERS"],
                source_scope=SourceScope.REPOSITORY,
                reason_code="INDEPENDENT_ENDPOINT_SWEEP",
            ),
            InvestigationQuery(
                id="query-validation",
                purpose=InvestigationPurpose.VALIDATION,
                terms=["readiness", "healthcheck", "rollout status"],
                source_scope=SourceScope.BOTH,
                reason_code="INDEPENDENT_VALIDATION_SWEEP",
            ),
            InvestigationQuery(
                id="query-profile",
                purpose=InvestigationPurpose.PROFILE_CONFLICTS,
                terms=["cloud", "on-prem", "docker", "kubernetes"],
                source_scope=SourceScope.BOTH,
                reason_code="INDEPENDENT_PROFILE_SWEEP",
            ),
        ]
    )
