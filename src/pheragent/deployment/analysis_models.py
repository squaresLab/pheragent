from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from .models import IDENTIFIER_PATTERN, ContractModel

BLOCK_IDENTIFIER_PATTERN = rf"^(?:{IDENTIFIER_PATTERN[1:-1]}|B[0-9]+)$"
COMPONENT_IDENTIFIER_PATTERN = r"^C[0-9]+_[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$"


class AnalysisBlockType(StrEnum):
    BASE_INFRASTRUCTURE = "base_infrastructure"
    RUNTIME_ENVIRONMENT = "runtime_environment"
    SHARED_SERVICES = "shared_services"
    APPLICATION = "application"
    OPERATIONS = "operations"


class ArtifactRole(StrEnum):
    DEPLOYMENT_GUIDE = "deployment_guide"
    DEPLOYMENT_INDEX = "deployment_index"
    ORCHESTRATOR = "orchestrator"
    COMPONENT_INSTALLER = "component_installer"
    DESIRED_STATE = "desired_state"
    CONFIGURATION = "configuration"
    INITIALIZATION = "initialization"
    VALIDATION = "validation"
    OPERATIONS = "operations"
    HUMAN_PREREQUISITE = "human_prerequisite"
    UNKNOWN = "unknown"


class AnalysisRelationType(StrEnum):
    REFERENCES = "references"
    INVOKES = "invokes"
    CONTAINS = "contains"
    COMPOSES = "composes"
    ORDERED_BEFORE = "ordered_before"
    REQUIRES = "requires"
    HEALTH_GATED_BY = "health_gated_by"
    DEPLOYS_TO = "deploys_to"
    CONFIGURES = "configures"
    INITIALIZES = "initializes"


class SignalStrength(StrEnum):
    EXPLICIT_DEPENDENCY = "explicit_dependency"
    EXPLICIT_HEALTH_DEPENDENCY = "explicit_health_dependency"
    DECLARED_ORDER = "declared_order"
    OBSERVED_EXECUTION_ORDER = "observed_execution_order"
    STRUCTURAL_REFERENCE = "structural_reference"
    LLM_INFERRED = "llm_inferred"


class AnalysisExecutor(StrEnum):
    SHELL = "shell"
    TERRAFORM = "terraform"
    ANSIBLE = "ansible"
    HELM = "helm"
    KUSTOMIZE = "kustomize"
    DOCKER_COMPOSE = "docker_compose"
    KUBERNETES = "kubernetes"
    GITHUB_ACTIONS = "github_actions"
    GITOPS = "gitops"
    UNKNOWN = "unknown"


class DeploymentSelection(ContractModel):
    version: str | None = None
    profile: str | None = None


class ProvidedBlock(ContractModel):
    id: str = Field(pattern=BLOCK_IDENTIFIER_PATTERN)
    type: AnalysisBlockType
    subtype: str = Field(min_length=1)
    implementation: str | None = None
    state: str = "provided"
    provides: list[str] = Field(default_factory=list)
    after: list[str] = Field(default_factory=list)


class ContextHints(ContractModel):
    documentation: list[str] = Field(default_factory=list)


class DeploymentContext(ContractModel):
    system: str = Field(min_length=1)
    deployment: DeploymentSelection = Field(default_factory=DeploymentSelection)
    provided_blocks: list[ProvidedBlock] = Field(default_factory=list)
    hints: ContextHints = Field(default_factory=ContextHints)

    @model_validator(mode="after")
    def validate_provided_blocks(self) -> DeploymentContext:
        ids = [block.id for block in self.provided_blocks]
        if len(ids) != len(set(ids)):
            raise ValueError("deployment context contains duplicate provided block IDs")
        known_ids = set(ids)
        for block in self.provided_blocks:
            unknown = sorted(set(block.after) - known_ids)
            if unknown:
                raise ValueError(
                    f"provided block {block.id} references unknown block IDs: "
                    f"{', '.join(unknown)}"
                )
            if block.id in block.after:
                raise ValueError(f"provided block {block.id} cannot depend on itself")
        remaining = {block.id: set(block.after) for block in self.provided_blocks}
        emitted: set[str] = set()
        while len(emitted) < len(remaining):
            ready = {
                block_id
                for block_id, dependencies in remaining.items()
                if block_id not in emitted and dependencies <= emitted
            }
            if not ready:
                raise ValueError("provided block dependencies contain a cycle")
            emitted.update(ready)
        _validate_profile_and_infrastructure(self.deployment, self.provided_blocks)
        return self


def _validate_profile_and_infrastructure(
    deployment: DeploymentSelection,
    blocks: list[ProvidedBlock],
) -> None:
    profile = (deployment.profile or "").casefold().replace("_", "-")
    base_blocks = [
        block for block in blocks if block.type == AnalysisBlockType.BASE_INFRASTRUCTURE
    ]
    cloud_providers = {"aws", "azure", "gcp", "google-cloud", "openstack"}
    for block in base_blocks:
        subtype = block.subtype.casefold().replace("_", "-")
        implementation = (block.implementation or "").casefold().replace("_", "-")
        if profile.startswith("on-prem") and (
            subtype == "cloud-infrastructure" or implementation in cloud_providers
        ):
            raise ValueError(
                "on-premises deployment profile conflicts with cloud base infrastructure "
                f"in block {block.id}"
            )
        if profile in {"cloud", *cloud_providers} and subtype == "on-prem-infrastructure":
            raise ValueError(
                "cloud deployment profile conflicts with on-premises base infrastructure "
                f"in block {block.id}"
            )
        if (
            profile in cloud_providers
            and implementation in cloud_providers
            and implementation != profile
            and not ({implementation, profile} <= {"gcp", "google-cloud"})
        ):
            raise ValueError(
                f"deployment profile {deployment.profile} conflicts with "
                f"{block.implementation} infrastructure in block {block.id}"
            )


class AnalysisSourceRef(ContractModel):
    repo_id: str = Field(pattern=IDENTIFIER_PATTERN)
    path: str = Field(min_length=1)
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)


class RepositoryFile(ContractModel):
    repo_id: str = Field(pattern=IDENTIFIER_PATTERN)
    path: str = Field(min_length=1)
    file_type: str = Field(min_length=1)
    size: int = Field(ge=0)
    version_context: str | None = None
    profile_context: str | None = None
    deployment_roles: list[ArtifactRole] = Field(default_factory=list)


class ReferenceNode(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    repo_id: str = Field(pattern=IDENTIFIER_PATTERN)
    path: str = Field(min_length=1)
    node_type: str = Field(min_length=1)
    roles: list[ArtifactRole] = Field(default_factory=list)


class AnalysisRelation(ContractModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    relation: AnalysisRelationType
    strength: SignalStrength
    evidence: list[AnalysisSourceRef] = Field(default_factory=list)


class ValidationSignal(ContractModel):
    subject: str = Field(min_length=1)
    check: str = Field(min_length=1)
    readiness: str | None = None
    source_ref: AnalysisSourceRef


class ComponentDeployment(ContractModel):
    executor: AnalysisExecutor
    entrypoint: str = Field(min_length=1)


class ComponentClassification(ContractModel):
    block_type: AnalysisBlockType
    subtype: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    domain: str | None = None


class CandidateComponent(ContractModel):
    id: str = Field(pattern=COMPONENT_IDENTIFIER_PATTERN)
    name: str = Field(min_length=1)
    implementation: str | None = None
    deployable: bool = True
    external: bool = False
    aliases: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    source_ref: AnalysisSourceRef
    deployment: ComponentDeployment | None = None
    validation_candidates: list[ValidationSignal] = Field(default_factory=list)
    classification: ComponentClassification


class DeploymentRoot(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    role: ArtifactRole
    source_ref: AnalysisSourceRef
    score: float
    selection_reasons: list[str] = Field(default_factory=list)


class StageSignal(ContractModel):
    id: str = Field(pattern=IDENTIFIER_PATTERN)
    entrypoint: AnalysisSourceRef
    component_ids: list[str] = Field(default_factory=list)


class ExternalRequirement(ContractModel):
    name: str = Field(min_length=1)
    required_by: str | None = None
    source_ref: AnalysisSourceRef


class AnalysisQuestion(ContractModel):
    question: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class DeploymentSignalBundle(ContractModel):
    context_blocks: list[ProvidedBlock] = Field(default_factory=list)
    deployment_roots: list[DeploymentRoot] = Field(default_factory=list)
    candidate_components: list[CandidateComponent] = Field(default_factory=list)
    relations: list[AnalysisRelation] = Field(default_factory=list)
    deployment_stages: list[StageSignal] = Field(default_factory=list)
    validation_signals: list[ValidationSignal] = Field(default_factory=list)
    external_requirements: list[ExternalRequirement] = Field(default_factory=list)
    unresolved: list[AnalysisQuestion] = Field(default_factory=list)


class ComponentClassificationAssignment(ContractModel):
    block_type: AnalysisBlockType
    subtype: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    domain: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$",
    )


class ComponentClassificationResponse(ContractModel):
    assignments: dict[str, ComponentClassificationAssignment]
    unresolved: list[AnalysisQuestion]


class FunctionalDeployRef(ContractModel):
    executor: AnalysisExecutor
    ref: str = Field(min_length=1)
    repo_id: str = Field(pattern=IDENTIFIER_PATTERN)


class FunctionalComponent(ContractModel):
    id: str = Field(pattern=COMPONENT_IDENTIFIER_PATTERN)
    name: str = Field(min_length=1)
    implementation: str | None = None
    deployable: bool
    external: bool
    deploy: FunctionalDeployRef | None = None


class FunctionalBlock(ContractModel):
    id: str = Field(pattern=BLOCK_IDENTIFIER_PATTERN)
    name: str = Field(min_length=1)
    type: AnalysisBlockType
    subtype: str = Field(min_length=1)
    implementation: str | None = None
    state: str = "discovered"
    after: list[str] = Field(default_factory=list)
    provides: list[str] = Field(default_factory=list)
    components: list[FunctionalComponent] = Field(default_factory=list)


class AnalysisEvaluation(ContractModel):
    component_count: int = Field(ge=0)
    deployable_component_count: int = Field(ge=0)
    deployability_coverage: float = Field(ge=0.0, le=1.0)
    grounded_component_rate: float = Field(ge=0.0, le=1.0)
    forbidden_component_count: int = Field(ge=0)
    relation_count: int = Field(ge=0)
    source_derived_relation_count: int = Field(ge=0)
    artifact_line_count: int = Field(default=0, ge=0)
    llm_input_tokens: int = Field(default=0, ge=0)
    llm_requests: int = Field(default=0, ge=0)
    llm_status: str = "not_requested"
    component_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    component_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    classification_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    edge_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    edge_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    entrypoint_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    grouping_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    hallucination_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class GoldClassification(ContractModel):
    block_type: AnalysisBlockType
    subtype: str = Field(min_length=1)


class GoldEdge(ContractModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)


class GoldDefinition(ContractModel):
    expected_components: list[str] = Field(default_factory=list)
    expected_classifications: dict[str, GoldClassification] = Field(default_factory=dict)
    expected_major_edges: list[GoldEdge] = Field(default_factory=list)
    expected_entrypoints: dict[str, str] = Field(default_factory=dict)
    expected_groups: list[list[str]] = Field(default_factory=list)
    forbidden_components: list[str] = Field(default_factory=list)


class FunctionalBlocksDocument(ContractModel):
    version: str = Field(default="0.1", pattern=r"^0\.1$")
    system: str = Field(min_length=1)
    deployment: DeploymentSelection
    blocks: list[FunctionalBlock] = Field(default_factory=list)
    levels: list[list[str]] = Field(default_factory=list)
    unresolved: list[AnalysisQuestion] = Field(default_factory=list)
    evaluation: AnalysisEvaluation
