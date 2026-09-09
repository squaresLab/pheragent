from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from .graph import topological_order
from .models import IDENTIFIER_PATTERN, ContractModel

BLOCK_IDENTIFIER_PATTERN = rf"^(?:{IDENTIFIER_PATTERN[1:-1]}|B[0-9]+)$"
COMPONENT_IDENTIFIER_PATTERN = r"^C[0-9]+_[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$"
ACTION_IDENTIFIER_PATTERN = r"^A[0-9]+_[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$"


class AnalysisBlockType(StrEnum):
    BASE_INFRASTRUCTURE = "base_infrastructure"
    RUNTIME_ENVIRONMENT = "runtime_environment"
    SHARED_SERVICES = "shared_services"
    APPLICATION = "application"
    OPERATIONS = "operations"
    UNKNOWN = "unknown"


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
    CI_ONLY = "ci_only"
    TEST_OR_EXAMPLE = "test_or_example"
    IRRELEVANT = "irrelevant"
    UNKNOWN = "unknown"


class ArtifactScope(StrEnum):
    PRIMARY = "primary"
    SUPPORTING = "supporting"
    EXCLUDE = "exclude"
    UNCERTAIN = "uncertain"


class ComponentDisposition(StrEnum):
    DEPLOYMENT_COMPONENT = "deployment_component"
    PROVIDED_PREREQUISITE = "provided_prerequisite"
    EXTERNAL_DEPENDENCY = "external_dependency"
    IMPLEMENTATION_DETAIL = "implementation_detail"
    UNCERTAIN = "uncertain"


class ComponentEvidenceStrength(StrEnum):
    EXPLICIT_ORCHESTRATOR = "explicit_orchestrator"
    DECLARED_DEPLOYMENT = "declared_deployment"
    DOCUMENTED_COMMAND = "documented_command"
    INFERRED = "inferred"


class DeploymentActionType(StrEnum):
    CONFIGURE = "configure"
    INITIALIZE = "initialize"
    VALIDATE = "validate"
    OPERATE = "operate"


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


class DisagreementResolution(StrEnum):
    REPOSITORY_EVIDENCE = "repository_evidence"
    DOCUMENTATION_INTENT = "documentation_intent"
    PROFILE_SPECIFIC = "profile_specific"
    UNRESOLVED = "unresolved"


class ClaimSource(StrEnum):
    CONTEXT = "context"
    REPOSITORY = "repository"
    DOCUMENTATION = "documentation"


class WorkflowStepStatus(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"


class WorkflowStepKind(StrEnum):
    COMPONENT = "component"
    ACTION = "action"


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
    description: str | None = None
    deployment: DeploymentSelection = Field(default_factory=DeploymentSelection)
    provided_blocks: list[ProvidedBlock] = Field(default_factory=list)
    hints: ContextHints = Field(default_factory=ContextHints)
    objectives: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    domain_roles: list[str] = Field(default_factory=list)

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
                    f"provided block {block.id} references unknown block IDs: {', '.join(unknown)}"
                )
            if block.id in block.after:
                raise ValueError(f"provided block {block.id} cannot depend on itself")
        dependencies = {block.id: set(block.after) for block in self.provided_blocks}
        topological_order(
            ids,
            dependencies,
            cycle_label="provided block dependencies",
        )
        _validate_profile_and_infrastructure(self.deployment, self.provided_blocks)
        return self


def _validate_profile_and_infrastructure(
    deployment: DeploymentSelection,
    blocks: list[ProvidedBlock],
) -> None:
    profile = (deployment.profile or "").casefold().replace("_", "-")
    base_blocks = [block for block in blocks if block.type == AnalysisBlockType.BASE_INFRASTRUCTURE]
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
    command: str | None = None
    working_directory: str | None = None
    operation_source_ref: AnalysisSourceRef | None = None
    required_inputs: list[str] = Field(default_factory=list)


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
    materialized_names: list[str] = Field(default_factory=list)
    disposition: ComponentDisposition = ComponentDisposition.DEPLOYMENT_COMPONENT
    artifact_roles: list[ArtifactRole] = Field(default_factory=list)
    artifact_scope: ArtifactScope = ArtifactScope.SUPPORTING
    source_ref: AnalysisSourceRef
    deployment: ComponentDeployment | None = None
    installed_by: str | None = Field(default=None, pattern=COMPONENT_IDENTIFIER_PATTERN)
    validation_candidates: list[ValidationSignal] = Field(default_factory=list)
    classification: ComponentClassification
    classification_source: str = "deterministic_hint"
    classification_reason_code: str | None = None
    evidence_strength: ComponentEvidenceStrength = ComponentEvidenceStrength.INFERRED
    existence_locked: bool = False
    investigation_evidence_ids: list[str] = Field(default_factory=list)


class DeploymentAction(ContractModel):
    id: str = Field(pattern=ACTION_IDENTIFIER_PATTERN)
    name: str = Field(min_length=1)
    source_candidate_id: str = Field(pattern=COMPONENT_IDENTIFIER_PATTERN)
    owner_component_id: str = Field(pattern=COMPONENT_IDENTIFIER_PATTERN)
    action_type: DeploymentActionType
    source_ref: AnalysisSourceRef
    deployment: ComponentDeployment
    evidence_ids: list[str] = Field(default_factory=list)


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


class SourceDisagreement(ContractModel):
    subject: str = Field(min_length=1, max_length=120)
    left_source: ClaimSource
    right_source: ClaimSource
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    resolution: DisagreementResolution


class ReadinessCoverage(ContractModel):
    primary_roots_traced: bool
    deployment_actions_accounted: bool
    independent_gap_search_complete: bool
    docs_repo_disagreements_reviewed: bool
    docs_repo_disagreements_resolved: bool
    external_requirements_reviewed: bool
    dynamic_deployments_resolved: bool
    deployable_components_grounded: bool
    semantic_coverage_complete: bool
    validation_checks_grounded: bool
    documentation_evidence_reviewed: bool


class WorkflowTarget(ContractModel):
    """A component affected by one executable workflow operation."""

    id: str = Field(pattern=COMPONENT_IDENTIFIER_PATTERN)
    name: str = Field(min_length=1)


class DeploymentWorkflowStep(ContractModel):
    id: str = Field(pattern=r"^S[0-9]+$")
    kind: WorkflowStepKind
    targets: list[WorkflowTarget] = Field(min_length=1)
    action_id: str | None = None
    action_name: str | None = None
    executor: AnalysisExecutor
    source_ref: AnalysisSourceRef
    operation_source_ref: AnalysisSourceRef | None = None
    working_directory: str | None = None
    command: str | None = None
    required_inputs: list[str] = Field(default_factory=list)
    after: list[str] = Field(default_factory=list)
    status: WorkflowStepStatus
    blockers: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_component_fields(cls, value: Any) -> Any:
        """Read earlier one-component workflows without exposing that shape to callers."""
        if not isinstance(value, dict) or "targets" in value:
            return value
        component_id = value.get("component_id")
        component_name = value.get("component_name")
        if component_id is None or component_name is None:
            return value
        normalized = dict(value)
        normalized.pop("component_id")
        normalized.pop("component_name")
        normalized["targets"] = [{"id": component_id, "name": component_name}]
        return normalized

    @model_validator(mode="after")
    def validate_targets(self) -> DeploymentWorkflowStep:
        target_ids = [target.id for target in self.targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError(f"workflow step {self.id} contains duplicate component targets")
        return self

    @property
    def component_id(self) -> str:
        """Return the primary target for compatibility with one-component callers."""
        return self.targets[0].id

    @property
    def component_name(self) -> str:
        """Return the primary target name for compatibility with one-component callers."""
        return self.targets[0].name


class DeploymentWorkflow(ContractModel):
    version: str = "0.1"
    system: str = Field(min_length=1)
    ready_for_execution: bool
    coverage: ReadinessCoverage
    provided_blocks: list[ProvidedBlock] = Field(default_factory=list)
    steps: list[DeploymentWorkflowStep] = Field(default_factory=list)
    external_requirements: list[ExternalRequirement] = Field(default_factory=list)
    validation_checks: list[ValidationSignal] = Field(default_factory=list)
    disagreements: list[SourceDisagreement] = Field(default_factory=list)
    unresolved: list[AnalysisQuestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_step_graph(self) -> DeploymentWorkflow:
        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("deployment workflow contains duplicate step IDs")
        known = set(step_ids)
        for step in self.steps:
            unknown = sorted(set(step.after) - known)
            if unknown:
                raise ValueError(
                    f"workflow step {step.id} references unknown prerequisites: "
                    + ", ".join(unknown)
                )
            if step.id in step.after:
                raise ValueError(f"workflow step {step.id} cannot depend on itself")
            if step.kind == WorkflowStepKind.ACTION and not step.action_id:
                raise ValueError(f"action workflow step {step.id} is missing an action ID")
            if step.kind == WorkflowStepKind.COMPONENT and step.action_id:
                raise ValueError(f"component workflow step {step.id} cannot have an action ID")
        dependencies = {step.id: set(step.after) for step in self.steps}
        topological_order(
            step_ids,
            dependencies,
            cycle_label="deployment workflow step dependencies",
        )
        if self.ready_for_execution and any(
            step.status != WorkflowStepStatus.READY for step in self.steps
        ):
            raise ValueError("a ready deployment workflow cannot contain blocked steps")
        if self.ready_for_execution and not all(self.coverage.model_dump(mode="python").values()):
            raise ValueError("a ready deployment workflow cannot contain incomplete coverage")
        return self


class DeploymentSignalBundle(ContractModel):
    context_blocks: list[ProvidedBlock] = Field(default_factory=list)
    deployment_roots: list[DeploymentRoot] = Field(default_factory=list)
    candidate_components: list[CandidateComponent] = Field(default_factory=list)
    deployment_actions: list[DeploymentAction] = Field(default_factory=list)
    relations: list[AnalysisRelation] = Field(default_factory=list)
    deployment_stages: list[StageSignal] = Field(default_factory=list)
    validation_signals: list[ValidationSignal] = Field(default_factory=list)
    external_requirements: list[ExternalRequirement] = Field(default_factory=list)
    unresolved: list[AnalysisQuestion] = Field(default_factory=list)


class FunctionalDeployRef(ContractModel):
    executor: AnalysisExecutor
    ref: str = Field(min_length=1)
    repo_id: str = Field(pattern=IDENTIFIER_PATTERN)
    required_inputs: list[str] = Field(default_factory=list)


class FunctionalComponent(ContractModel):
    id: str = Field(pattern=COMPONENT_IDENTIFIER_PATTERN)
    name: str = Field(min_length=1)
    implementation: str | None = None
    deployable: bool
    external: bool
    disposition: ComponentDisposition = ComponentDisposition.DEPLOYMENT_COMPONENT
    deploy: FunctionalDeployRef | None = None
    installed_by: str | None = Field(default=None, pattern=COMPONENT_IDENTIFIER_PATTERN)


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
    candidate_component_count: int = Field(default=0, ge=0)
    component_count: int = Field(ge=0)
    implementation_detail_count: int = Field(default=0, ge=0)
    uncertain_component_count: int = Field(default=0, ge=0)
    deployable_component_count: int = Field(ge=0)
    deployability_coverage: float = Field(ge=0.0, le=1.0)
    grounded_component_rate: float = Field(ge=0.0, le=1.0)
    forbidden_component_count: int = Field(ge=0)
    relation_count: int = Field(ge=0)
    source_derived_relation_count: int = Field(ge=0)
    artifact_line_count: int = Field(default=0, ge=0)
    llm_input_tokens: int = Field(default=0, ge=0)
    llm_output_tokens: int = Field(default=0, ge=0)
    llm_requests: int = Field(default=0, ge=0)
    llm_status: str = "not_requested"
    llm_stages: dict[str, str] = Field(default_factory=dict)
    component_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    component_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    classification_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    edge_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    edge_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    entrypoint_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    grouping_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    hallucination_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    discovery_closure: float = Field(default=0.0, ge=0.0, le=1.0)
    executable_route_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    orphan_component_count: int = Field(default=0, ge=0)
    independently_grounded: bool = False


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
