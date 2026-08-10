from enum import StrEnum


class SourceKind(StrEnum):
    GIT = "git"
    LOCAL_DIRECTORY = "local_directory"
    LOCAL_FILE = "local_file"


class BlockType(StrEnum):
    HOST_READINESS = "host_readiness"
    PLATFORM = "platform"
    SHARED_SERVICES = "shared_services"
    APPLICATION = "application"
    INITIALIZATION = "initialization"
    OPERATIONS = "operations"
    VERIFICATION = "verification"
    HUMAN_GATE = "human_gate"


class TargetKind(StrEnum):
    HOST = "host"
    HOST_GROUP = "host_group"
    KUBERNETES_CLUSTER = "kubernetes_cluster"
    CONTAINER_PLATFORM = "container_platform"
    EXTERNAL_SERVICE = "external_service"
    LOGICAL_ENVIRONMENT = "logical_environment"
    UNKNOWN = "unknown"


class ProvenanceOrigin(StrEnum):
    EXTRACTED = "extracted"
    INFERRED = "inferred"
    SYNTHESIZED = "synthesized"
    HUMAN_APPROVED = "human_approved"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ComponentRelationType(StrEnum):
    REQUIRES = "requires"
    HOSTS = "hosts"
    CONNECTS_TO = "connects_to"
    CONFIGURED_BY = "configured_by"
    INITIALIZED_BY = "initialized_by"
    ORDERED_BEFORE = "ordered_before"
    PROVIDES_TO = "provides_to"


class OperationPhase(StrEnum):
    INSPECT = "inspect"
    PREPARE = "prepare"
    INSTALL = "install"
    CONFIGURE = "configure"
    START = "start"
    INITIALIZE = "initialize"
    VALIDATE = "validate"
    UPGRADE = "upgrade"
    REMOVE = "remove"
    MANUAL = "manual"


class Executor(StrEnum):
    SHELL = "shell"
    ANSIBLE = "ansible"
    TERRAFORM = "terraform"
    HELM = "helm"
    HELMSMAN = "helmsman"
    KUSTOMIZE = "kustomize"
    KUBECTL = "kubectl"
    DOCKER_COMPOSE = "docker_compose"
    GITHUB_ACTIONS = "github_actions"
    HTTP_API = "http_api"
    MANUAL = "manual"
    UNKNOWN = "unknown"


class ValidationLevel(StrEnum):
    STRUCTURAL = "structural"
    READINESS = "readiness"
    FUNCTIONAL = "functional"
    INTEGRATION = "integration"
    END_TO_END = "end_to_end"


class ReviewStatus(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class FactPredicate(StrEnum):
    CONTAINS = "contains"
    REQUIRES = "requires"
    PROVIDES = "provides"
    DEPLOYS_TO = "deploys_to"
    IMPLEMENTED_BY = "implemented_by"
    ORDERED_BEFORE = "ordered_before"
    VALIDATED_BY = "validated_by"
    CONFIGURED_BY = "configured_by"
    INITIALIZED_BY = "initialized_by"
    HUMAN_INPUT_REQUIRED = "human_input_required"


class InventoryCategory(StrEnum):
    DOCUMENTATION = "documentation"
    SHELL = "shell"
    ANSIBLE = "ansible"
    TERRAFORM = "terraform"
    HELM = "helm"
    HELMSMAN = "helmsman"
    KUSTOMIZE = "kustomize"
    KUBERNETES = "kubernetes"
    COMPOSE = "compose"
    CI_WORKFLOW = "ci_workflow"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class DeterministicFindingKind(StrEnum):
    DOCUMENT_HEADING = "document_heading"
    PROCEDURE_STEP = "procedure_step"
    COMMAND = "command"
    FILE_REFERENCE = "file_reference"
    COMPONENT = "component"
    DEPENDENCY = "dependency"
    RESOURCE = "resource"
    VALIDATION = "validation"
    OUTPUT = "output"
    CONFIGURATION = "configuration"


class FindingSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
