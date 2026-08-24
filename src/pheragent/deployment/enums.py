from enum import StrEnum


class SourceKind(StrEnum):
    """The acquisition mechanism for one analysis source."""

    GIT = "git"
    LOCAL_DIRECTORY = "local_directory"
    LOCAL_FILE = "local_file"


class AnalysisTreatment(StrEnum):
    """The evidence-selection strategy used by repository analysis."""

    DETERMINISTIC = "a0"
    HYBRID = "a1"
    HYBRID_GRAPH = "a2"


class InventoryCategory(StrEnum):
    """A source file's deployment-relevant format."""

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
