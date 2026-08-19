class DeploymentError(Exception):
    """Base error for expected deployment-analysis and execution failures."""


class DeploymentInputError(DeploymentError):
    """The user-provided context, source mapping, or approval is invalid."""


class WorkflowNotExecutableError(DeploymentError):
    """A workflow cannot safely execute in its current state."""


class WorkflowExecutionError(DeploymentError):
    """A grounded deployment operation failed after execution began."""
