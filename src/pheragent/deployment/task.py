"""Task and decision contracts for progressive deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceLocation(StrictModel):
    location: str
    revision: str | None = None


class TaskSources(StrictModel):
    repositories: list[str | SourceLocation] = Field(default_factory=list)
    documentation: list[str | SourceLocation] = Field(default_factory=list)


class Target(StrictModel):
    type: Literal["shell", "kubernetes"]
    kubeconfig: Path | None = None
    context: str | None = None
    sandbox: bool = False

    @model_validator(mode="after")
    def require_kubernetes_context(self) -> Target:
        if self.type == "kubernetes" and not self.context:
            raise ValueError("Kubernetes tasks require environment.context")
        return self


class Constraints(StrictModel):
    allow_new_infrastructure: bool = False
    allow_destructive_actions: bool = False
    allowed_namespaces: list[str] = Field(default_factory=list)


class Budgets(StrictModel):
    max_cycles: int = Field(default=30, gt=0)
    max_mutating_actions: int = Field(default=20, gt=0)
    max_runtime_minutes: int = Field(default=60, gt=0)
    max_read_actions_per_cycle: int = Field(default=20, gt=0)


class ContextSettings(StrictModel):
    mode: Literal["recent", "summary"] = "recent"
    history_window: int = Field(default=8, gt=0, le=50)


class Check(StrictModel):
    command: list[str] = Field(min_length=1)
    contains: str | None = None
    unsatisfied_exit_codes: list[int] = Field(default_factory=list)


class TaskGoal(StrictModel):
    objective: str = Field(min_length=1)
    stop_after_verified_outcomes: int | None = Field(default=None, gt=0)


class TaskInput(StrictModel):
    value: str | None = None
    from_env: str | None = None
    from_file: Path | None = None
    sensitive: bool = False

    @model_validator(mode="after")
    def require_one_source(self) -> TaskInput:
        sources = [self.value is not None, self.from_env is not None, self.from_file is not None]
        if sum(sources) != 1:
            raise ValueError("input needs exactly one of value, from_env, or from_file")
        if self.sensitive and self.value is not None:
            raise ValueError("sensitive inputs cannot be stored inline")
        return self


class DeploymentTask(StrictModel):
    task: TaskGoal
    sources: TaskSources
    environment: Target
    constraints: Constraints = Field(default_factory=Constraints)
    budgets: Budgets = Field(default_factory=Budgets)
    context: ContextSettings = Field(default_factory=ContextSettings)
    success_checks: list[Check] = Field(default_factory=list)
    inputs: dict[str, TaskInput] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_sources(self) -> DeploymentTask:
        if not self.sources.repositories and not self.sources.documentation:
            raise ValueError("at least one repository or documentation source is required")
        if self.task.stop_after_verified_outcomes and self.success_checks:
            raise ValueError("choose fixed success checks or a verified-outcome target")
        return self


class OverviewChange(StrictModel):
    operation: Literal["insert_before", "update"]
    step_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    success_condition: str = Field(min_length=1)
    components: list[str] = Field(default_factory=list, max_length=8)
    source_refs: list[str] = Field(min_length=1)
    related_sources: list[str] = Field(default_factory=list, max_length=12)
    before_step_id: str | None = None


class Decision(StrictModel):
    kind: Literal["ACT", "DONE", "BLOCKED", "ASK_HUMAN", "WAITING_FOR_INPUT"]
    tool: (
        Literal[
            "inventory_sources",
            "search_sources",
            "read_file",
            "list_directory",
            "add_source",
            "observe",
            "execute",
        ]
        | None
    )
    reason: str
    focus: str
    query: str | None
    source_path: str | None
    start_line: int | None
    end_line: int | None
    command: list[str]
    working_directory: str | None
    evidence: list[str]
    source: SourceLocation | None = None
    selected_route: str | None = None
    step_id: str | None = None
    required_inputs: list[str] = Field(default_factory=list)
    sensitive_inputs: list[str] = Field(default_factory=list)
    overview_change: OverviewChange | None = None
    completes_step: bool = False
    options: list[str] = Field(default_factory=list)
    outcome_id: str | None = None
    expected_change: str | None
    validation: list[Check]
    add_gaps: list[str]
    resolve_gaps: list[str]
    add_questions: list[str]
    resolve_questions: list[str]

    @model_validator(mode="after")
    def require_sensitive_inputs(self) -> Decision:
        if not set(self.sensitive_inputs) <= set(self.required_inputs):
            raise ValueError("sensitive inputs must also be required inputs")
        return self
