"""Contracts for a deployment task and one agent decision."""

from __future__ import annotations

import re
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
    max_runtime_minutes: int = Field(default=60, gt=0)


class TaskGoal(StrictModel):
    objective: str = Field(min_length=1)
    invariants: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_invariants(self) -> TaskGoal:
        if any(
            not re.fullmatch(r"[a-z0-9][a-z0-9-]*", key) or not text.strip()
            for key, text in self.invariants.items()
        ):
            raise ValueError("invariants need nonempty descriptions and lowercase hyphenated ids")
        return self


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


class InputRequest(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    prompt: str = Field(min_length=1)
    sensitive: bool = False


class DeploymentTask(StrictModel):
    task: TaskGoal
    sources: TaskSources
    environment: Target
    constraints: Constraints = Field(default_factory=Constraints)
    budgets: Budgets = Field(default_factory=Budgets)
    inputs: dict[str, TaskInput] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_sources(self) -> DeploymentTask:
        if not self.sources.repositories and not self.sources.documentation:
            raise ValueError("at least one repository or documentation source is required")
        return self


class RunNote(StrictModel):
    key: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    text: str = Field(min_length=1, max_length=1_000)
    evidence_events: list[int] = Field(default_factory=list)


class Decision(StrictModel):
    action: Literal["shell", "ask", "record", "finish"]
    reason: str = Field(min_length=1)
    notes: list[RunNote] = Field(default_factory=list)
    command: list[str] = Field(default_factory=list)
    working_directory: str | None = None
    component: str | None = None
    deployed: bool | None = None
    evidence_events: list[int] = Field(default_factory=list)
    verification_events: list[int] = Field(default_factory=list)
    human_action: str | None = None
    required_inputs: list[InputRequest] = Field(default_factory=list)
    stdin_input: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")

    @model_validator(mode="after")
    def validate_action(self) -> Decision:
        if self.action == "shell" and not self.command:
            raise ValueError("shell action requires command")
        if self.action == "ask" and not self.required_inputs:
            raise ValueError("ask action requires required_inputs")
        if self.action == "record" and (not self.component or self.deployed is None):
            raise ValueError("record action requires component and deployed")
        if self.deployed is True and not self.evidence_events:
            raise ValueError("deployed component requires evidence_events")
        if self.verification_events and not set(self.verification_events) <= set(
            self.evidence_events
        ):
            raise ValueError("verification_events must be included in evidence_events")
        if self.deployed is False and not self.human_action:
            raise ValueError("not deployed component requires human_action")
        names = [item.name for item in self.required_inputs]
        if len(names) != len(set(names)):
            raise ValueError("required input names must be distinct")
        return self
