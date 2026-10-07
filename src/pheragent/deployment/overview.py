"""Build and maintain the small deployment overview that guides the next action."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .analysis_llm import CachedStructuredClassifier, strict_response_format
from .sources import SourceTools
from .task import OverviewChange, Record

StepStatus = Literal["pending", "active", "verified", "waiting_for_input", "blocked"]


class EntrypointSelection(Record):
    route: str = Field(min_length=1)
    files: list[str] = Field(min_length=1, max_length=4)
    reason: str = Field(min_length=1)


class OverviewStep(Record):
    id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    success_condition: str = Field(min_length=1)
    source_refs: list[str] = Field(min_length=1)
    status: StepStatus = "pending"


class OverviewStage(Record):
    id: str = Field(min_length=1)
    kind: Literal["prepare", "provision", "deploy", "configure", "initialize", "verify"]
    goal: str = Field(min_length=1)
    steps: list[OverviewStep] = Field(min_length=1, max_length=4)


class DeploymentOverview(Record):
    route: str = Field(min_length=1)
    route_evidence: list[str] = Field(min_length=1)
    stages: list[OverviewStage] = Field(min_length=1, max_length=6)
    discoveries: list[str] = Field(default_factory=list)
    revision: int = 1


_ENTRYPOINT_INSTRUCTIONS = """You are a senior DevOps engineer choosing where to start reading.
The harness provides a repository tree, generic deployment-artifact candidates, and local
references between files. These hints are not a semantic ranking. Reason from filenames,
directory structure, references, the target, and the objective. Select one coherent deployment
route and one to four exact files that best explain it. Include a root guide and at least one
command-bearing file such as a Makefile, script, or workflow when one is available; a manifest
alone may not show how it is invoked. Do not mix unrelated routes, examples, tests, or CI.
Return only paths present in the inventory.
"""

_OVERVIEW_INSTRUCTIONS = """You are a senior DevOps engineer forming a compact working map.
Use the complete selected files to describe the chosen deployment route in order. Produce no
more than six semantic stages and no more than four steps per stage. This is a two-level
overview, not a command transcript or full plan. Stages and steps describe deployable
outcomes, not host inspection or speculative human inputs. Do not invent commands, runtime
state, or missing details. Exact commands and inputs are resolved only for the active step.
For every step, cite one or more selected files in source_refs. These files are the starting
point for resolving that step, not proof that it is already complete.
"""


def _usage(total: dict[str, int], addition: dict[str, int]) -> None:
    for key, value in addition.items():
        total[key] = total.get(key, 0) + value


def create_overview(
    classifier: CachedStructuredClassifier,
    *,
    objective: str,
    target: dict,
    sources: SourceTools,
) -> tuple[DeploymentOverview, dict[str, int]]:
    inventory = sources.inventory()
    selected = classifier.classify(
        stage="deployment_entrypoints",
        prompt_version="deployment-entrypoints-v1",
        instructions=_ENTRYPOINT_INSTRUCTIONS,
        payload={"objective": objective, "target": target, "inventory": inventory},
        response_format=strict_response_format(EntrypointSelection, name="entrypoint_selection"),
        response_model=EntrypointSelection,
        validate=lambda value: _validate_files(value.files, sources),
    )
    if selected.value is None:
        raise RuntimeError(selected.warning or "deployment entrypoint selection failed")
    documents = [sources.read_file(reference) for reference in selected.value.files]
    if any(not document.get("complete") for document in documents):
        raise RuntimeError("a selected deployment entrypoint is too large to read in full")
    planned = classifier.classify(
        stage="deployment_overview",
        prompt_version="deployment-overview-v1",
        instructions=_OVERVIEW_INSTRUCTIONS,
        payload={
            "objective": objective,
            "target": target,
            "route": selected.value.model_dump(),
            "documents": documents,
        },
        response_format=strict_response_format(DeploymentOverview, name="deployment_overview"),
        response_model=DeploymentOverview,
        validate=lambda value: _validate_overview(value, sources),
    )
    if planned.value is None:
        raise RuntimeError(planned.warning or "deployment overview failed")
    overview = planned.value.model_copy(
        update={
            "route": selected.value.route,
            "route_evidence": selected.value.files,
            "discoveries": [],
            "stages": [
                stage.model_copy(
                    update={
                        "steps": [
                            step.model_copy(update={"status": "pending"}) for step in stage.steps
                        ]
                    }
                )
                for stage in planned.value.stages
            ],
        }
    )
    usage: dict[str, int] = {}
    _usage(usage, selected.usage)
    _usage(usage, planned.usage)
    return overview, usage


def active_step(overview: DeploymentOverview) -> OverviewStep | None:
    for stage in overview.stages:
        for step in stage.steps:
            if step.status not in {"verified", "blocked"}:
                return step
    return None


def set_step_status(overview: DeploymentOverview, step_id: str, status: StepStatus) -> None:
    matches = [step for stage in overview.stages for step in stage.steps if step.id == step_id]
    if len(matches) != 1:
        raise ValueError(f"overview step must exist exactly once: {step_id}")
    matches[0].status = status
    overview.revision += 1


def set_discoveries(overview: DeploymentOverview, discoveries: list[str]) -> None:
    discoveries = sorted(set(discoveries))
    if overview.discoveries != discoveries:
        overview.discoveries = discoveries
        overview.revision += 1


def apply_change(
    overview: DeploymentOverview, change: OverviewChange, sources: SourceTools
) -> None:
    missing = set(change.source_refs) - sources.readable_paths
    if missing:
        raise ValueError(f"overview evidence is absent from inventory: {sorted(missing)}")
    existing = [
        step for stage in overview.stages for step in stage.steps if step.id == change.step_id
    ]
    if change.operation == "update":
        if len(existing) != 1 or existing[0].status == "verified":
            raise ValueError("only one unverified overview step can be updated")
        existing[0].goal = change.goal
        existing[0].success_condition = change.success_condition
        existing[0].source_refs = change.source_refs
    else:
        if existing or not change.before_step_id:
            raise ValueError("inserted steps need a new ID and before_step_id")
        for stage in overview.stages:
            for index, step in enumerate(stage.steps):
                if step.id == change.before_step_id:
                    stage.steps.insert(
                        index,
                        OverviewStep(
                            id=change.step_id,
                            goal=change.goal,
                            success_condition=change.success_condition,
                            source_refs=change.source_refs,
                        ),
                    )
                    overview.revision += 1
                    return
        raise ValueError(f"overview step does not exist: {change.before_step_id}")
    overview.revision += 1


def _validate_files(files: list[str], sources: SourceTools) -> None:
    missing = set(files) - sources.readable_paths
    if missing:
        raise ValueError(f"entrypoint files are absent from inventory: {sorted(missing)}")
    if len(files) != len(set(files)):
        raise ValueError("entrypoint files must be distinct")


def _validate_overview(overview: DeploymentOverview, sources: SourceTools | None = None) -> None:
    steps = [step for stage in overview.stages for step in stage.steps]
    ids = [stage.id for stage in overview.stages] + [step.id for step in steps]
    if len(ids) != len(set(ids)):
        raise ValueError("overview IDs must be unique")
    if sources:
        missing = {
            reference
            for step in steps
            for reference in step.source_refs
            if reference not in sources.readable_paths
        }
        if missing:
            raise ValueError(f"overview evidence is absent from inventory: {sorted(missing)}")
