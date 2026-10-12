"""Build and maintain the small deployment overview that guides the next action."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .llm import LLMClient
from .sources import SourceTools
from .task import OverviewChange, StrictModel

StepStatus = Literal["pending", "active", "verified", "waiting_for_input", "blocked"]


class EntrypointSelection(StrictModel):
    route: str = Field(min_length=1)
    files: list[str] = Field(min_length=1, max_length=4)
    reason: str = Field(min_length=1)


class OverviewStep(StrictModel):
    id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    success_condition: str = Field(min_length=1)
    components: list[str] = Field(default_factory=list, max_length=8)
    source_refs: list[str] = Field(min_length=1)
    related_sources: list[str] = Field(default_factory=list, max_length=12)
    status: StepStatus = "pending"


class OverviewStage(StrictModel):
    id: str = Field(min_length=1)
    kind: Literal["prepare", "provision", "deploy", "configure", "initialize", "verify"]
    goal: str = Field(min_length=1)
    steps: list[OverviewStep] = Field(min_length=1, max_length=4)


class DeploymentOverview(StrictModel):
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
For every step, cite primary selected files in source_refs and list named system components
only when useful. related_sources may contain relevant exact paths from the supplied neighboring
files; they are navigation hints, not evidence. Keep both lists small. Primary files are the
starting point for resolving that step, not proof that it is already complete.
"""


def _usage(total: dict[str, int], addition: dict[str, int]) -> None:
    for key, value in addition.items():
        total[key] = total.get(key, 0) + value


def create_overview(
    client: LLMClient,
    *,
    objective: str,
    target: dict,
    sources: SourceTools,
) -> tuple[DeploymentOverview, dict[str, int]]:
    inventory = sources.inventory()
    selection, selection_usage = client.complete(
        EntrypointSelection,
        instructions=_ENTRYPOINT_INSTRUCTIONS,
        payload={"objective": objective, "target": target, "inventory": inventory},
    )
    _validate_files(selection.files, sources)
    documents = [sources.read_file(reference) for reference in selection.files]
    if any(not document.get("complete") for document in documents):
        raise RuntimeError("a selected deployment entrypoint is too large to read in full")
    related_sources = sources.related_paths(selection.files)
    plan, plan_usage = client.complete(
        DeploymentOverview,
        instructions=_OVERVIEW_INSTRUCTIONS,
        payload={
            "objective": objective,
            "target": target,
            "route": selection.model_dump(),
            "documents": documents,
            "related_sources": related_sources,
        },
    )
    _validate_overview(plan, sources)
    overview = plan.model_copy(
        update={
            "route": selection.route,
            "route_evidence": selection.files,
            "discoveries": [],
            "stages": [
                stage.model_copy(
                    update={
                        "steps": [
                            step.model_copy(update={"status": "pending"}) for step in stage.steps
                        ]
                    }
                )
                for stage in plan.stages
            ],
        }
    )
    usage: dict[str, int] = {}
    _usage(usage, selection_usage)
    _usage(usage, plan_usage)
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
    missing = set(change.source_refs + change.related_sources) - sources.readable_paths
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
        existing[0].components = change.components
        existing[0].source_refs = change.source_refs
        existing[0].related_sources = change.related_sources
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
                            components=change.components,
                            source_refs=change.source_refs,
                            related_sources=change.related_sources,
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
            for reference in step.source_refs + step.related_sources
            if reference not in sources.readable_paths
        }
        if missing:
            raise ValueError(f"overview evidence is absent from inventory: {sorted(missing)}")
