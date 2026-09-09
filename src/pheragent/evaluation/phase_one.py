from __future__ import annotations

import json
import posixpath
import re
import shlex
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from pheragent.deployment.analysis_llm import (
    CachedStructuredClassifier,
    ClassificationOutcome,
    aggregate_usage,
)
from pheragent.deployment.analysis_models import (
    AnalysisExecutor,
    AnalysisSourceRef,
    ComponentDisposition,
    DeploymentContext,
    DeploymentWorkflow,
    DeploymentWorkflowStep,
    FunctionalBlock,
    FunctionalBlocksDocument,
    FunctionalComponent,
    WorkflowStepKind,
    WorkflowStepStatus,
)
from pheragent.deployment.redaction import redact_secrets

from ._evidence import (
    DeploymentEntity,
    SourceCatalog,
    deployment_mode,
    identity_matches,
    normalize_identity,
)
from ._judge import (
    CompletenessJudgeDecision,
    CompletenessVerdict,
    ComponentJudgeDecision,
    ComponentJudgeResponse,
    JudgeVerdict,
    PhaseOneJudgeConfig,
    judge_completeness,
    judge_components,
)

_ARTIFACT_NAMES = ("functional-blocks.yaml", "deployment-workflow.yaml")
_DOCUMENTATION_SUFFIXES = {".md", ".rst", ".txt"}
_OPERATION_LABEL = re.compile(
    r"^(?:apply|configure|copy|create|delete|deploy|enable|import|install|mount|restart|"
    r"run|setup|update|upgrade|use|validate|verify)\b|^(?:getting-started|introduction|"
    r"overview|prerequisites|usage)$",
    re.IGNORECASE,
)
_PROFILE_TERMS = {"aws", "azure", "compose", "docker", "gcp", "kubernetes", "sandbox"}
_COMPONENT_JUDGE_BATCH_SIZE = 20


class EvaluationModel(BaseModel):
    """Base contract for serializable evaluation inputs and results."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EvaluationDimension(StrEnum):
    VALIDITY = "validity"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    DEPLOYABILITY = "deployability"
    CONSISTENCY = "consistency"


class PhaseOneEvaluationInput(EvaluationModel):
    """Stable artifacts and pinned evidence required to evaluate Phase 1."""

    run_directories: tuple[Path, ...] = Field(min_length=1)
    source_roots: dict[str, Path] = Field(min_length=1)
    deployment_context: Path

    @model_validator(mode="after")
    def reject_duplicate_runs(self) -> PhaseOneEvaluationInput:
        resolved = [path.expanduser().resolve() for path in self.run_directories]
        if len(resolved) != len(set(resolved)):
            raise ValueError("evaluation input contains duplicate run directories")
        return self


class MetricFinding(EvaluationModel):
    run_id: str
    subject: str
    score: float = Field(ge=0.0, le=1.0)
    resolved: bool = True
    included: bool = True
    reason: str
    evidence_refs: tuple[str, ...] = ()


class MetricResult(EvaluationModel):
    dimension: EvaluationDimension
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    assessed_items: int = Field(default=0, ge=0)
    unresolved_items: int = Field(default=0, ge=0)
    explanation: str | None = None
    issues: tuple[MetricFinding, ...] = ()
    findings: tuple[MetricFinding, ...] = ()


class LLMJudgeSummary(EvaluationModel):
    enabled: bool
    model: str | None = None
    complete: bool
    stages: dict[str, str] = Field(default_factory=dict)
    usage: dict[str, int] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class PhaseOneRunReport(EvaluationModel):
    run_id: str
    run_directory: Path
    validity: MetricResult
    relevance: MetricResult
    completeness: MetricResult
    deployability: MetricResult
    judge: LLMJudgeSummary


class PhaseOneEvaluationReport(EvaluationModel):
    evaluation_version: str = "0.4"
    runs: tuple[PhaseOneRunReport, ...]
    consistency: MetricResult


@dataclass(frozen=True, slots=True)
class _RunArtifacts:
    directory: Path
    document: FunctionalBlocksDocument
    workflow: DeploymentWorkflow

    @property
    def run_id(self) -> str:
        return self.directory.name


class _PhaseOneEvaluator:
    def __init__(
        self,
        evaluation_input: PhaseOneEvaluationInput,
        judge_config: PhaseOneJudgeConfig | None = None,
    ) -> None:
        self.input = evaluation_input
        self.judge_config = judge_config
        self.context = _load_model(evaluation_input.deployment_context, DeploymentContext)
        self.catalog = SourceCatalog(evaluation_input.source_roots)
        self.runs = tuple(
            _load_run(path, self.context.system) for path in evaluation_input.run_directories
        )
        self.entities = {
            run.run_id: self.catalog.deployment_entities(
                self.context,
                _selected_source_paths(run, self.context),
            )
            for run in self.runs
        }

    def evaluate(self) -> PhaseOneEvaluationReport:
        run_reports = []
        for run in self.runs:
            validity = self.validity(run)
            relevance = self.relevance(run)
            completeness = self.completeness(run)
            deployability = self.deployability(run)
            judge = _disabled_judge_summary()
            if self.judge_config is not None:
                validity, relevance, completeness, deployability, judge = self._judge(
                    run,
                    validity,
                    relevance,
                    completeness,
                    deployability,
                )
            run_reports.append(
                PhaseOneRunReport(
                    run_id=run.run_id,
                    run_directory=run.directory,
                    validity=validity,
                    relevance=relevance,
                    completeness=completeness,
                    deployability=deployability,
                    judge=judge,
                )
            )
        return PhaseOneEvaluationReport(
            runs=tuple(run_reports),
            consistency=self.consistency(),
        )

    def _judge(
        self,
        run: _RunArtifacts,
        validity: MetricResult,
        relevance: MetricResult,
        completeness: MetricResult,
        deployability: MetricResult,
    ) -> tuple[MetricResult, MetricResult, MetricResult, MetricResult, LLMJudgeSummary]:
        config = self.judge_config
        if config is None:
            raise RuntimeError("judge configuration is required")
        classifier = config.classifier()
        component_outcomes, component_evidence = self._judge_component_batches(
            run,
            config,
            classifier,
        )
        component_ids = {
            component.id for _, component in _discovered_components(run.document.blocks)
        }

        completeness_outcome = None
        entity_ids: dict[str, DeploymentEntity] = {}
        workflow_step_ids = {
            step.id for step in run.workflow.steps if step.kind == WorkflowStepKind.ACTION
        }
        if self.entities[run.run_id]:
            completeness_payload, completeness_evidence, entity_ids = _completeness_judge_payload(
                run,
                self.context,
                self.catalog,
                self.entities[run.run_id],
                max_evidence_characters=config.max_evidence_characters,
            )
            completeness_outcome = judge_completeness(
                completeness_payload,
                entity_ids=set(entity_ids),
                component_ids=component_ids,
                workflow_step_ids=workflow_step_ids,
                evidence_ids=set(completeness_evidence),
                classifier=classifier,
            )

        component_judge_complete = all(outcome.value is not None for outcome in component_outcomes)
        if component_judge_complete:
            decisions = {
                decision.component_id: decision
                for outcome in component_outcomes
                if outcome.value is not None
                for decision in outcome.value.decisions
            }
            validity = _apply_component_decisions(
                validity,
                decisions,
                "validity",
                component_evidence,
            )
            relevance = _apply_component_decisions(
                relevance,
                decisions,
                "relevance",
                component_evidence,
            )
            deployability = _apply_component_decisions(
                deployability,
                decisions,
                "deployability",
                component_evidence,
            )
        if completeness_outcome is not None and completeness_outcome.value is not None:
            completeness = _apply_completeness_decisions(
                completeness,
                completeness_outcome.value.decisions,
                entity_ids,
                completeness_evidence,
            )

        outcomes = list(component_outcomes)
        if completeness_outcome is not None:
            outcomes.append(completeness_outcome)
        stages = {outcome.stage: outcome.status for outcome in outcomes}
        complete = component_judge_complete and (
            completeness_outcome is None or completeness_outcome.value is not None
        )
        warnings = tuple(outcome.warning for outcome in outcomes if outcome.warning)
        usage = aggregate_usage(*outcomes)
        usage["input_tokens_estimate"] = sum(outcome.input_tokens_estimate for outcome in outcomes)
        return (
            validity,
            relevance,
            completeness,
            deployability,
            LLMJudgeSummary(
                enabled=True,
                model=config.model,
                complete=complete,
                stages=stages,
                usage=usage,
                warnings=warnings,
            ),
        )

    def _judge_component_batches(
        self,
        run: _RunArtifacts,
        config: PhaseOneJudgeConfig,
        classifier: CachedStructuredClassifier,
    ) -> tuple[
        tuple[ClassificationOutcome[ComponentJudgeResponse], ...],
        dict[str, dict[str, str]],
    ]:
        component_batches = _component_batches(
            _discovered_components(run.document.blocks),
            _COMPONENT_JUDGE_BATCH_SIZE,
        )
        if not component_batches:
            return (), {}
        evidence_budget = max(
            1,
            config.max_evidence_characters // len(component_batches),
        )
        outcomes = []
        combined_evidence = {}
        for batch_number, components in enumerate(component_batches, start=1):
            payload, evidence = _component_judge_payload(
                run,
                self.context,
                self.catalog,
                components=components,
                batch_number=batch_number,
                total_batches=len(component_batches),
                max_evidence_characters=evidence_budget,
            )
            outcomes.append(
                judge_components(
                    payload,
                    component_ids={component.id for _, component in components},
                    evidence_ids=set(evidence),
                    batch_number=batch_number,
                    classifier=classifier,
                )
            )
            combined_evidence.update(evidence)
        return tuple(outcomes), combined_evidence

    def validity(self, run: _RunArtifacts) -> MetricResult:
        components = _discovered_components(run.document.blocks)
        duplicate_ids = _duplicates(component.id for _, component in components)
        duplicate_names = _duplicates(
            normalize_identity(component.name) for _, component in components
        )
        by_id = {component.id: component for _, component in components}
        findings = []
        for _block, component in components:
            if (
                component.id in duplicate_ids
                or normalize_identity(component.name) in duplicate_names
            ):
                findings.append(
                    _finding(run, component.id, 0.0, "component identity is not unique")
                )
                continue
            if _looks_like_operation(component.name):
                findings.append(
                    _finding(
                        run, component.id, 0.0, "component name describes an action or heading"
                    )
                )
                continue
            entity = _matching_entity(component, self.entities[run.run_id])
            if entity is not None:
                findings.append(
                    _finding(
                        run,
                        component.id,
                        1.0,
                        f"component is declared by a {entity.kind} source artifact",
                        entity.evidence_refs,
                    )
                )
                continue
            if component.installed_by:
                owner = by_id.get(component.installed_by)
                if owner is not None and owner.deploy is not None:
                    findings.append(
                        _finding(
                            run,
                            component.id,
                            0.5,
                            "component is linked to a source-backed shared installer but lacks an "
                            "independent declaration",
                            (_deploy_ref(owner),),
                            resolved=False,
                        )
                    )
                    continue
            if component.deploy is None:
                findings.append(
                    _finding(
                        run,
                        component.id,
                        0.0,
                        "component has no independently checkable source reference",
                        resolved=False,
                    )
                )
                continue
            evidence_refs = (_deploy_ref(component),)
            path = self.catalog.resolve(component.deploy.repo_id, component.deploy.ref)
            if path is None:
                findings.append(
                    _finding(
                        run,
                        component.id,
                        0.0,
                        "component source reference does not exist",
                        evidence_refs,
                    )
                )
                continue
            if not self.catalog.supports_identity(
                component.deploy.repo_id,
                component.deploy.ref,
                component.name,
                component.implementation,
            ):
                findings.append(
                    _finding(
                        run,
                        component.id,
                        0.0,
                        "referenced source does not identify the claimed component",
                        evidence_refs,
                        resolved=False,
                    )
                )
                continue
            if path.suffix.casefold() in _DOCUMENTATION_SUFFIXES:
                findings.append(
                    _finding(
                        run,
                        component.id,
                        0.5,
                        "documentation supports the name, but no independent "
                        "deployable declaration was observed",
                        evidence_refs,
                        resolved=False,
                    )
                )
                continue
            findings.append(
                _finding(
                    run,
                    component.id,
                    1.0,
                    "component identity is grounded in source",
                    evidence_refs,
                )
            )
        return _result(EvaluationDimension.VALIDITY, findings)

    def relevance(self, run: _RunArtifacts) -> MetricResult:
        mode = deployment_mode(self.context)
        findings = []
        for block, component in _discovered_components(run.document.blocks):
            ref = _deploy_ref(component) if component.deploy else None
            evidence_refs = (ref,) if ref else ()
            if _looks_like_operation(component.name):
                findings.append(
                    _finding(
                        run, component.id, 0.0, "action or documentation heading is not a component"
                    )
                )
                continue
            if _matches_exclusion(component, self.context):
                findings.append(
                    _finding(
                        run,
                        component.id,
                        0.0,
                        "component belongs to an excluded profile",
                        evidence_refs,
                    )
                )
                continue
            if _matches_provided_block(component, self.context):
                correctly_provided = (
                    block.state == "provided"
                    or component.disposition == ComponentDisposition.PROVIDED_PREREQUISITE
                )
                findings.append(
                    _finding(
                        run,
                        component.id,
                        1.0 if correctly_provided else 0.0,
                        "provided prerequisite is labelled as provided"
                        if correctly_provided
                        else "provided infrastructure is reported as a discovered component",
                        evidence_refs,
                    )
                )
                continue
            executor = _component_executor(component, run)
            if executor is not None and not _executor_matches_mode(executor, mode):
                findings.append(
                    _finding(
                        run,
                        component.id,
                        0.0,
                        f"{executor.value} route conflicts with the {mode} deployment profile",
                        evidence_refs,
                    )
                )
                continue
            findings.append(
                _finding(
                    run,
                    component.id,
                    1.0,
                    "no deployment-context conflict was found",
                    evidence_refs,
                )
            )
        return _result(EvaluationDimension.RELEVANCE, findings)

    def completeness(self, run: _RunArtifacts) -> MetricResult:
        entities = self.entities[run.run_id]
        if not entities:
            return _unavailable(
                EvaluationDimension.COMPLETENESS,
                "no high-confidence deployment entities were observable in the pinned sources",
            )
        claims = _accounted_claims(run, self.context)
        findings = []
        for entity in entities:
            match = next(
                (claim for claim in claims if identity_matches(entity.name, claim[0])), None
            )
            if match is None:
                findings.append(
                    _finding(
                        run,
                        entity.name,
                        0.0,
                        f"{entity.kind} entity is not accounted for by the Phase 1 artifacts",
                        entity.evidence_refs,
                    )
                )
                continue
            findings.append(
                _finding(
                    run,
                    entity.name,
                    1.0,
                    f"{entity.kind} entity is accounted for as {match[1]}",
                    entity.evidence_refs,
                )
            )
        return _result(EvaluationDimension.COMPLETENESS, findings)

    def deployability(self, run: _RunArtifacts) -> MetricResult:
        components = _discovered_components(run.document.blocks)
        by_id = {component.id: component for _, component in components}
        steps_by_target = _steps_by_target(run.workflow.steps)
        findings = []
        for _block, component in components:
            if not component.deployable:
                continue
            owner_id = _route_owner(component, by_id)
            candidate_steps = steps_by_target.get(owner_id, ())
            if not candidate_steps:
                findings.append(
                    _finding(run, component.id, 0.0, "deployable component has no workflow route")
                )
                continue
            problems = [self._route_problem(step, by_id.get(owner_id)) for step in candidate_steps]
            problem = next((candidate for candidate in problems if candidate is None), problems[0])
            step = candidate_steps[problems.index(problem)]
            refs = tuple(
                dict.fromkeys(
                    filter(
                        None,
                        [
                            _source_ref(step.source_ref.repo_id, step.source_ref.path),
                            (
                                _source_ref(
                                    step.operation_source_ref.repo_id,
                                    step.operation_source_ref.path,
                                )
                                if step.operation_source_ref
                                else None
                            ),
                        ],
                    )
                )
            )
            findings.append(
                _finding(
                    run,
                    component.id,
                    1.0 if problem is None else 0.0,
                    "component has a source-backed executable route"
                    if problem is None
                    else problem,
                    refs,
                )
            )
        return _result(EvaluationDimension.DEPLOYABILITY, findings)

    def _route_problem(
        self,
        step: DeploymentWorkflowStep,
        component: FunctionalComponent | None,
    ) -> str | None:
        if component is None:
            return "workflow route targets an unknown component"
        if step.status != WorkflowStepStatus.READY or step.blockers:
            return "workflow route is blocked"
        if not step.command or not step.command.strip():
            return "workflow route has no executable command"
        if step.executor == AnalysisExecutor.UNKNOWN:
            return "workflow route has an unknown executor"
        if component.deploy is not None and step.executor != component.deploy.executor:
            return "workflow and component executors disagree"
        if self.catalog.resolve(step.source_ref.repo_id, step.source_ref.path) is None:
            return "workflow source reference does not exist"
        if (
            step.operation_source_ref is not None
            and self.catalog.resolve(
                step.operation_source_ref.repo_id,
                step.operation_source_ref.path,
            )
            is None
        ):
            return "workflow operation source reference does not exist"
        if self.catalog.resolve_directory(step.source_ref.repo_id, step.working_directory) is None:
            return "workflow working directory does not exist in its source"
        if any(not value.strip() for value in step.required_inputs):
            return "workflow route contains an unnamed required input"
        if len(step.required_inputs) != len(set(step.required_inputs)):
            return "workflow route contains duplicate required inputs"
        missing_command_ref = _missing_command_reference(step, self.catalog)
        if missing_command_ref:
            return f"workflow command references a missing source file: {missing_command_ref}"
        return None

    def consistency(self) -> MetricResult:
        if len(self.runs) < 2:
            return _unavailable(
                EvaluationDimension.CONSISTENCY,
                "consistency requires at least two comparable runs",
            )
        signatures = [_run_signature(run.directory) for run in self.runs]
        if any(signature is None for signature in signatures):
            return _unavailable(
                EvaluationDimension.CONSISTENCY,
                "all compared runs require completed run manifests with pins and analysis settings",
            )
        if len(set(signatures)) != 1:
            return _unavailable(
                EvaluationDimension.CONSISTENCY,
                "run pins, deployment context, model, or analysis budgets differ",
            )
        findings = []
        for left, right in combinations(self.runs, 2):
            scores = _projection_similarities(left, right)
            findings.append(
                MetricFinding(
                    run_id=f"{left.run_id}..{right.run_id}",
                    subject="normalized Phase 1 projection",
                    score=fmean(scores.values()),
                    reason=", ".join(f"{name}={value:.3f}" for name, value in scores.items()),
                    evidence_refs=tuple(
                        str(run.directory / artifact)
                        for run in (left, right)
                        for artifact in _ARTIFACT_NAMES
                    ),
                )
            )
        return _result(EvaluationDimension.CONSISTENCY, findings)


def evaluate_phase_one(
    evaluation_input: PhaseOneEvaluationInput,
    *,
    judge_config: PhaseOneJudgeConfig | None = None,
) -> PhaseOneEvaluationReport:
    """Evaluate all Phase 1 runs and optionally resolve semantic findings with an LLM."""
    return _PhaseOneEvaluator(evaluation_input, judge_config=judge_config).evaluate()


def evaluate_validity(evaluation_input: PhaseOneEvaluationInput) -> MetricResult:
    """Measure whether discovered components are supported by pinned source evidence."""
    evaluator = _PhaseOneEvaluator(evaluation_input)
    return _combine(
        EvaluationDimension.VALIDITY, [evaluator.validity(run) for run in evaluator.runs]
    )


def evaluate_relevance(evaluation_input: PhaseOneEvaluationInput) -> MetricResult:
    """Measure whether discovered components belong to the selected deployment profile."""
    evaluator = _PhaseOneEvaluator(evaluation_input)
    return _combine(
        EvaluationDimension.RELEVANCE, [evaluator.relevance(run) for run in evaluator.runs]
    )


def evaluate_completeness(evaluation_input: PhaseOneEvaluationInput) -> MetricResult:
    """Measure whether observed deployment entities are accounted for by the analysis."""
    evaluator = _PhaseOneEvaluator(evaluation_input)
    return _combine(
        EvaluationDimension.COMPLETENESS,
        [evaluator.completeness(run) for run in evaluator.runs],
    )


def evaluate_deployability(evaluation_input: PhaseOneEvaluationInput) -> MetricResult:
    """Measure coverage by source-supported, statically valid deployment routes."""
    evaluator = _PhaseOneEvaluator(evaluation_input)
    return _combine(
        EvaluationDimension.DEPLOYABILITY,
        [evaluator.deployability(run) for run in evaluator.runs],
    )


def evaluate_consistency(evaluation_input: PhaseOneEvaluationInput) -> MetricResult:
    """Measure stability across comparable repeated Phase 1 runs."""
    return _PhaseOneEvaluator(evaluation_input).consistency()


@dataclass(slots=True)
class _EvidenceLedger:
    catalog: SourceCatalog
    remaining_characters: int
    entries: dict[str, dict[str, str]]
    prefix: str = "E"

    @classmethod
    def create(
        cls,
        catalog: SourceCatalog,
        max_characters: int,
        *,
        prefix: str = "E",
    ) -> _EvidenceLedger:
        return cls(
            catalog=catalog,
            remaining_characters=max_characters,
            entries={},
            prefix=prefix,
        )

    def add(
        self,
        source_id: str,
        path: str,
        *,
        names: tuple[str, ...],
        start_line: int | None = None,
        end_line: int | None = None,
        max_characters: int = 900,
    ) -> str | None:
        if self.remaining_characters <= 0:
            return None
        limit = min(max_characters, self.remaining_characters)
        excerpt = self.catalog.excerpt(
            source_id,
            path,
            names=names,
            start_line=start_line,
            end_line=end_line,
            max_characters=limit,
        )
        if not excerpt:
            return None
        evidence_id = f"{self.prefix}{len(self.entries) + 1:03d}"
        self.entries[evidence_id] = {
            "id": evidence_id,
            "source_ref": _source_ref(source_id, path),
            "excerpt": excerpt,
        }
        self.remaining_characters -= len(excerpt)
        return evidence_id


def _component_judge_payload(
    run: _RunArtifacts,
    context: DeploymentContext,
    catalog: SourceCatalog,
    *,
    components: tuple[tuple[FunctionalBlock, FunctionalComponent], ...],
    batch_number: int,
    total_batches: int,
    max_evidence_characters: int,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    ledger = _EvidenceLedger.create(
        catalog,
        max_evidence_characters,
        prefix=f"B{batch_number:02d}E",
    )
    all_components = _discovered_components(run.document.blocks)
    by_id = {component.id: component for _, component in all_components}
    steps_by_target = _steps_by_target(run.workflow.steps)
    excerpt_limit = max(
        1,
        min(700, max_evidence_characters // max(1, len(components) * 2)),
    )
    cards = []
    for block, component in components:
        owner_id = _route_owner(component, by_id)
        step = next(iter(steps_by_target.get(owner_id, ())), None)
        evidence_ids = []
        for source_ref in _component_source_refs(component, by_id, step):
            evidence_id = ledger.add(
                source_ref.repo_id,
                source_ref.path,
                names=tuple(filter(None, (component.name, component.implementation))),
                start_line=source_ref.start_line,
                end_line=source_ref.end_line,
                max_characters=excerpt_limit,
            )
            if evidence_id:
                evidence_ids.append(evidence_id)
            if len(evidence_ids) == 2:
                break
        cards.append(
            {
                "id": component.id,
                "name": component.name,
                "implementation": component.implementation,
                "block": f"{block.type.value}/{block.subtype}",
                "disposition": component.disposition.value,
                "deployable": component.deployable,
                "external": component.external,
                "installed_by": component.installed_by,
                "route": (
                    {
                        "executor": step.executor.value,
                        "command": redact_secrets(step.command or ""),
                        "working_directory": step.working_directory,
                        "status": step.status.value,
                    }
                    if step
                    else None
                ),
                "evidence_ids": evidence_ids,
            }
        )
    return (
        {
            "task": "judge_phase1_components",
            "run_id": run.run_id,
            "batch": {
                "number": batch_number,
                "total": total_batches,
                "expected_decisions": len(cards),
                "expected_component_ids": [card["id"] for card in cards],
            },
            "context": _context_card(context),
            "components": cards,
            "evidence": list(ledger.entries.values()),
        },
        ledger.entries,
    )


def _completeness_judge_payload(
    run: _RunArtifacts,
    context: DeploymentContext,
    catalog: SourceCatalog,
    entities: tuple[DeploymentEntity, ...],
    *,
    max_evidence_characters: int,
) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, DeploymentEntity]]:
    ledger = _EvidenceLedger.create(catalog, max_evidence_characters)
    entity_ids = {f"D{index:03d}": entity for index, entity in enumerate(entities, start=1)}
    excerpt_limit = max(
        1,
        min(700, max_evidence_characters // max(1, len(entities) * 2)),
    )
    cards = []
    for entity_id, entity in entity_ids.items():
        evidence_ids = []
        for raw_ref in entity.evidence_refs[:2]:
            source_id, separator, path = raw_ref.partition(":")
            if not separator:
                continue
            evidence_id = ledger.add(
                source_id,
                path,
                names=(entity.name,),
                max_characters=excerpt_limit,
            )
            if evidence_id:
                evidence_ids.append(evidence_id)
        cards.append(
            {
                "id": entity_id,
                "name": entity.name,
                "kind": entity.kind,
                "evidence_ids": evidence_ids,
            }
        )
    components = [
        {
            "id": component.id,
            "name": component.name,
            "implementation": component.implementation,
            "disposition": component.disposition.value,
        }
        for _, component in _discovered_components(run.document.blocks)
    ]
    return (
        {
            "task": "judge_phase1_completeness",
            "run_id": run.run_id,
            "context": _context_card(context),
            "components": components,
            "workflow_actions": _workflow_action_cards(run.workflow.steps),
            "entities": cards,
            "evidence": list(ledger.entries.values()),
        },
        ledger.entries,
        entity_ids,
    )


def _workflow_action_cards(steps: list[DeploymentWorkflowStep]) -> list[dict[str, Any]]:
    return [
        {
            "id": step.id,
            "action_id": step.action_id,
            "name": step.action_name or step.id,
            "executor": step.executor.value,
            "source_ref": _source_ref(step.source_ref.repo_id, step.source_ref.path),
            "command": redact_secrets(step.command or ""),
            "status": step.status.value,
        }
        for step in steps
        if step.kind == WorkflowStepKind.ACTION
    ]


def _component_source_refs(
    component: FunctionalComponent,
    components: dict[str, FunctionalComponent],
    step: DeploymentWorkflowStep | None,
) -> tuple[AnalysisSourceRef, ...]:
    refs = []
    routed_component = components.get(_route_owner(component, components), component)
    if routed_component.deploy:
        refs.append(
            AnalysisSourceRef(
                repo_id=routed_component.deploy.repo_id,
                path=routed_component.deploy.ref,
            )
        )
    if step:
        refs.append(step.source_ref)
        if step.operation_source_ref:
            refs.append(step.operation_source_ref)
    unique = {(ref.repo_id, ref.path, ref.start_line, ref.end_line): ref for ref in refs}
    return tuple(unique.values())


def _context_card(context: DeploymentContext) -> dict[str, Any]:
    return {
        "system": context.system,
        "description": redact_secrets(context.description or ""),
        "profile": context.deployment.profile,
        "version": context.deployment.version,
        "provided_blocks": [
            {
                "type": block.type.value,
                "subtype": block.subtype,
                "implementation": block.implementation,
            }
            for block in context.provided_blocks
        ],
        "objectives": [redact_secrets(value) for value in context.objectives],
        "exclusions": [redact_secrets(value) for value in context.exclusions],
        "domain_roles": [redact_secrets(value) for value in context.domain_roles],
    }


def _apply_component_decisions(
    metric: MetricResult,
    decisions: dict[str, ComponentJudgeDecision],
    verdict_field: str,
    evidence: dict[str, dict[str, str]],
) -> MetricResult:
    findings = []
    for finding in metric.findings:
        decision = decisions.get(finding.subject)
        if decision is None or _is_hard_failure(metric.dimension, finding):
            findings.append(finding)
            continue
        verdict = getattr(decision, verdict_field)
        score = 1.0 if verdict == JudgeVerdict.SUPPORTED else 0.0
        findings.append(
            finding.model_copy(
                update={
                    "score": score,
                    "resolved": verdict != JudgeVerdict.INSUFFICIENT_EVIDENCE,
                    "reason": f"LLM judge: {decision.reason}",
                    "evidence_refs": (
                        _decision_refs(decision.evidence_ids, evidence) or finding.evidence_refs
                    ),
                }
            )
        )
    return _result(metric.dimension, findings)


def _apply_completeness_decisions(
    metric: MetricResult,
    decisions: list[CompletenessJudgeDecision],
    entities: dict[str, DeploymentEntity],
    evidence: dict[str, dict[str, str]],
) -> MetricResult:
    findings_by_subject = {finding.subject: finding for finding in metric.findings}
    findings = []
    for decision in decisions:
        entity = entities[decision.entity_id]
        finding = findings_by_subject[entity.name]
        included = decision.verdict != CompletenessVerdict.NOT_REQUIRED
        score = 1.0 if decision.verdict == CompletenessVerdict.REQUIRED_AND_ACCOUNTED else 0.0
        findings.append(
            finding.model_copy(
                update={
                    "score": score,
                    "included": included,
                    "resolved": decision.verdict != CompletenessVerdict.INSUFFICIENT_EVIDENCE,
                    "reason": f"LLM judge: {decision.reason}",
                    "evidence_refs": (
                        _decision_refs(decision.evidence_ids, evidence) or finding.evidence_refs
                    ),
                }
            )
        )
    return _result(metric.dimension, findings)


def _is_hard_failure(dimension: EvaluationDimension, finding: MetricFinding) -> bool:
    if finding.score != 0.0 or not finding.resolved:
        return False
    if dimension == EvaluationDimension.DEPLOYABILITY:
        return True
    hard_reasons = (
        "not unique",
        "does not exist",
        "provided infrastructure",
        "excluded profile",
        "conflicts with",
    )
    return any(reason in finding.reason for reason in hard_reasons)


def _decision_refs(
    evidence_ids: list[str],
    evidence: dict[str, dict[str, str]],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            evidence[evidence_id]["source_ref"]
            for evidence_id in evidence_ids
            if evidence_id in evidence
        )
    )


def _disabled_judge_summary() -> LLMJudgeSummary:
    return LLMJudgeSummary(enabled=False, complete=False)


def _load_run(run_directory: Path, expected_system: str) -> _RunArtifacts:
    directory = run_directory.expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"run directory does not exist: {directory}")
    document = _load_model(directory / _ARTIFACT_NAMES[0], FunctionalBlocksDocument)
    workflow = _load_model(directory / _ARTIFACT_NAMES[1], DeploymentWorkflow)
    systems = {expected_system, document.system, workflow.system}
    if len(systems) != 1:
        raise ValueError(f"run and deployment context system names disagree: {directory}")
    return _RunArtifacts(directory=directory, document=document, workflow=workflow)


def _load_model(path: Path, model_type):
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required evaluation input does not exist: {path}") from exc
    if payload is None:
        raise ValueError(f"required evaluation input is empty: {path}")
    return model_type.model_validate(payload)


def _discovered_components(
    blocks: list[FunctionalBlock],
) -> list[tuple[FunctionalBlock, FunctionalComponent]]:
    return [
        (block, component)
        for block in blocks
        if block.state != "provided"
        for component in block.components
    ]


def _component_batches(
    components: list[tuple[FunctionalBlock, FunctionalComponent]],
    batch_size: int,
) -> tuple[tuple[tuple[FunctionalBlock, FunctionalComponent], ...], ...]:
    return tuple(
        tuple(components[start : start + batch_size])
        for start in range(0, len(components), batch_size)
    )


def _duplicates(values) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _matching_entity(
    component: FunctionalComponent,
    entities: tuple[DeploymentEntity, ...],
) -> DeploymentEntity | None:
    names = tuple(filter(None, (component.name, component.implementation)))
    return next(
        (
            entity
            for entity in entities
            if any(identity_matches(name, entity.name) for name in names)
        ),
        None,
    )


def _looks_like_operation(name: str) -> bool:
    normalized = normalize_identity(name)
    return not normalized or bool(_OPERATION_LABEL.search(normalized))


def _matches_exclusion(component: FunctionalComponent, context: DeploymentContext) -> bool:
    identity = normalize_identity(component.name).replace("-", "")
    if len(identity) >= 5 and any(
        identity in normalize_identity(exclusion).replace("-", "")
        for exclusion in context.exclusions
    ):
        return True
    if component.deploy is None:
        return False
    path_terms = set(re.findall(r"[a-z0-9]+", component.deploy.ref.casefold()))
    active_profile = " ".join(
        filter(
            None,
            [
                context.deployment.profile,
                *(block.subtype for block in context.provided_blocks),
                *(block.implementation for block in context.provided_blocks),
            ],
        )
    ).casefold()
    excluded_profile_terms = {
        term
        for exclusion in context.exclusions
        for term in _PROFILE_TERMS
        if term in exclusion.casefold() and term not in active_profile
    }
    return bool(path_terms & excluded_profile_terms)


def _matches_provided_block(
    component: FunctionalComponent,
    context: DeploymentContext,
) -> bool:
    names = tuple(filter(None, (component.name, component.implementation)))
    return any(
        identity_matches(name, provided)
        for block in context.provided_blocks
        for provided in filter(None, (block.implementation, block.subtype))
        for name in names
    )


def _component_executor(
    component: FunctionalComponent,
    run: _RunArtifacts,
) -> AnalysisExecutor | None:
    if component.deploy is not None:
        return component.deploy.executor
    if not component.installed_by:
        return None
    owner = next(
        (
            candidate
            for _, candidate in _discovered_components(run.document.blocks)
            if candidate.id == component.installed_by
        ),
        None,
    )
    return owner.deploy.executor if owner and owner.deploy else None


def _executor_matches_mode(executor: AnalysisExecutor, mode: str) -> bool:
    if mode == "compose":
        return executor not in {
            AnalysisExecutor.HELM,
            AnalysisExecutor.KUBERNETES,
            AnalysisExecutor.KUSTOMIZE,
        }
    if mode == "kubernetes":
        return executor != AnalysisExecutor.DOCKER_COMPOSE
    return True


def _accounted_claims(
    run: _RunArtifacts,
    context: DeploymentContext,
) -> tuple[tuple[str, str], ...]:
    claims = [
        (name, "component")
        for _, component in _discovered_components(run.document.blocks)
        for name in filter(None, (component.name, component.implementation))
    ]
    claims.extend(
        (name, "provided prerequisite")
        for block in context.provided_blocks
        for name in filter(None, (block.implementation, block.subtype))
    )
    claims.extend(
        (step.action_name, "deployment action") for step in run.workflow.steps if step.action_name
    )
    claims.extend((exclusion, "excluded deployment profile") for exclusion in context.exclusions)
    claims.extend(
        (text, "unresolved entity")
        for question in (*run.document.unresolved, *run.workflow.unresolved)
        for text in (question.question, question.reason)
    )
    return tuple(claims)


def _steps_by_target(
    steps: list[DeploymentWorkflowStep],
) -> dict[str, tuple[DeploymentWorkflowStep, ...]]:
    result: dict[str, list[DeploymentWorkflowStep]] = {}
    for step in steps:
        for target in step.targets:
            result.setdefault(target.id, []).append(step)
    return {target: tuple(target_steps) for target, target_steps in result.items()}


def _route_owner(
    component: FunctionalComponent,
    components: dict[str, FunctionalComponent],
) -> str:
    visited = {component.id}
    current = component
    while current.installed_by and current.installed_by not in visited:
        visited.add(current.installed_by)
        owner = components.get(current.installed_by)
        if owner is None:
            return current.installed_by
        current = owner
    return current.id


def _missing_command_reference(
    step: DeploymentWorkflowStep,
    catalog: SourceCatalog,
) -> str | None:
    try:
        tokens = shlex.split(step.command or "")
    except ValueError:
        return "<unparseable command>"
    references: set[str] = set()
    for index, token in enumerate(tokens):
        cleaned = token.rstrip(";|&")
        if cleaned in {"-f", "--file", "--values", "bash", "sh"} and index + 1 < len(tokens):
            references.add(tokens[index + 1].rstrip(";|&"))
        if cleaned.startswith(("./", "../")):
            references.add(cleaned)
    for reference in sorted(references):
        if (
            not reference
            or reference == "-"
            or "://" in reference
            or "$" in reference
            or "{{" in reference
            or Path(reference).is_absolute()
        ):
            continue
        if not catalog.relative_file_exists(
            step.source_ref.repo_id,
            step.working_directory,
            reference,
        ):
            return reference
    return None


def _selected_source_paths(
    run: _RunArtifacts,
    context: DeploymentContext,
) -> dict[str, set[str]]:
    selected: dict[str, set[str]] = {}

    def add(source_id: str, path: str) -> None:
        if not path or "$" in path or "{{" in path or path.startswith("/"):
            return
        normalized = posixpath.normpath(path)
        if normalized == ".." or normalized.startswith("../"):
            return
        selected.setdefault(source_id, set()).add(normalized)

    for _block, component in _discovered_components(run.document.blocks):
        if component.deploy:
            add(component.deploy.repo_id, component.deploy.ref)
    for step in run.workflow.steps:
        add(step.source_ref.repo_id, step.source_ref.path)
        if step.operation_source_ref:
            add(step.operation_source_ref.repo_id, step.operation_source_ref.path)
        try:
            tokens = shlex.split(step.command or "")
        except ValueError:
            tokens = []
        for index, token in enumerate(tokens):
            if token in {"-f", "--file"} and index + 1 < len(tokens):
                add(
                    step.source_ref.repo_id,
                    posixpath.join(step.working_directory or ".", tokens[index + 1]),
                )
            if token.startswith(("-f=", "--file=")):
                add(
                    step.source_ref.repo_id,
                    posixpath.join(step.working_directory or ".", token.partition("=")[2]),
                )
    source_ids = set(selected)
    for hint in context.hints.documentation:
        for source_id in source_ids:
            add(source_id, hint)
    return selected


def _run_signature(run_directory: Path) -> str | None:
    path = run_directory / "run-manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError, json.JSONDecodeError:
        return None
    if manifest.get("status") != "completed":
        return None
    inputs = manifest.get("inputs", {})
    sources = manifest.get("sources", {}).get("sources", [])
    if not isinstance(inputs, dict) or not isinstance(sources, list) or not sources:
        return None
    source_signature = []
    for source in sources:
        if not isinstance(source, dict) or not source.get("id") or not source.get("content_hash"):
            return None
        source_signature.append(
            (
                source["id"],
                source.get("resolved_revision"),
                source["content_hash"],
            )
        )
    context = inputs.get("context")
    if isinstance(context, dict) and context.get("sha256"):
        context = context["sha256"]
    signature = {
        "analysis_method": manifest.get("analysis_method"),
        "context": context,
        "model": inputs.get("model"),
        "budgets": inputs.get("budgets"),
        "sources": sorted(source_signature),
    }
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def _projection_similarities(
    left: _RunArtifacts,
    right: _RunArtifacts,
) -> dict[str, float]:
    left_components = _component_names(left.document)
    right_components = _component_names(right.document)
    return {
        "components": _jaccard(left_components, right_components),
        "assignments": _mapping_similarity(
            _block_assignments(left.document),
            _block_assignments(right.document),
        ),
        "edges": _jaccard(_block_edges(left.document), _block_edges(right.document)),
        "routes": _mapping_similarity(_routes(left), _routes(right)),
    }


def _component_names(document: FunctionalBlocksDocument) -> set[str]:
    return {
        normalize_identity(component.name)
        for _, component in _discovered_components(document.blocks)
    }


def _block_assignments(document: FunctionalBlocksDocument) -> dict[str, str]:
    return {
        normalize_identity(component.name): f"{block.type.value}/{block.subtype}"
        for block, component in _discovered_components(document.blocks)
    }


def _block_edges(document: FunctionalBlocksDocument) -> set[tuple[str, str]]:
    by_id = {block.id: block for block in document.blocks}
    return {
        (
            f"{by_id[parent].type.value}/{by_id[parent].subtype}",
            f"{block.type.value}/{block.subtype}",
        )
        for block in document.blocks
        for parent in block.after
        if parent in by_id
    }


def _routes(run: _RunArtifacts) -> dict[str, str]:
    names = {
        component.id: normalize_identity(component.name)
        for _, component in _discovered_components(run.document.blocks)
    }
    return {
        names[target.id]: "|".join(
            (
                step.executor.value,
                " ".join((step.command or "").split()),
                step.source_ref.repo_id,
                step.source_ref.path,
                step.working_directory or ".",
            )
        )
        for step in run.workflow.steps
        for target in step.targets
        if target.id in names
    }


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _mapping_similarity(left: dict[str, str], right: dict[str, str]) -> float:
    keys = set(left) | set(right)
    return sum(left.get(key) == right.get(key) for key in keys) / len(keys) if keys else 1.0


def _finding(
    run: _RunArtifacts,
    subject: str,
    score: float,
    reason: str,
    evidence_refs: tuple[str, ...] = (),
    *,
    resolved: bool = True,
) -> MetricFinding:
    return MetricFinding(
        run_id=run.run_id,
        subject=subject,
        score=score,
        resolved=resolved,
        reason=reason,
        evidence_refs=evidence_refs,
    )


def _result(
    dimension: EvaluationDimension,
    findings: list[MetricFinding],
) -> MetricResult:
    assessed = [finding for finding in findings if finding.included]
    issues = tuple(
        finding for finding in assessed if finding.score < 1.0 or not finding.resolved
    )
    if not assessed:
        return MetricResult(
            dimension=dimension,
            explanation="the metric has no assessable items",
            issues=issues,
            findings=tuple(findings),
        )
    return MetricResult(
        dimension=dimension,
        score=fmean(finding.score for finding in assessed),
        assessed_items=len(assessed),
        unresolved_items=sum(not finding.resolved for finding in assessed),
        issues=issues,
        findings=tuple(findings),
    )


def _unavailable(dimension: EvaluationDimension, explanation: str) -> MetricResult:
    return MetricResult(dimension=dimension, explanation=explanation)


def _combine(
    dimension: EvaluationDimension,
    results: list[MetricResult],
) -> MetricResult:
    findings = [finding for result in results for finding in result.findings]
    if findings:
        return _result(dimension, findings)
    explanations = tuple(
        dict.fromkeys(result.explanation for result in results if result.explanation)
    )
    return _unavailable(dimension, "; ".join(explanations) or "the metric is unavailable")


def _deploy_ref(component: FunctionalComponent) -> str:
    if component.deploy is None:
        return ""
    return _source_ref(component.deploy.repo_id, component.deploy.ref)


def _source_ref(source_id: str, path: str) -> str:
    return f"{source_id}:{path}"
