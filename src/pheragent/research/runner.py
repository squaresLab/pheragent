from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import yaml
from pydantic import Field

from pheragent.deployment.analysis_models import DeploymentContext, GoldDefinition
from pheragent.deployment.analyzer import AnalysisConfig, run_repository_analysis
from pheragent.deployment.artifacts import analysis_metrics, publish_analysis_artifacts
from pheragent.deployment.enums import AnalysisTreatment, SourceKind
from pheragent.deployment.models import ContractModel, SourcesConfig
from pheragent.deployment.output import create_timestamped_run_directory
from pheragent.deployment.run_records import RunRecorder
from pheragent.deployment.serialization import load_sources_config

from .evaluation import summarize_study

ProgressCallback = Callable[[str], None]


class ResearchCase(ContractModel):
    id: str = Field(min_length=1)
    sources: Path
    context: Path
    invariants: Path | None = None


class ResearchBudgets(ContractModel):
    node: int = Field(default=200, ge=1)
    llm_requests: int = Field(default=2, ge=1)
    llm_output_tokens: int = Field(default=5000, ge=1)
    evidence_observations: int = Field(default=32, ge=1)
    evidence_characters: int = Field(default=18_000, ge=1)


class ResearchStudy(ContractModel):
    version: str = "0.1"
    id: str = Field(min_length=1)
    model: str = "gpt-4o-mini"
    repetitions: int = Field(default=1, ge=1)
    max_total_llm_requests: int = Field(default=48, ge=0)
    treatments: list[AnalysisTreatment] = Field(
        default_factory=lambda: list(AnalysisTreatment)
    )
    budgets: ResearchBudgets = Field(default_factory=ResearchBudgets)
    cases: list[ResearchCase] = Field(min_length=1)


def load_study(path: Path) -> ResearchStudy:
    payload = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("research study must contain a YAML mapping")
    return ResearchStudy.model_validate(payload)


def describe_study(
    study: ResearchStudy,
    *,
    selected_cases: set[str] | None = None,
    selected_treatments: set[AnalysisTreatment] | None = None,
    repetitions: int | None = None,
) -> dict[str, int]:
    cases = _select_cases(study, selected_cases)
    treatments = _select_treatments(study, selected_treatments)
    repeat_count = repetitions or study.repetitions
    runs = len(cases) * len(treatments) * repeat_count
    llm_runs = sum(treatment != AnalysisTreatment.DETERMINISTIC for treatment in treatments)
    requests = len(cases) * llm_runs * repeat_count * study.budgets.llm_requests
    if requests > study.max_total_llm_requests:
        raise ValueError(
            f"study requests at most {requests} LLM calls, above its ceiling of "
            f"{study.max_total_llm_requests}"
        )
    return {"runs": runs, "maximum_llm_requests": requests}


def validate_study_inputs(
    study_path: Path,
    study: ResearchStudy,
    *,
    selected_cases: set[str] | None = None,
) -> None:
    """Fail preflight before acquisition if a case contract is missing, invalid, or unpinned."""
    base = study_path.expanduser().resolve().parent
    for case in _select_cases(study, selected_cases):
        paths = _resolve_case_paths(base, case)
        _load_pinned_sources(paths.sources)
        validate_invariants(paths.invariants)
        payload = yaml.safe_load(paths.context.read_text(encoding="utf-8"))
        DeploymentContext.model_validate(payload)


def run_study(
    study_path: Path,
    output_root: Path,
    *,
    selected_cases: set[str] | None = None,
    selected_treatments: set[AnalysisTreatment] | None = None,
    repetitions: int | None = None,
    refresh_llm: bool = False,
    debug: bool = False,
    progress: ProgressCallback | None = None,
) -> tuple[int, Path]:
    study_file = study_path.expanduser().resolve()
    study = load_study(study_file)
    validate_study_inputs(study_file, study, selected_cases=selected_cases)
    describe_study(
        study,
        selected_cases=selected_cases,
        selected_treatments=selected_treatments,
        repetitions=repetitions,
    )
    cases = _select_cases(study, selected_cases)
    treatments = _select_treatments(study, selected_treatments)
    repeat_count = repetitions or study.repetitions
    study_root = output_root.expanduser().resolve() / study.id
    notify = progress or (lambda _message: None)
    failures = 0

    for case in cases:
        case_paths = _resolve_case_paths(study_file.parent, case)
        validate_invariants(case_paths.invariants)
        source_config = _load_pinned_sources(case_paths.sources)
        for repetition in range(1, repeat_count + 1):
            for treatment in treatments:
                label = f"{case.id}-{treatment.value}-r{repetition:02d}"
                run_dir = create_timestamped_run_directory(study_root, name=label)
                recorder = RunRecorder.start(
                    run_dir,
                    run_kind="research",
                    analysis_method=treatment.value,
                    inputs={
                        "study": study.id,
                        "case": case.id,
                        "repetition": repetition,
                        "sources": case_paths.sources,
                        "context": case_paths.context,
                        "invariants": case_paths.invariants,
                        "model": study.model,
                        "budgets": study.budgets.model_dump(mode="json"),
                    },
                )

                try:
                    result = run_repository_analysis(
                        _analysis_config(
                            study,
                            case_paths,
                            source_config,
                            treatment,
                            study_root,
                            refresh_llm=refresh_llm,
                        ),
                        progress=_run_progress(recorder, label, notify),
                    )
                    publish_analysis_artifacts(run_dir, result, debug=debug)
                    recorder.complete(
                        metrics=analysis_metrics(result),
                        sources=result.acquisition.manifest.model_dump(mode="json"),
                        llm={"usage": result.llm_usage, "stages": result.llm_stage_statuses},
                    )
                except Exception as exc:
                    failures += 1
                    recorder.fail(exc)
                    notify(f"{label}: failed: {exc}")

    summarize_study(study_root)
    return failures, study_root


def _analysis_config(
    study: ResearchStudy,
    case: ResearchCase,
    sources: SourcesConfig,
    treatment: AnalysisTreatment,
    study_root: Path,
    *,
    refresh_llm: bool,
) -> AnalysisConfig:
    budgets = study.budgets
    return AnalysisConfig(
        repositories=[],
        documentation=[],
        sources=sources,
        context_path=case.context,
        cache_dir=study_root / ".source-cache",
        gold_path=case.invariants,
        strict=True,
        model=study.model,
        llm_max_requests=budgets.llm_requests,
        llm_max_output_tokens=budgets.llm_output_tokens,
        llm_cache_dir=study_root / ".llm-cache",
        refresh_llm=refresh_llm,
        node_budget=budgets.node,
        investigation_max_observations=budgets.evidence_observations,
        investigation_max_evidence_chars=budgets.evidence_characters,
        treatment=treatment,
    )


def _load_pinned_sources(path: Path) -> SourcesConfig:
    sources = load_sources_config(path)
    unpinned = [
        source.id
        for source in sources.sources
        if source.kind == SourceKind.GIT and not source.revision
    ]
    if unpinned:
        raise ValueError("research sources must be pinned: " + ", ".join(sorted(unpinned)))
    return sources


def _select_cases(study: ResearchStudy, selected: set[str] | None) -> list[ResearchCase]:
    if not selected:
        return study.cases
    cases = [case for case in study.cases if case.id in selected]
    missing = sorted(selected - {case.id for case in cases})
    if missing:
        raise ValueError("unknown research cases: " + ", ".join(missing))
    return cases


def _select_treatments(
    study: ResearchStudy,
    selected: set[AnalysisTreatment] | None,
) -> list[AnalysisTreatment]:
    if not selected:
        return study.treatments
    treatments = [treatment for treatment in study.treatments if treatment in selected]
    missing = sorted(item.value for item in selected - set(treatments))
    if missing:
        raise ValueError("treatments are not enabled by this study: " + ", ".join(missing))
    return treatments


def _resolve_case_paths(base: Path, case: ResearchCase) -> ResearchCase:
    return ResearchCase(
        id=case.id,
        sources=_resolve(base, case.sources),
        context=_resolve(base, case.context),
        invariants=_resolve(base, case.invariants) if case.invariants else None,
    )


def _resolve(base: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (base / path).resolve()


def _run_progress(
    recorder: RunRecorder,
    label: str,
    notify: ProgressCallback,
) -> ProgressCallback:
    def report(message: str) -> None:
        recorder.record_event("analysis", message)
        notify(f"{label}: {message}")

    return report


def validate_invariants(path: Path | None) -> None:
    if path is None:
        return
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    GoldDefinition.model_validate(payload)
