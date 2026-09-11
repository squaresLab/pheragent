from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from pheragent.utils import slugify

from .analysis_llm import (
    DEFAULT_ANALYSIS_MODEL,
    AnalysisLLMConfig,
    CachedStructuredClassifier,
    LLMRequestBudget,
    aggregate_usage,
)
from .analysis_models import (
    AnalysisRelation,
    DeploymentContext,
    DeploymentSignalBundle,
    DeploymentWorkflow,
    FunctionalBlocksDocument,
    GoldDefinition,
    ReferenceNode,
    RepositoryFile,
)
from .discovery import (
    apply_investigation_synthesis,
    bind_retrieved_installation_routes,
    discover_repository,
)
from .enums import AnalysisTreatment, SourceKind
from .evidence import (
    EvidenceBudget,
    collect_investigation_evidence,
    collect_query_evidence,
    merge_investigation_evidence,
    schedule_investigation_queries,
)
from .functional_blocks import build_functional_blocks
from .inventory import RepositoryInventoryBuilder
from .investigation import (
    InvestigationResult,
    build_follow_up_queries,
    build_plan_input,
    build_synthesis_input,
    plan_investigation_with_llm,
    reconcile_investigation_synthesis,
    synthesize_investigation_with_llm,
)
from .investigation_models import InvestigationQuery, SourcePurpose, default_investigation_plan
from .knowledge_graph import (
    DeploymentKnowledgeGraph,
    graph_gap_queries,
    project_deployment_knowledge,
)
from .models import RepositoryInventory, SourcesConfig, SourceSpec
from .runtime_context import RuntimeContextSnapshot, compact_runtime_context
from .source_manager import AcquisitionResult, SourceManager
from .workflow import build_deployment_workflow

ProgressCallback = Callable[[str], None]
_MAX_FOLLOW_UP_OBSERVATIONS = 8
_MAX_FOLLOW_UP_EVIDENCE_CHARACTERS = 4_000


@dataclass(slots=True)
class AnalysisConfig:
    repositories: list[str]
    documentation: list[str]
    context_path: Path
    cache_dir: Path
    node_budget: int = 200
    strict: bool = False
    source_timeout: float = 900.0
    gold_path: Path | None = None
    model: str = DEFAULT_ANALYSIS_MODEL
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    base_url: str | None = None
    llm_timeout: float = 120.0
    llm_max_output_tokens: int = 5000
    llm_max_requests: int = 2
    llm_cache_dir: Path | None = None
    retry_failed_llm: bool = False
    refresh_llm: bool = False
    llm_reasoning_effort: str | None = None
    llm_enabled: bool = True
    investigation_max_observations: int = 32
    investigation_max_evidence_chars: int = 18_000
    treatment: AnalysisTreatment = AnalysisTreatment.HYBRID
    sources: SourcesConfig | None = None
    runtime_context: RuntimeContextSnapshot | None = None


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    context: DeploymentContext
    acquisition: AcquisitionResult
    inventory: RepositoryInventory
    repository_files: tuple[RepositoryFile, ...]
    investigation: InvestigationResult
    workflow: DeploymentWorkflow
    reference_nodes: tuple[ReferenceNode, ...]
    reference_relations: tuple[AnalysisRelation, ...]
    signals: DeploymentSignalBundle
    document: FunctionalBlocksDocument
    selected_paths: tuple[str, ...]
    warnings: tuple[str, ...]
    llm_usage: dict[str, int]
    llm_stage_statuses: dict[str, str]
    llm_failure_history_paths: tuple[Path, ...]
    knowledge_graph: DeploymentKnowledgeGraph | None
    graph_queries: tuple[str, ...]
    retrieval_queries: tuple[InvestigationQuery, ...]
    evidence_characters: int
    runtime_context: RuntimeContextSnapshot | None


def run_repository_analysis(
    config: AnalysisConfig,
    *,
    progress: ProgressCallback | None = None,
) -> AnalysisResult:
    return RepositoryAnalysisPipeline(config, progress=progress).run()


class RepositoryAnalysisPipeline:
    """Orchestrates Phase 1 while keeping discovery and model judgment separate."""

    def __init__(
        self,
        config: AnalysisConfig,
        *,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.config = config
        self.notify = progress or (lambda _message: None)

    def run(self) -> AnalysisResult:
        config = self.config
        notify = self.notify
        context = _load_context(config.context_path)
        sources_config = config.sources or _sources_config(
            config.repositories,
            config.documentation,
            context.system,
        )
        source_purposes = _source_purposes(sources_config)
        notify(f"acquiring {len(sources_config.sources)} source(s)")
        acquisition = SourceManager(
            cache_dir=config.cache_dir,
            config_dir=config.context_path.expanduser().resolve().parent,
            strict=config.strict,
            timeout=config.source_timeout,
            progress=notify,
        ).acquire(sources_config)
        notify("inventorying repositories")
        inventory = RepositoryInventoryBuilder().build(acquisition.sources)
        source_by_id = {source.id: source for source in acquisition.sources}
        discovery = discover_repository(
            inventory,
            context,
            source_by_id,
            source_purposes,
            node_budget=config.node_budget,
        )
        files = list(discovery.files)
        nodes = discovery.reference_nodes
        reference_relations = discovery.reference_relations
        roots = discovery.roots
        selected = discovery.selected_paths
        signals = discovery.signals
        warnings = list(discovery.warnings)
        outlines = discovery.outlines
        notify(f"extracting deterministic evidence from {len(files)} file(s)")

        knowledge_graph = None
        guided_queries = ()
        if config.treatment == AnalysisTreatment.HYBRID_GRAPH:
            knowledge_graph = project_deployment_knowledge(
                signals,
                list(nodes),
                list(reference_relations),
            )
            guided_queries = graph_gap_queries(knowledge_graph)
            notify(
                f"projected {len(knowledge_graph.nodes)} graph node(s), "
                f"{len(knowledge_graph.edges)} edge(s), and {len(knowledge_graph.gaps)} gap(s)"
            )

        llm_config = AnalysisLLMConfig(
            enabled=(
                config.llm_enabled
                and config.treatment != AnalysisTreatment.DETERMINISTIC
            ),
            model=config.model,
            api_key_env=config.api_key_env,
            base_url_env=config.base_url_env,
            base_url=config.base_url,
            timeout=config.llm_timeout,
            max_output_tokens=config.llm_max_output_tokens,
            max_requests=config.llm_max_requests,
            cache_dir=config.llm_cache_dir,
            retry_failed=config.retry_failed_llm,
            refresh_cache=config.refresh_llm,
            reasoning_effort=config.llm_reasoning_effort,
        )
        request_budget = LLMRequestBudget(limit=config.llm_max_requests)
        classifier = CachedStructuredClassifier(llm_config, request_budget)

        notify(f"selected {len(roots)} deployment root(s) and {len(selected)} evidence node(s)")
        runtime_context = compact_runtime_context(config.runtime_context)
        plan_input = build_plan_input(
            context,
            outlines,
            signals,
            runtime_context=runtime_context,
        )
        notify(f"planning bounded evidence investigation from {len(outlines)} source outline(s)")
        plan_outcome = plan_investigation_with_llm(plan_input, classifier=classifier)
        plan = plan_outcome.value or default_investigation_plan()
        if plan_outcome.warning:
            warnings.append(plan_outcome.warning)

        retrieval_queries = schedule_investigation_queries(
            plan,
            guided_queries=guided_queries,
        )
        notify(f"running {len(retrieval_queries)} bounded read-only retrieval queries")
        observations, mandatory_probes = collect_investigation_evidence(
            queries=retrieval_queries,
            sources=source_by_id,
            source_purposes=source_purposes,
            searchable_paths=discovery.searchable_paths,
            retrieval_documents=discovery.retrieval_documents,
            reference_relations=discovery.reference_relations,
            roots=[root.source_ref for root in roots],
            components=signals.candidate_components,
            budget=EvidenceBudget(
                max_observations=config.investigation_max_observations,
                max_characters=config.investigation_max_evidence_chars,
            ),
        )
        signals, route_warnings = bind_retrieved_installation_routes(
            signals,
            observations,
            source_by_id,
        )
        warnings.extend(route_warnings)
        notify(
            f"collected {len(observations)} redacted evidence observation(s) "
            f"under a {config.investigation_max_evidence_chars}-character budget"
        )
        synthesis_input = build_synthesis_input(
            context,
            signals,
            observations,
            mandatory_probes,
            runtime_context=runtime_context,
        )
        notify("synthesizing grounded deployment facts and component decisions")
        synthesis_outcome = synthesize_investigation_with_llm(
            synthesis_input,
            components=signals.candidate_components,
            context=context,
            evidence_ids={observation.id for observation in observations},
            classifier=classifier,
        )
        synthesis = synthesis_outcome.value
        if synthesis is not None:
            synthesis, reconciliation_warnings = reconcile_investigation_synthesis(
                synthesis,
                components=signals.candidate_components,
                context=context,
                evidence_ids={observation.id for observation in observations},
            )
            warnings.extend(reconciliation_warnings)
        if synthesis_outcome.warning:
            warnings.append(synthesis_outcome.warning)

        settled_signals = (
            apply_investigation_synthesis(signals, synthesis, observations)
            if synthesis is not None
            else signals
        )
        outcomes = [plan_outcome, synthesis_outcome]
        follow_up_queries = build_follow_up_queries(
            synthesis.unresolved if synthesis else [],
            settled_signals.candidate_components,
        )
        follow_up_outcome = None
        follow_up_status = "not_requested_no_synthesis"
        follow_up_accepted = False
        unresolved_before_follow_up = len(synthesis.unresolved) if synthesis else 0
        novel_evidence = ()
        if follow_up_queries:
            notify(f"probing {len(follow_up_queries)} uncovered deployment fact(s)")
            focused_evidence = collect_query_evidence(
                queries=follow_up_queries,
                source_purposes=source_purposes,
                retrieval_documents=discovery.retrieval_documents,
                reference_relations=discovery.reference_relations,
                components=settled_signals.candidate_components,
                budget=EvidenceBudget(
                    max_observations=min(
                        _MAX_FOLLOW_UP_OBSERVATIONS,
                        config.investigation_max_observations,
                    ),
                    max_characters=min(
                        _MAX_FOLLOW_UP_EVIDENCE_CHARACTERS,
                        config.investigation_max_evidence_chars,
                    ),
                    results_per_query=1,
                ),
            )
            known_evidence = {observation.id for observation in observations}
            novel_evidence = tuple(
                observation
                for observation in focused_evidence
                if observation.id not in known_evidence
            )
            if novel_evidence:
                observations = merge_investigation_evidence(
                    observations,
                    novel_evidence,
                    budget=EvidenceBudget(
                        max_observations=config.investigation_max_observations,
                        max_characters=config.investigation_max_evidence_chars,
                    ),
                )
        if synthesis is not None and not synthesis.unresolved:
            follow_up_status = "not_requested_no_unresolved"
        elif synthesis is not None and (
            config.llm_max_requests < 3 or request_budget.remaining < 1
        ):
            follow_up_status = "not_requested_budget_exhausted"
        elif synthesis is not None and not follow_up_queries:
            follow_up_status = "not_requested_no_queries"
        elif synthesis is not None and not novel_evidence:
            follow_up_status = "not_requested_no_new_evidence"
        elif synthesis is not None:
            follow_up_input = build_synthesis_input(
                context,
                settled_signals,
                observations,
                mandatory_probes,
                questions_to_recheck=synthesis.unresolved,
                runtime_context=runtime_context,
            )
            notify("re-synthesizing unresolved deployment questions")
            follow_up_outcome = synthesize_investigation_with_llm(
                follow_up_input,
                components=settled_signals.candidate_components,
                context=context,
                evidence_ids={observation.id for observation in observations},
                classifier=classifier,
                stage="investigation_follow_up_synthesis",
            )
            outcomes.append(follow_up_outcome)
            if follow_up_outcome.value is None:
                follow_up_status = "failed"
            else:
                follow_up_synthesis, follow_up_warnings = reconcile_investigation_synthesis(
                    follow_up_outcome.value,
                    components=settled_signals.candidate_components,
                    context=context,
                    evidence_ids={observation.id for observation in observations},
                )
                warnings.extend(follow_up_warnings)
                if len(follow_up_synthesis.unresolved) < len(synthesis.unresolved):
                    synthesis = follow_up_synthesis
                    synthesis_outcome = follow_up_outcome
                    synthesis_input = follow_up_input
                    settled_signals = apply_investigation_synthesis(
                        settled_signals,
                        synthesis,
                        observations,
                    )
                    follow_up_status = "accepted"
                    follow_up_accepted = True
                else:
                    follow_up_status = "rejected_no_improvement"
            if follow_up_outcome.warning:
                warnings.append(follow_up_outcome.warning)

        signals = settled_signals

        retrieval_queries = (*retrieval_queries, *follow_up_queries)

        investigation = InvestigationResult(
            plan_input=plan_input,
            plan=plan,
            plan_outcome=plan_outcome,
            observations=observations,
            mandatory_probes=mandatory_probes,
            synthesis_input=synthesis_input,
            synthesis=synthesis,
            synthesis_outcome=synthesis_outcome,
            follow_up_queries=follow_up_queries,
            follow_up_outcome=follow_up_outcome,
            follow_up_status=follow_up_status,
            follow_up_accepted=follow_up_accepted,
            unresolved_before_follow_up=unresolved_before_follow_up,
        )
        workflow = build_deployment_workflow(
            context=context,
            signals=signals,
            observations=observations,
            synthesis=synthesis,
            llm_completed=synthesis is not None,
            mandatory_probes=mandatory_probes,
            documentation_expected=(
                SourcePurpose.DOCUMENTATION in source_purposes.values()
            ),
        )

        gold = _load_gold(config.gold_path)
        usage = aggregate_usage(*outcomes)
        statuses = {outcome.stage: outcome.status for outcome in outcomes}
        document = build_functional_blocks(
            context,
            signals,
            source_by_id,
            gold,
            llm_usage=usage,
            llm_stage_statuses=statuses,
        )
        failure_paths = tuple(
            path
            for path in (outcome.failure_history_path for outcome in outcomes)
            if path is not None
        )
        return AnalysisResult(
            context=context,
            acquisition=acquisition,
            inventory=inventory,
            repository_files=tuple(files),
            investigation=investigation,
            workflow=workflow,
            reference_nodes=tuple(nodes),
            reference_relations=tuple(reference_relations),
            signals=signals,
            document=document,
            selected_paths=tuple(sorted(selected)),
            warnings=tuple(warnings),
            llm_usage=usage,
            llm_stage_statuses=statuses,
            llm_failure_history_paths=failure_paths,
            knowledge_graph=knowledge_graph,
            graph_queries=tuple(query.reason_code for query in guided_queries),
            retrieval_queries=retrieval_queries,
            evidence_characters=sum(
                len(observation.excerpt or "") for observation in observations
            ),
            runtime_context=config.runtime_context,
        )


def _load_context(path: Path) -> DeploymentContext:
    resolved = path.expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("deployment context must contain a YAML mapping")
    return DeploymentContext.model_validate(payload)


def _load_gold(path: Path | None) -> GoldDefinition | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("gold definition must contain a YAML mapping")
    return GoldDefinition.model_validate(payload)


def _sources_config(
    repositories: list[str], documentation: list[str], system: str
) -> SourcesConfig:
    specs: list[SourceSpec] = []
    used_ids: set[str] = set()
    for kind, locations in (("repo", repositories), ("docs", documentation)):
        for index, location in enumerate(locations, start=1):
            candidate = _source_id(location, fallback=f"{kind}-{index}")
            source_id = candidate
            suffix = 2
            while source_id in used_ids:
                source_id = f"{candidate}-{suffix}"
                suffix += 1
            used_ids.add(source_id)
            source_kind = _source_kind(location)
            resolved_location = (
                str(Path(location).expanduser().resolve())
                if source_kind != SourceKind.GIT
                else location
            )
            specs.append(
                SourceSpec(
                    id=source_id,
                    kind=source_kind,
                    location=resolved_location,
                    purpose=("repository" if kind == "repo" else "documentation"),
                )
            )
    if not specs:
        raise ValueError("deployment analysis requires at least one repository or docs source")
    return SourcesConfig(system=system, sources=specs)


def _source_kind(location: str) -> SourceKind:
    if urlsplit(location).scheme in {"git", "http", "https", "ssh"} or location.startswith("git@"):
        return SourceKind.GIT
    path = Path(location).expanduser()
    return SourceKind.LOCAL_FILE if path.is_file() else SourceKind.LOCAL_DIRECTORY


def _source_purposes(config: SourcesConfig) -> dict[str, SourcePurpose]:
    return {source.id: SourcePurpose(source.purpose) for source in config.sources}


def _source_id(location: str, *, fallback: str) -> str:
    parsed = urlsplit(location)
    raw = Path(parsed.path or location).name.removesuffix(".git") or fallback
    return slugify(raw, fallback=fallback)
