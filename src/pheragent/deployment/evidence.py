from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .analysis_models import (
    AnalysisSourceRef,
    ArtifactRole,
    CandidateComponent,
    ComponentEvidenceStrength,
)
from .investigation_models import (
    EvidenceKind,
    EvidenceObservation,
    EvidenceStrength,
    InvestigationPlan,
    InvestigationPurpose,
    InvestigationQuery,
    SourcePurpose,
    SourceScope,
    default_investigation_plan,
)
from .redaction import redact_secrets
from .retrieval import DeploymentRetrievalEngine, RetrievalQuery
from .source_manager import AcquiredSource

_PROMPT_INJECTION = re.compile(
    r"(?i)(?:ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions|"
    r"reveal\s+(?:the\s+)?system\s+prompt|act\s+as\s+(?:the\s+)?system|"
    r"upload\s+[^\n]*(?:key|credential|secret)|"
    r"read\s+[^\n]*(?:\.ssh|id_rsa|id_ed25519|credentials))"
)
_DYNAMIC_PATTERNS: tuple[tuple[str, re.Pattern[str], bool], ...] = (
    (
        "dynamic_variable_assignment",
        re.compile(r"(?m)^\s*eval\s+\$\{?[A-Za-z_][A-Za-z0-9_]*\}?\s*="),
        False,
    ),
    (
        "dynamic_command",
        re.compile(r"(?m)^\s*eval\b(?!\s+\$\{?[A-Za-z_][A-Za-z0-9_]*\}?\s*=)"),
        True,
    ),
    (
        "remote_script_pipe",
        re.compile(r"(?i)\b(?:curl|wget)\b[^\n|]*\|\s*(?:ba)?sh\b"),
        True,
    ),
    ("process_substitution", re.compile(r"(?i)\b(?:source|bash|sh)\s+<\("), True),
    (
        "indirect_command",
        re.compile(r"(?m)^\s*[\"']?\$\{?[A-Z_]*(?:CMD|COMMAND)[A-Z_]*\}?"),
        True,
    ),
)


@dataclass(frozen=True, slots=True)
class EvidenceBudget:
    max_observations: int = 32
    max_characters: int = 18_000
    lines_before: int = 2
    lines_after: int = 4
    results_per_query: int = 2
    max_excerpt_characters: int = 1_200


def collect_investigation_evidence(
    *,
    queries: tuple[InvestigationQuery, ...],
    sources: dict[str, AcquiredSource],
    source_purposes: dict[str, SourcePurpose],
    searchable_paths: dict[str, list[str]],
    roots: list[AnalysisSourceRef],
    components: list[CandidateComponent],
    budget: EvidenceBudget | None = None,
) -> tuple[tuple[EvidenceObservation, ...], tuple[str, ...]]:
    budget = budget or EvidenceBudget()
    collector = _EvidenceCollector(budget)
    retriever = _build_retriever(
        sources=sources,
        purposes=source_purposes,
        searchable_paths=searchable_paths,
    )
    references = [*roots, *(component.source_ref for component in components)]
    for root in roots:
        collector.add_reference(
            sources=sources,
            purposes=source_purposes,
            reference=root,
            kind=EvidenceKind.PRIMARY_ROOT,
            strength=EvidenceStrength.AUTHORITATIVE,
            summary="Selected deployment root",
        )
    collector.scan_dynamic_deployments(
        sources=sources,
        purposes=source_purposes,
        references=references,
        searchable_paths=searchable_paths,
    )

    # Reserve a small part of the ledger for concrete candidate entrypoints.
    # Without this reservation, broad searches can consume the entire budget.
    component_references = sorted(
        components,
        key=lambda item: (
            not item.existence_locked,
            item.evidence_strength == ComponentEvidenceStrength.INFERRED,
            item.source_ref.repo_id,
            item.source_ref.path,
        ),
    )[:6]
    for component in component_references:
        collector.add_reference(
            sources=sources,
            purposes=source_purposes,
            reference=component.source_ref,
            kind=EvidenceKind.COMPONENT_ENTRYPOINT,
            strength=(
                EvidenceStrength.AUTHORITATIVE
                if component.existence_locked
                else EvidenceStrength.SUPPORTING
            ),
            summary=f"Deployment evidence for {component.name}",
        )

    # Retrieve execution routes for the least actionable candidates first. The
    # query is component-scoped, but ranking and structural expansion are generic.
    route_candidates = sorted(
        (
            component
            for component in components
            if component.deployable
            and component.deployment is not None
            and not component.deployment.command
            and ArtifactRole.DEPLOYMENT_GUIDE not in component.artifact_roles
        ),
        key=lambda component: (
            PurePosixPath(component.deployment.entrypoint).suffix.casefold() in {".sh", ".bash"},
            not component.existence_locked,
            component.id,
        ),
    )[:12]
    for component in route_candidates:
        collector.retrieve_component_route(
            component=component,
            retriever=retriever,
            sources=sources,
            purposes=source_purposes,
        )

    referenced_entrypoints = {
        (component.source_ref.repo_id, component.deployment.entrypoint)
        for component in components
        if component.deployment is not None
    }
    collector.add_unreferenced_installers(
        sources=sources,
        purposes=source_purposes,
        searchable_paths=searchable_paths,
        referenced=referenced_entrypoints,
        max_results=3,
    )

    # Mandatory probes run alongside a bounded subset of model-selected terms.
    # This prevents known-name bias without allowing searches to crowd out roots.
    component_by_id = {component.id: component for component in components}
    for query in queries:
        target = component_by_id.get(query.component_id or "")
        collector.search(
            query=query,
            retriever=retriever,
            sources=sources,
            purposes=source_purposes,
            seed_paths=(
                (target.deployment.entrypoint,)
                if target is not None and target.deployment is not None
                else ()
            ),
        )
    probes = (
        "primary_roots",
        "component_entrypoints",
        "unreferenced_installers",
        "documentation_requirements",
        "configuration_endpoints",
        "dynamic_deployments",
        "profile_conflicts",
    )
    return tuple(collector.observations), probes


class _EvidenceCollector:
    def __init__(self, budget: EvidenceBudget):
        self.budget = budget
        self.observations: list[EvidenceObservation] = []
        self._characters = 0
        self._keys: set[tuple[str, str, int, EvidenceKind]] = set()

    def add_reference(
        self,
        *,
        sources: dict[str, AcquiredSource],
        purposes: dict[str, SourcePurpose],
        reference: AnalysisSourceRef,
        kind: EvidenceKind,
        strength: EvidenceStrength,
        summary: str,
    ) -> None:
        lines = _read_source_lines(sources[reference.repo_id], reference.path)
        if not lines:
            return
        line = reference.start_line or 1
        self._add_excerpt(
            source_id=reference.repo_id,
            purpose=purposes[reference.repo_id],
            path=reference.path,
            lines=lines,
            line=line,
            kind=kind,
            strength=strength,
            summary=summary,
        )

    def search(
        self,
        *,
        query: InvestigationQuery,
        retriever: DeploymentRetrievalEngine,
        sources: dict[str, AcquiredSource],
        purposes: dict[str, SourcePurpose],
        seed_paths: tuple[str, ...] = (),
    ) -> None:
        kind = (
            EvidenceKind.DOCUMENTATION_REQUIREMENT
            if query.purpose == InvestigationPurpose.EXTERNAL_REQUIREMENTS
            else EvidenceKind.INSTALLATION_ROUTE
            if query.purpose == InvestigationPurpose.DEPLOYMENT_ENTRYPOINTS
            else EvidenceKind.CONFIGURATION_REFERENCE
            if query.purpose == InvestigationPurpose.MISSING_COMPONENTS
            else EvidenceKind.SEARCH_RESULT
        )
        source_kind = None if query.source_scope == SourceScope.BOTH else query.source_scope.value
        hits = retriever.search(
            RetrievalQuery(
                terms=tuple(query.terms),
                source_kind=source_kind,
                path_prefix=query.path_prefix,
                seed_paths=seed_paths,
            ),
            limit=self.budget.results_per_query,
        )
        for hit in hits:
            self._add_excerpt(
                source_id=hit.document.source_id,
                purpose=purposes[hit.document.source_id],
                path=hit.document.path,
                lines=_read_source_lines(sources[hit.document.source_id], hit.document.path),
                line=hit.line,
                kind=kind,
                strength=EvidenceStrength.SUPPORTING,
                summary=f"Ranked retrieval for {', '.join(query.terms)}",
                query_id=(f"component-{query.component_id}" if query.component_id else query.id),
            )

    def retrieve_component_route(
        self,
        *,
        component: CandidateComponent,
        retriever: DeploymentRetrievalEngine,
        sources: dict[str, AcquiredSource],
        purposes: dict[str, SourcePurpose],
    ) -> None:
        assert component.deployment is not None
        terms = tuple(
            dict.fromkeys(
                (
                    component.name,
                    *component.aliases,
                    *component.materialized_names,
                    PurePosixPath(component.deployment.entrypoint).stem,
                )
            )
        )
        hits = retriever.search(
            RetrievalQuery(
                terms=terms,
                source_kind=SourcePurpose.REPOSITORY.value,
                seed_paths=(component.deployment.entrypoint,),
            ),
            limit=2,
        )
        for hit in hits:
            if PurePosixPath(hit.document.path).name.casefold() not in {
                "install.sh",
                "deploy.sh",
                "setup.sh",
            }:
                continue
            self._add_excerpt(
                source_id=hit.document.source_id,
                purpose=purposes[hit.document.source_id],
                path=hit.document.path,
                lines=_read_source_lines(sources[hit.document.source_id], hit.document.path),
                line=hit.line,
                kind=EvidenceKind.INSTALLATION_ROUTE,
                strength=EvidenceStrength.STRONG,
                summary=f"Candidate installation route for {component.name}",
                query_id=f"component-{component.id}",
            )

    def add_unreferenced_installers(
        self,
        *,
        sources: dict[str, AcquiredSource],
        purposes: dict[str, SourcePurpose],
        searchable_paths: dict[str, list[str]],
        referenced: set[tuple[str, str]],
        max_results: int,
    ) -> None:
        added = 0
        for source_id, paths in sorted(searchable_paths.items()):
            if purposes[source_id] != SourcePurpose.REPOSITORY:
                continue
            for path in paths:
                if PurePosixPath(path).name.casefold() not in {
                    "install.sh",
                    "deploy.sh",
                    "setup.sh",
                }:
                    continue
                if (source_id, path) in referenced:
                    continue
                lines = _read_source_lines(sources[source_id], path)
                self._add_excerpt(
                    source_id=source_id,
                    purpose=purposes[source_id],
                    path=path,
                    lines=lines or [path],
                    line=1,
                    kind=EvidenceKind.UNREFERENCED_INSTALLER,
                    strength=EvidenceStrength.SUPPORTING,
                    summary="Installer not mapped to a known component",
                )
                added += 1
                if self.full or added >= max_results:
                    return

    def scan_dynamic_deployments(
        self,
        *,
        sources: dict[str, AcquiredSource],
        purposes: dict[str, SourcePurpose],
        references: list[AnalysisSourceRef],
        searchable_paths: dict[str, list[str]],
    ) -> None:
        execution_relevant = {(reference.repo_id, reference.path) for reference in references}
        seen: set[tuple[str, str]] = set()
        candidates = [
            *references,
            *(
                AnalysisSourceRef(repo_id=source_id, path=path)
                for source_id, paths in sorted(searchable_paths.items())
                if purposes[source_id] == SourcePurpose.REPOSITORY
                for path in paths
                if PurePosixPath(path).suffix.casefold() in {".sh", ".bash", ".zsh"}
            ),
        ]
        for reference in candidates:
            key = (reference.repo_id, reference.path)
            if key in seen:
                continue
            seen.add(key)
            lines = _read_source_lines(sources[reference.repo_id], reference.path)
            text = "\n".join(lines)
            for label, pattern, intrinsically_blocking in _DYNAMIC_PATTERNS:
                match = pattern.search(text)
                if match is None:
                    continue
                line = text.count("\n", 0, match.start()) + 1
                self._add_excerpt(
                    source_id=reference.repo_id,
                    purpose=purposes[reference.repo_id],
                    path=reference.path,
                    lines=lines,
                    line=line,
                    kind=EvidenceKind.DYNAMIC_DEPLOYMENT,
                    strength=EvidenceStrength.AUTHORITATIVE,
                    summary=(
                        f"Unresolved dynamic deployment operation: {label}"
                        if intrinsically_blocking and key in execution_relevant
                        else f"Reviewed non-blocking dynamic behavior: {label}"
                    ),
                    dynamic=True,
                    requires_resolution=intrinsically_blocking,
                    blocks_execution=intrinsically_blocking and key in execution_relevant,
                )

    @property
    def full(self) -> bool:
        return (
            len(self.observations) >= self.budget.max_observations
            or self._characters >= self.budget.max_characters
        )

    def _add_excerpt(
        self,
        *,
        source_id: str,
        purpose: SourcePurpose,
        path: str,
        lines: list[str],
        line: int,
        kind: EvidenceKind,
        strength: EvidenceStrength,
        summary: str,
        query_id: str | None = None,
        dynamic: bool = False,
        requires_resolution: bool = False,
        blocks_execution: bool = False,
    ) -> None:
        if self.full:
            return
        line = min(max(1, line), len(lines))
        start = max(1, line - self.budget.lines_before)
        end = min(len(lines), line + self.budget.lines_after)
        key = (source_id, path, start, kind)
        if key in self._keys:
            return
        raw = "\n".join(lines[start - 1 : end])
        sanitized, injection = _sanitize_untrusted(raw)
        remaining = self.budget.max_characters - self._characters
        if remaining <= 0:
            return
        sanitized = sanitized[: min(remaining, self.budget.max_excerpt_characters)]
        digest = hashlib.sha256(raw.encode(errors="replace")).hexdigest()
        identity = f"{source_id}:{path}:{start}:{end}:{kind}"
        evidence_id = f"evidence-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
        self.observations.append(
            EvidenceObservation(
                id=evidence_id,
                source_id=source_id,
                source_purpose=purpose,
                path=path,
                start_line=start,
                end_line=end,
                kind=kind,
                strength=strength,
                summary=sanitize_metadata(summary),
                excerpt=sanitized or None,
                excerpt_hash=digest,
                query_id=query_id,
                prompt_injection_detected=injection,
                dynamic_deployment=dynamic,
                requires_resolution=requires_resolution,
                blocks_execution=blocks_execution,
            )
        )
        self._keys.add(key)
        self._characters += len(sanitized)


def schedule_investigation_queries(
    plan: InvestigationPlan,
    *,
    guided_queries: tuple[InvestigationQuery, ...] = (),
) -> tuple[InvestigationQuery, ...]:
    """Run independent probes first, then spend the remaining fixed budget on guidance."""
    maximum_queries = 8
    mandatory = default_investigation_plan().queries
    candidates = [*mandatory, *guided_queries[:2], *plan.queries]
    result: list[InvestigationQuery] = []
    seen: set[tuple[InvestigationPurpose, tuple[str, ...], SourceScope, str | None, str | None]] = (
        set()
    )
    for query in candidates:
        signature = (
            query.purpose,
            tuple(term.casefold() for term in query.terms),
            query.source_scope,
            query.path_prefix,
            query.component_id,
        )
        if signature not in seen:
            result.append(query)
            seen.add(signature)
        if len(result) == maximum_queries:
            break
    return tuple(result)


def _build_retriever(
    *,
    sources: dict[str, AcquiredSource],
    purposes: dict[str, SourcePurpose],
    searchable_paths: dict[str, list[str]],
) -> DeploymentRetrievalEngine:
    documents = []
    for source_id, paths in sorted(searchable_paths.items()):
        for path in paths:
            lines = _read_source_lines(sources[source_id], path)
            if not lines:
                continue
            documents.append(
                DeploymentRetrievalEngine.document(
                    source_id=source_id,
                    source_kind=purposes[source_id].value,
                    path=path,
                    text="\n".join(lines),
                )
            )
    return DeploymentRetrievalEngine(documents)


def _read_source_lines(source: AcquiredSource, relative_path: str) -> list[str]:
    try:
        candidate = source.resolve_path(relative_path)
        if not candidate.is_file() or candidate.stat().st_size > 2 * 1024 * 1024:
            return []
        raw = candidate.read_bytes()
        if _looks_binary(candidate, raw):
            return []
        return raw.decode("utf-8", errors="replace").splitlines()
    except OSError, ValueError:
        return []


_BINARY_SUFFIXES = {
    ".7z",
    ".avi",
    ".bin",
    ".bmp",
    ".class",
    ".doc",
    ".docx",
    ".eot",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".pyc",
    ".so",
    ".tar",
    ".ttf",
    ".webp",
    ".woff",
    ".woff2",
    ".xls",
    ".xlsx",
    ".zip",
}


def _looks_binary(path: Path, raw: bytes) -> bool:
    if path.suffix.casefold() in _BINARY_SUFFIXES or b"\x00" in raw[:8_192]:
        return True
    sample = raw[:8_192]
    if not sample:
        return False
    control_bytes = sum(byte < 9 or 13 < byte < 32 for byte in sample)
    return control_bytes / len(sample) > 0.02


def _sanitize_untrusted(text: str) -> tuple[str, bool]:
    injection = bool(_PROMPT_INJECTION.search(text))
    safe_lines = [
        "[POTENTIAL PROMPT INJECTION REMOVED]" if _PROMPT_INJECTION.search(line) else line
        for line in text.splitlines()
    ]
    return redact_secrets("\n".join(safe_lines)), injection


def sanitize_metadata(value: str) -> str:
    """Remove prompt-injection phrases and secrets from prompt metadata."""
    return _sanitize_untrusted(value)[0]
