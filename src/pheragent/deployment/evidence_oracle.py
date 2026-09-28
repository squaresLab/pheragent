from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from pydantic import Field

from pheragent.deployment.analysis_models import AnalysisSourceRef
from pheragent.deployment.models import ContractModel, InventoryEntry
from pheragent.deployment.redaction import redact_secrets
from pheragent.deployment.retrieval import DeploymentRetrievalEngine, RetrievalQuery
from pheragent.deployment.source_manager import AcquiredSource, AcquisitionResult

_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_RST_LINK = re.compile(r"`[^`<>]*<([^>]+)>`_+")


class Evidence(ContractModel):
    """A source passage returned by the oracle without interpretation."""

    id: str
    repo_id: str
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    kind: str
    content: str
    score: float = Field(ge=0)


@dataclass(slots=True)
class EvidenceOracle:
    """Search redacted source passages and report provenance facts."""

    acquisition: AcquisitionResult
    retriever: DeploymentRetrievalEngine
    evidence: dict[str, Evidence]

    @classmethod
    def build(
        cls,
        acquisition: AcquisitionResult,
        entries: list[InventoryEntry],
    ) -> EvidenceOracle:
        sources = {source.id: source for source in acquisition.sources}
        texts = {
            (entry.source_id, entry.path): redact_secrets(
                sources[entry.source_id]
                .resolve_path(entry.path)
                .read_text(encoding="utf-8", errors="replace")
            )
            for entry in entries
            if entry.selected
        }
        documents = [
            passage
            for (source_id, path), text in texts.items()
            for passage in DeploymentRetrievalEngine.passages(
                source_id=source_id,
                source_kind=sources[source_id].spec.purpose,
                path=path,
                text=text,
            )
        ]
        relations = tuple(
            (source, target)
            for source, targets in _source_links(texts, sources).items()
            for target in targets
        )
        return cls(
            acquisition=acquisition,
            retriever=DeploymentRetrievalEngine(documents, relations=relations),
            evidence={
                document.key: Evidence(
                    id=document.key,
                    repo_id=document.source_id,
                    path=document.path,
                    start_line=document.start_line,
                    end_line=document.end_line,
                    kind=document.kind,
                    content=document.text,
                    score=0,
                )
                for document in documents
            },
        )

    def search(
        self,
        query: str,
        *,
        path_prefix: str | None = None,
        limit: int = 4,
        exclude: frozenset[str] = frozenset(),
    ) -> tuple[Evidence, ...]:
        request = RetrievalQuery(terms=(query,), path_prefix=path_prefix)
        hits = self.retriever.search(request, limit=limit + len(exclude))
        if path_prefix and len(hits) < limit + len(exclude):
            broad = self.retriever.search(
                RetrievalQuery(terms=request.terms),
                limit=limit + len(exclude),
            )
            known = {hit.document.key for hit in hits}
            hits.extend(hit for hit in broad if hit.document.key not in known)
        return tuple(
            self.evidence[hit.document.key].model_copy(update={"score": hit.score})
            for hit in hits
            if hit.document.key not in exclude
        )[:limit]

    def cited_text_contains(self, value: str, evidence_ids: list[str]) -> bool:
        normalized = _normalize_command(value)
        return any(
            normalized in _normalize_command(item.content)
            for evidence_id in evidence_ids
            if (item := self.evidence.get(evidence_id)) is not None
        )

    def references(self, evidence_ids: list[str]) -> list[AnalysisSourceRef]:
        return [
            AnalysisSourceRef(
                repo_id=item.repo_id,
                path=item.path,
                start_line=item.start_line,
                end_line=item.end_line,
            )
            for evidence_id in evidence_ids
            if (item := self.evidence.get(evidence_id)) is not None
        ]

    def command_evidence(self, command: str, evidence_ids: list[str]) -> Evidence | None:
        normalized = _normalize_command(command)
        return next(
            (
                item
                for evidence_id in evidence_ids
                if (item := self.evidence.get(evidence_id)) is not None
                and normalized in _normalize_command(item.content)
            ),
            None,
        )

    def directory_exists(self, repo_id: str, path: str) -> bool:
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts:
            return False
        source = next((item for item in self.acquisition.sources if item.id == repo_id), None)
        if source is None:
            return False
        try:
            return source.resolve_path(str(pure / ".keep")).parent.is_dir()
        except (OSError, ValueError):
            return False


def _source_links(
    texts: dict[tuple[str, str], str],
    sources: dict[str, AcquiredSource],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for (source_id, path), text in texts.items():
        targets: list[str] = []
        for raw in _local_link_targets(text):
            target = raw.split("#", 1)[0].split("?", 1)[0].strip(" <>\"")
            if not target or "://" in target or target.startswith(("mailto:", "/")):
                continue
            resolved = posixpath.normpath(str(PurePosixPath(path).parent / target))
            for candidate in (resolved, f"{resolved}.rst", f"{resolved}/README.md"):
                if sources[source_id].contains_file(candidate):
                    targets.append(f"{source_id}:{candidate}")
                    break
        result[f"{source_id}:{path}"] = tuple(dict.fromkeys(targets))
    return result


def _local_link_targets(text: str) -> list[str]:
    targets = [*_MARKDOWN_LINK.findall(text), *_RST_LINK.findall(text)]
    in_toctree = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(".. toctree::"):
            in_toctree = True
        elif in_toctree and line and not line[0].isspace():
            in_toctree = False
        elif in_toctree and stripped and not stripped.startswith(":"):
            targets.append(stripped.rsplit("<", 1)[-1].rstrip(">"))
    return targets


def _normalize_command(value: str) -> str:
    return " ".join(value.replace("\\\n", " ").split())
