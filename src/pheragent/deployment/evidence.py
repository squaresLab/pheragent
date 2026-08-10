from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

from .models import EvidenceRecord
from .redaction import redact_secrets
from .serialization import write_jsonl
from .source_manager import AcquiredSource


class EvidenceStore:
    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}
        self._line_cache: dict[Path, tuple[str, ...]] = {}

    @property
    def records(self) -> list[EvidenceRecord]:
        return sorted(
            self._records.values(),
            key=lambda item: (item.source_id, item.path, item.start_line, item.end_line, item.id),
        )

    def capture(
        self,
        source: AcquiredSource,
        *,
        path: str,
        start_line: int,
        end_line: int,
        heading: str | None = None,
    ) -> EvidenceRecord:
        evidence_path, display_path = _resolve_evidence_path(source, path)
        lines = self._lines(evidence_path, display_path)
        if start_line < 1 or end_line < start_line or end_line > len(lines):
            raise ValueError(
                f"invalid evidence line range {start_line}-{end_line} for "
                f"{display_path} ({len(lines)} lines)"
            )
        excerpt = redact_secrets("\n".join(lines[start_line - 1 : end_line]))
        excerpt_hash = hashlib.sha256(excerpt.encode()).hexdigest()
        identity = "\0".join(
            (source.id, display_path, str(start_line), str(end_line), excerpt_hash)
        )
        evidence_id = f"evidence-{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
        record = EvidenceRecord(
            id=evidence_id,
            source_id=source.id,
            path=display_path,
            start_line=start_line,
            end_line=end_line,
            heading=heading,
            excerpt_hash=excerpt_hash,
            excerpt=excerpt,
        )
        existing = self._records.get(evidence_id)
        if existing is not None and existing != record:
            raise ValueError(f"evidence ID collision: {evidence_id}")
        self._records[evidence_id] = record
        return record

    def _lines(self, path: Path, display_path: str) -> tuple[str, ...]:
        cached = self._line_cache.get(path)
        if cached is not None:
            return cached
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"evidence file is not UTF-8 text: {display_path}") from exc
        lines = tuple(text.splitlines())
        self._line_cache[path] = lines
        return lines

    def write(self, path: Path) -> None:
        write_jsonl(path, self.records)


def _resolve_evidence_path(source: AcquiredSource, path: str) -> tuple[Path, str]:
    normalized = PurePosixPath(path)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"evidence path must stay inside the source: {path}")

    if source.path.is_file():
        if path not in {".", source.path.name}:
            raise ValueError(f"unknown path for local file source {source.id}: {path}")
        return source.path, source.path.name

    candidate = source.path.joinpath(*normalized.parts).resolve(strict=True)
    try:
        candidate.relative_to(source.path.resolve())
    except ValueError as exc:
        raise ValueError(f"evidence path escapes source {source.id}: {path}") from exc
    if not candidate.is_file():
        raise ValueError(f"evidence path is not a file: {path}")
    return candidate, normalized.as_posix()
