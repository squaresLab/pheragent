from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..enums import DeterministicFindingKind
from ..evidence import EvidenceStore
from ..models import DeterministicFinding, InventoryEntry
from ..redaction import redact_secrets
from ..source_manager import AcquiredSource


@dataclass(slots=True)
class ParseResult:
    parser: str
    findings: list[DeterministicFinding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DeterministicParser(Protocol):
    name: str

    def parse(
        self,
        source: AcquiredSource,
        entry: InventoryEntry,
        evidence: EvidenceStore,
    ) -> ParseResult: ...


def make_finding(
    *,
    source: AcquiredSource,
    entry: InventoryEntry,
    evidence: EvidenceStore,
    kind: DeterministicFindingKind,
    name: str,
    start_line: int,
    end_line: int,
    attributes: dict[str, Any] | None = None,
    heading: str | None = None,
) -> DeterministicFinding:
    record = evidence.capture(
        source,
        path=entry.path,
        start_line=start_line,
        end_line=end_line,
        heading=heading,
    )
    safe_name = redact_secrets(name)
    safe_attributes = _redact_value(attributes or {})
    identity_payload = {
        "source_id": source.id,
        "path": entry.path,
        "category": entry.category,
        "kind": kind,
        "name": safe_name,
        "attributes": safe_attributes,
        "evidence_refs": [record.id],
    }
    canonical = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"), default=str)
    finding_id = f"finding-{hashlib.sha256(canonical.encode()).hexdigest()[:20]}"
    return DeterministicFinding(
        id=finding_id,
        source_id=source.id,
        path=entry.path,
        category=entry.category,
        kind=kind,
        name=safe_name,
        attributes=safe_attributes,
        evidence_refs=[record.id],
    )


def source_file_path(source: AcquiredSource, entry: InventoryEntry):
    if source.path.is_file():
        return source.path
    return source.path / entry.path


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {str(key): _redact_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_redact_value(item) for item in value]
    return value
