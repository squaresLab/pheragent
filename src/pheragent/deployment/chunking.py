from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from .models import (
    DeterministicFinding,
    EvidenceRecord,
    ExtractionChunk,
    RepositoryInventory,
)


def build_extraction_chunks(
    findings: tuple[DeterministicFinding, ...],
    evidence: tuple[EvidenceRecord, ...],
    *,
    max_chars: int = 12_000,
) -> tuple[ExtractionChunk, ...]:
    if max_chars < 1_000:
        raise ValueError("extraction chunk size must be at least 1000 characters")
    evidence_by_id = {record.id: record for record in evidence}
    grouped: dict[tuple[str, str], list[DeterministicFinding]] = defaultdict(list)
    for finding in findings:
        grouped[(finding.source_id, finding.path)].append(finding)

    chunks: list[ExtractionChunk] = []
    for (source_id, path), path_findings in sorted(grouped.items()):
        batch: list[DeterministicFinding] = []
        batch_size = 0
        for finding in path_findings:
            referenced_evidence = [
                evidence_by_id[ref].model_dump(mode="json")
                for ref in finding.evidence_refs
                if ref in evidence_by_id
            ]
            rendered = json.dumps(
                {"finding": finding.model_dump(mode="json"), "evidence": referenced_evidence},
                sort_keys=True,
                separators=(",", ":"),
            )
            if batch and batch_size + len(rendered) > max_chars:
                chunks.append(_make_chunk(source_id, path, batch, evidence_by_id))
                batch = []
                batch_size = 0
            batch.append(finding)
            batch_size += len(rendered)
        if batch:
            chunks.append(_make_chunk(source_id, path, batch, evidence_by_id))
    return tuple(chunks)


def rank_extraction_chunks(
    chunks: tuple[ExtractionChunk, ...],
    inventory: RepositoryInventory,
) -> tuple[ExtractionChunk, ...]:
    relevance = {
        (entry.source_id, entry.path): entry.relevance_score for entry in inventory.entries
    }
    return tuple(
        sorted(
            chunks,
            key=lambda chunk: (
                -relevance.get((chunk.source_id, chunk.path), 0.0),
                chunk.source_id,
                chunk.path,
                chunk.id,
            ),
        )
    )


def _make_chunk(
    source_id: str,
    path: str,
    findings: list[DeterministicFinding],
    evidence_by_id: dict[str, EvidenceRecord],
) -> ExtractionChunk:
    evidence_refs = sorted({ref for finding in findings for ref in finding.evidence_refs})
    missing = [ref for ref in evidence_refs if ref not in evidence_by_id]
    if missing:
        raise ValueError(f"findings reference missing evidence: {', '.join(missing)}")
    identity = json.dumps(
        {
            "source_id": source_id,
            "path": path,
            "finding_ids": [finding.id for finding in findings],
            "evidence_refs": evidence_refs,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return ExtractionChunk(
        id=f"chunk-{hashlib.sha256(identity.encode()).hexdigest()[:20]}",
        source_id=source_id,
        path=path,
        finding_ids=[finding.id for finding in findings],
        evidence=[evidence_by_id[ref] for ref in evidence_refs],
        findings=findings,
    )
