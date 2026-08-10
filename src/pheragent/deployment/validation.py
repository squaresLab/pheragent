from __future__ import annotations

from .enums import BlockType, Confidence, FindingSeverity, SourceKind
from .models import (
    ArtifactValidationResult,
    DeploymentArtifact,
    EvidenceRecord,
    ValidationFinding,
)


def validate_artifact(
    artifact: DeploymentArtifact,
    *,
    evidence: tuple[EvidenceRecord, ...] | None = None,
    strict: bool = False,
) -> ArtifactValidationResult:
    findings: list[ValidationFinding] = []
    evidence_ids = {record.id for record in evidence} if evidence is not None else None

    if strict:
        for source in artifact.sources:
            if source.kind == SourceKind.GIT and not source.revision:
                findings.append(
                    _error(
                        "unpinned_git_source",
                        f"Git source has no resolved revision: {source.id}",
                        source.id,
                    )
                )

    for block in artifact.blocks:
        if not block.provides:
            findings.append(
                _warning("block_without_capability", "Block provides no capability.", block.id)
            )
        if block.type != BlockType.VERIFICATION and not block.validations:
            findings.append(
                _warning("block_without_validation", "Block has no validation.", block.id)
            )
        if len(block.components) == 1 and not block.component_relations:
            findings.append(
                _warning(
                    "possibly_fine_grained_block",
                    "Block contains one component and may be too fine-grained.",
                    block.id,
                )
            )
        if not block.grouping_provenance.evidence_refs:
            findings.append(
                _warning(
                    "grouping_without_evidence",
                    "Block grouping has no supporting evidence.",
                    block.id,
                )
            )
        if (
            block.provenance.origin.value == "inferred"
            and block.provenance.confidence == Confidence.LOW
        ):
            findings.append(
                _warning(
                    "low_confidence_inference",
                    "Block is based on a low-confidence inferred claim.",
                    block.id,
                )
            )
        for requirement in block.requires:
            if requirement.mandatory and requirement.provider_block is None:
                findings.append(
                    _warning(
                        "unmatched_capability",
                        f"Required capability has no provider: {requirement.capability}",
                        block.id,
                    )
                )
        if evidence_ids is not None:
            for item in (*block.operations, *block.validations):
                unknown = sorted(set(item.provenance.evidence_refs) - evidence_ids)
                if unknown:
                    findings.append(
                        _error(
                            "unknown_evidence_reference",
                            f"{item.id} references unknown evidence: {', '.join(unknown)}",
                            block.id,
                        )
                    )

    if len(artifact.blocks) > 10:
        findings.append(
            _warning(
                "too_many_blocks",
                f"Artifact contains {len(artifact.blocks)} global blocks; expected at most 10.",
            )
        )

    cycle = _find_hard_cycle(artifact)
    if cycle:
        findings.append(
            ValidationFinding(
                severity=FindingSeverity.ERROR,
                code="hard_dependency_cycle",
                message=f"Hard dependency cycle detected: {' -> '.join(cycle)}",
            )
        )

    valid = not any(finding.severity == FindingSeverity.ERROR for finding in findings)
    return ArtifactValidationResult(valid=valid, findings=findings)


def _warning(code: str, message: str, location: str | None = None) -> ValidationFinding:
    return ValidationFinding(
        severity=FindingSeverity.WARNING,
        code=code,
        message=message,
        location=location,
    )


def _error(code: str, message: str, location: str | None = None) -> ValidationFinding:
    return ValidationFinding(
        severity=FindingSeverity.ERROR,
        code=code,
        message=message,
        location=location,
    )


def _find_hard_cycle(artifact: DeploymentArtifact) -> list[str] | None:
    graph = {
        block.id: {
            requirement.provider_block
            for requirement in block.requires
            if requirement.mandatory and requirement.provider_block is not None
        }
        for block in artifact.blocks
    }
    visited: set[str] = set()
    active: list[str] = []
    active_set: set[str] = set()

    def visit(block_id: str) -> list[str] | None:
        if block_id in active_set:
            start = active.index(block_id)
            return [*active[start:], block_id]
        if block_id in visited:
            return None
        active.append(block_id)
        active_set.add(block_id)
        for provider_id in sorted(graph[block_id]):
            cycle = visit(provider_id)
            if cycle:
                return cycle
        active.pop()
        active_set.remove(block_id)
        visited.add(block_id)
        return None

    for block_id in sorted(graph):
        cycle = visit(block_id)
        if cycle:
            return cycle
    return None
