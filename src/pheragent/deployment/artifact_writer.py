from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .enums import DeterministicFindingKind
from .fact_extractor import FactExtractionResult
from .inspection import DeterministicInspectionResult
from .models import (
    ArtifactValidationResult,
    DeploymentArtifact,
    UnresolvedQuestionsDocument,
)
from .report import render_inspection_report
from .serialization import write_json, write_text, write_yaml
from .synthesis import SynthesisResult
from .validation import validate_artifact


@dataclass(frozen=True, slots=True)
class ArtifactWriteResult:
    artifact: DeploymentArtifact
    validation: ArtifactValidationResult
    report_path: Path


def write_artifact_outputs(
    *,
    output_dir: Path,
    inspection: DeterministicInspectionResult,
    extraction: FactExtractionResult,
    synthesis: SynthesisResult,
    strict: bool,
) -> ArtifactWriteResult:
    validation = validate_artifact(
        synthesis.artifact,
        evidence=inspection.evidence,
        strict=strict,
    )
    write_yaml(output_dir / "deployment-artifact.yaml", synthesis.artifact)
    write_json(output_dir / "dependency-graph.json", synthesis.graph)
    write_yaml(
        output_dir / "unresolved-questions.yaml",
        UnresolvedQuestionsDocument(questions=synthesis.artifact.unresolved_questions),
    )
    unassigned_commands = _unassigned_commands(inspection, synthesis)
    report_path = output_dir / "inspection-report.md"
    write_text(
        report_path,
        render_inspection_report(
            artifact=synthesis.artifact,
            inventory=inspection.inventory,
            graph=synthesis.graph,
            validation=validation,
            extraction_warnings=extraction.report.warnings,
            unassigned_commands=unassigned_commands,
        ),
    )
    return ArtifactWriteResult(
        artifact=synthesis.artifact,
        validation=validation,
        report_path=report_path,
    )


def _unassigned_commands(
    inspection: DeterministicInspectionResult, synthesis: SynthesisResult
) -> list[str]:
    assigned_evidence = {
        ref
        for block in synthesis.artifact.blocks
        for operation in block.operations
        for ref in operation.provenance.evidence_refs
    }
    return sorted(
        f"{finding.source_id}:{finding.path}:{finding.name}"
        for finding in inspection.findings
        if finding.kind == DeterministicFindingKind.COMMAND
        and not set(finding.evidence_refs).issubset(assigned_evidence)
    )
