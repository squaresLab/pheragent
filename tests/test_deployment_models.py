from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from pheragent.deployment.enums import (
    BlockType,
    Confidence,
    Executor,
    OperationPhase,
    ProvenanceOrigin,
    SourceKind,
)
from pheragent.deployment.models import (
    ArtifactMetadata,
    DeploymentArtifact,
    DeploymentBlock,
    DeploymentScope,
    Operation,
    Provenance,
    Requirement,
    SourceRecord,
    SourcesConfig,
    SourceSpec,
)
from pheragent.deployment.serialization import load_deployment_artifact, write_yaml
from pheragent.deployment.validation import validate_artifact

_HASH = "a" * 64


def _provenance() -> Provenance:
    return Provenance(origin=ProvenanceOrigin.INFERRED, confidence=Confidence.MEDIUM)


def _block(block_id: str, *, provider: str | None = None) -> DeploymentBlock:
    requirements = []
    if provider:
        requirements.append(
            Requirement(
                capability=f"capability-from-{provider}",
                provider_block=provider,
                provenance=_provenance(),
            )
        )
    return DeploymentBlock(
        id=block_id,
        name=block_id.replace("-", " ").title(),
        type=BlockType.PLATFORM,
        purpose="Provide a test deployment capability.",
        grouping_rationale="The fixture components share a lifecycle.",
        grouping_provenance=_provenance(),
        requires=requirements,
        provenance=_provenance(),
    )


def _artifact(*blocks: DeploymentBlock) -> DeploymentArtifact:
    return DeploymentArtifact(
        artifact_version="0.1",
        metadata=ArtifactMetadata(
            artifact_id="fixture-artifact",
            system_name="Fixture",
            generated_at=datetime(2026, 8, 4, tzinfo=UTC),
            generator_version="0.1.0",
        ),
        sources=[
            SourceRecord(
                id="fixture-source",
                kind=SourceKind.LOCAL_DIRECTORY,
                repository="fixtures/deployment",
                content_hash=_HASH,
            )
        ],
        scope=DeploymentScope(),
        blocks=list(blocks),
    )


def test_sources_config_rejects_duplicate_ids() -> None:
    source = SourceSpec(id="docs", kind=SourceKind.LOCAL_DIRECTORY, location="docs")

    with pytest.raises(ValidationError, match="duplicate source id"):
        SourcesConfig(system="demo", sources=[source, source])


def test_extracted_provenance_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="requires at least one evidence"):
        Provenance(origin=ProvenanceOrigin.EXTRACTED, confidence=Confidence.HIGH)


def test_operation_command_requires_source_reference() -> None:
    with pytest.raises(ValidationError, match="require a source_artifact"):
        Operation(
            id="install-platform",
            phase=OperationPhase.INSTALL,
            executor=Executor.SHELL,
            command="./install.sh",
            provenance=Provenance(
                origin=ProvenanceOrigin.EXTRACTED,
                confidence=Confidence.HIGH,
                evidence_refs=["evidence-1"],
            ),
        )


def test_artifact_yaml_round_trip_is_stable(tmp_path: Path) -> None:
    artifact = _artifact(_block("platform"))
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"

    write_yaml(first, artifact)
    loaded = load_deployment_artifact(first)
    write_yaml(second, loaded)

    assert loaded == artifact
    assert second.read_text(encoding="utf-8") == first.read_text(encoding="utf-8")


def test_artifact_validator_rejects_hard_cycle() -> None:
    artifact = _artifact(
        _block("platform", provider="application"),
        _block("application", provider="platform"),
    )

    result = validate_artifact(artifact)

    assert result.valid is False
    assert any(finding.code == "hard_dependency_cycle" for finding in result.findings)
