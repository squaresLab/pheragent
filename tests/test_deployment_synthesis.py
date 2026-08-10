from __future__ import annotations

import json
from pathlib import Path

import yaml

from pheragent.deployment.enums import BlockType, Confidence, ProvenanceOrigin
from pheragent.deployment.fact_extractor import FactExtractorConfig, run_fact_extraction
from pheragent.deployment.graph import build_dependency_graph
from pheragent.deployment.inspection import run_deterministic_inspection
from pheragent.deployment.models import (
    Capability,
    DeploymentBlock,
    Provenance,
    Requirement,
)
from pheragent.deployment.output import AtomicOutputTransaction
from pheragent.deployment.serialization import write_text
from pheragent.deployment.synthesis import synthesize_artifact
from pheragent.deployment.validation import validate_artifact


def _provenance() -> Provenance:
    return Provenance(origin=ProvenanceOrigin.INFERRED, confidence=Confidence.MEDIUM)


def _block(
    block_id: str,
    *,
    provides: tuple[str, ...] = (),
    requires: tuple[tuple[str, str | None], ...] = (),
) -> DeploymentBlock:
    return DeploymentBlock(
        id=block_id,
        name=block_id.title(),
        type=BlockType.APPLICATION,
        purpose="Fixture block.",
        grouping_rationale="Fixture grouping.",
        grouping_provenance=_provenance(),
        provides=[Capability(capability=item, provenance=_provenance()) for item in provides],
        requires=[
            Requirement(capability=item, provider_block=provider, provenance=_provenance())
            for item, provider in requires
        ],
        provenance=_provenance(),
    )


def test_dependency_graph_resolves_capabilities_and_detects_cycles() -> None:
    platform = _block(
        "platform",
        provides=("cluster",),
        requires=(("application-api", "application"),),
    )
    application = _block(
        "application",
        provides=("application-api",),
        requires=(("cluster", None), ("missing", None)),
    )

    graph = build_dependency_graph((platform, application))

    assert [
        (edge.provider_block, edge.consumer_block, edge.capability) for edge in graph.edges
    ] == [
        ("application", "platform", "application-api"),
        ("platform", "application", "cluster"),
    ]
    assert graph.unmatched_capabilities == ["application:missing"]
    assert graph.cycle == ["application", "platform", "application"]


def test_synthesis_groups_components_and_keeps_commands_evidence_backed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "compose.yaml").write_text(
        "services:\n"
        "  database:\n"
        "    image: postgres:16\n"
        "  api:\n"
        "    image: fixture/api\n"
        "    depends_on: [database]\n"
        "    command: python api.py\n",
        encoding="utf-8",
    )
    (source / "verify.sh").write_text("kubectl get pods\n", encoding="utf-8")
    sources = tmp_path / "sources.yaml"
    sources.write_text(
        yaml.safe_dump(
            {
                "system": "fixture",
                "sources": [
                    {
                        "id": "fixture",
                        "kind": "local_directory",
                        "location": "source",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    staged = tmp_path / "staged"
    inspection = run_deterministic_inspection(
        sources_path=sources,
        output_dir=staged,
        strict=True,
    )
    extraction = run_fact_extraction(
        inspection,
        output_dir=staged,
        config=FactExtractorConfig(extractor="deterministic"),
    )

    result = synthesize_artifact(inspection, extraction)
    repeated = synthesize_artifact(inspection, extraction)

    by_type = {block.type: block for block in result.artifact.blocks}
    assert set(by_type) == {
        BlockType.SHARED_SERVICES,
        BlockType.APPLICATION,
        BlockType.VERIFICATION,
    }
    assert [item.name for item in by_type[BlockType.SHARED_SERVICES].components] == ["database"]
    assert [item.name for item in by_type[BlockType.APPLICATION].components] == ["api"]
    application_operations = by_type[BlockType.APPLICATION].operations
    assert [item.command for item in application_operations] == ["python api.py"]
    assert all(item.provenance.evidence_refs for item in application_operations)
    assert result.graph.cycle == []
    assert repeated.artifact.blocks == result.artifact.blocks
    assert repeated.graph == result.graph
    assert validate_artifact(result.artifact, evidence=inspection.evidence, strict=True).valid


def test_atomic_output_transaction_does_not_publish_failed_stage(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    published = output / "deployment-artifact.yaml"
    published.write_text("old\n", encoding="utf-8")

    try:
        with AtomicOutputTransaction(output) as transaction:
            write_text(transaction.staging_dir / published.name, "new\n")
            raise RuntimeError("fixture failure")
    except RuntimeError:
        pass

    assert published.read_text(encoding="utf-8") == "old\n"
    assert not list(output.glob(".staging-*"))

    with AtomicOutputTransaction(output) as transaction:
        write_text(transaction.staging_dir / published.name, "new\n")
        transaction.commit()

    assert published.read_text(encoding="utf-8") == "new\n"
    manifest = json.loads((output / "output-manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"] == {
        "deployment-artifact.yaml": (
            "7aa7a5359173d05b63cfd682e3c38487f3cb4f7f1d60659fe59fab1505977d4c"
        )
    }
