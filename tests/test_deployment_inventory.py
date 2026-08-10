from __future__ import annotations

from pathlib import Path

from pheragent.deployment.enums import InventoryCategory, SourceKind
from pheragent.deployment.inventory import RepositoryInventoryBuilder, classify_file
from pheragent.deployment.models import SourceManifestEntry, SourceSpec
from pheragent.deployment.source_manager import AcquiredSource


def _source(path: Path, *, include: list[str] | None = None) -> AcquiredSource:
    spec = SourceSpec(
        id="fixture",
        kind=SourceKind.LOCAL_DIRECTORY,
        location="fixture",
        include_patterns=include or [],
        exclude_patterns=["**/*.png"],
    )
    return AcquiredSource(
        id=spec.id,
        kind=spec.kind,
        path=path,
        spec=spec,
        manifest=SourceManifestEntry(
            id=spec.id,
            kind=spec.kind,
            location=spec.location,
            include_patterns=spec.include_patterns,
            exclude_patterns=spec.exclude_patterns,
            content_hash="a" * 64,
        ),
    )


def test_inventory_applies_patterns_and_classifies_files(tmp_path: Path) -> None:
    source_path = tmp_path / "source"
    deployment = source_path / "deployment"
    deployment.mkdir(parents=True)
    (source_path / "README.md").write_text("# Install\n", encoding="utf-8")
    (deployment / "install.sh").write_text("#!/bin/sh\necho install\n", encoding="utf-8")
    (deployment / "cluster.yaml").write_text(
        "apiVersion: v1\nkind: Service\nmetadata:\n  name: api\n",
        encoding="utf-8",
    )
    (deployment / "diagram.png").write_bytes(b"not really an image")
    (source_path / "unrelated.txt").write_text("ignored\n", encoding="utf-8")

    inventory = RepositoryInventoryBuilder().build(
        (_source(source_path, include=["README*", "deployment/**"]),)
    )
    by_path = {entry.path: entry for entry in inventory.entries}

    assert by_path["README.md"].category == InventoryCategory.DOCUMENTATION
    assert by_path["deployment/install.sh"].category == InventoryCategory.SHELL
    assert by_path["deployment/cluster.yaml"].category == InventoryCategory.KUBERNETES
    assert by_path["deployment/diagram.png"].skip_reason == "excluded_pattern"
    assert by_path["unrelated.txt"].skip_reason == "not_included"
    assert inventory.detected_technologies == ["kubernetes", "shell"]


def test_classification_detects_deployment_yaml_types() -> None:
    assert (
        classify_file(".github/workflows/deploy.yml", "jobs:\n  deploy: {}\n")
        == InventoryCategory.CI_WORKFLOW
    )
    assert classify_file("compose.yaml", "services:\n  api: {}\n") == InventoryCategory.COMPOSE
    assert (
        classify_file("kustomization.yaml", "resources:\n  - app.yaml\n")
        == InventoryCategory.KUSTOMIZE
    )
    assert (
        classify_file("main.tf", 'resource "aws_instance" "node" {}\n')
        == InventoryCategory.TERRAFORM
    )
