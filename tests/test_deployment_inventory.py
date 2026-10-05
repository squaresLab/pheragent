from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pheragent.deployment.enums import InventoryCategory, SourceKind
from pheragent.deployment.inventory import RepositoryInventoryBuilder, classify_file
from pheragent.deployment.models import SourceManifestEntry, SourceSpec
from pheragent.deployment.source_manager import AcquiredSource
from pheragent.deployment.sources import SourceTools, _source_grounded


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
    assert classify_file("Makefile", "start:\n\tdocker compose up -d\n") == InventoryCategory.BUILD
    assert (
        classify_file(".github/workflows/deploy.yml", "jobs:\n  deploy: {}\n")
        == InventoryCategory.CI_WORKFLOW
    )
    assert classify_file("compose.yaml", "services:\n  api: {}\n") == InventoryCategory.COMPOSE
    assert (
        classify_file("compose.full.yaml", "services:\n  worker: {}\n") == InventoryCategory.COMPOSE
    )
    assert (
        classify_file("docker-compose.override.yml", "services:\n  worker: {}\n")
        == InventoryCategory.COMPOSE
    )
    assert (
        classify_file("kustomization.yaml", "resources:\n  - app.yaml\n")
        == InventoryCategory.KUSTOMIZE
    )
    assert (
        classify_file("main.tf", 'resource "aws_instance" "node" {}\n')
        == InventoryCategory.TERRAFORM
    )


def test_source_tools_search_build_file_and_read_unfamiliar_safe_text(tmp_path: Path) -> None:
    (tmp_path / "Makefile").write_text("start:\n\tdocker compose up -d\n")
    (tmp_path / "instructions.custom").write_text("Run make start.\n")
    (tmp_path / "binary.dat").write_bytes(b"\0not text")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/excluded.png").write_text("Run make start.\n")
    tools = SourceTools((_source(tmp_path),))

    hits = tools.call(SimpleNamespace(tool="search_sources", query="make start docker compose"))
    assert any(hit["source"] == "fixture:Makefile" for hit in hits["hits"])

    def read(name: str) -> dict:
        return tools.call(
            SimpleNamespace(
                tool="read_file", source_path=f"fixture:{name}", start_line=None, end_line=None
            )
        )

    assert read("instructions.custom")["text"] == "Run make start."
    assert _source_grounded(
        SimpleNamespace(
            command=["make", "start"],
            evidence=["fixture:instructions.custom"],
            working_directory=None,
        ),
        tools,
    )
    for name in ("binary.dat", "assets/excluded.png"):
        with pytest.raises(ValueError, match="readable inventory"):
            read(name)


def test_source_inventory_shows_root_entrypoint_before_deep_files(tmp_path: Path) -> None:
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    for index in range(25):
        (workflows / f"{index:02}.yaml").write_text("jobs: {}\n")
    (tmp_path / "Makefile").write_text("start:\n\tdocker compose up -d\n")

    result = SourceTools((_source(tmp_path),)).call(SimpleNamespace(tool="inventory_sources"))

    assert result["total"] == 26
    assert result["tree"].startswith("fixture:\n  Makefile\n")
    assert ".github/" in result["tree"]


def test_source_search_returns_distinct_files_and_matching_lines(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "\n".join(f"# Section {index}\ndocker compose deploy application" for index in range(9))
    )
    (tmp_path / "Makefile").write_text("start:\n\tdocker compose up -d\n")
    result = SourceTools((_source(tmp_path),)).call(
        SimpleNamespace(tool="search_sources", query="docker compose deploy")
    )

    assert [hit["source"] for hit in result["hits"]] == [
        "fixture:README.md",
        "fixture:Makefile",
    ]
    assert all(hit["start_line"] <= hit["line"] <= hit["end_line"] for hit in result["hits"])


def test_inventory_includes_sample_and_example_files(tmp_path: Path) -> None:
    (tmp_path / "global_configmap.yaml.sample").write_text(
        "apiVersion: v1\nkind: ConfigMap\n", encoding="utf-8"
    )
    (tmp_path / "settings.env.example").write_text("HOST=example\n", encoding="utf-8")

    inventory = RepositoryInventoryBuilder().build((_source(tmp_path),))
    entries = {entry.path: entry for entry in inventory.entries}

    assert entries["global_configmap.yaml.sample"].category == InventoryCategory.KUBERNETES
    assert entries["settings.env.example"].category == InventoryCategory.CONFIGURATION
