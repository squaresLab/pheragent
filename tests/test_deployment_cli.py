from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pheragent.cli import _build_parser, main


@pytest.fixture(autouse=True)
def _isolate_project_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def _artifact_payload() -> dict[str, object]:
    provenance = {"origin": "inferred", "confidence": "medium", "evidence_refs": []}
    return {
        "artifact_version": "0.1",
        "metadata": {
            "artifact_id": "fixture-artifact",
            "system_name": "Fixture",
            "system_version": "unknown",
            "profile": "test",
            "generated_at": "2026-08-04T00:00:00Z",
            "generator_version": "0.1.0",
        },
        "sources": [
            {
                "id": "fixture-source",
                "kind": "local_directory",
                "repository": "fixtures/deployment",
                "root_path": ".",
                "content_hash": "a" * 64,
            }
        ],
        "scope": {"includes": [], "excludes": []},
        "targets": [],
        "blocks": [
            {
                "id": "shared-services",
                "name": "Shared Services",
                "type": "shared_services",
                "purpose": "Provide reusable services.",
                "grouping_rationale": "The services share a target and lifecycle.",
                "grouping_provenance": provenance,
                "components": [],
                "component_relations": [],
                "requires": [],
                "provides": [],
                "operations": [],
                "validations": [],
                "recovery_boundary": {
                    "local_components": [],
                    "upstream_capabilities": [],
                },
                "provenance": provenance,
            }
        ],
        "policies": {
            "commands_are_declarative": True,
            "secret_values_forbidden": True,
        },
        "unresolved_questions": [],
    }


def _write_yaml(path: Path, payload: object) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_deployment_inspect_arguments(tmp_path: Path) -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "deployment",
            "inspect",
            "--sources",
            str(tmp_path / "sources.yaml"),
            "--output",
            str(tmp_path / "output"),
            "--strict",
        ]
    )

    assert args.command == "deployment"
    assert args.deployment_command == "inspect"
    assert args.strict is True
    assert args.llm_max_requests == 25


def test_deployment_validate_command(tmp_path: Path, capsys) -> None:
    artifact_path = tmp_path / "deployment-artifact.yaml"
    _write_yaml(artifact_path, _artifact_payload())

    exit_code = main(["deployment", "validate", str(artifact_path)])

    assert exit_code == 0
    assert "valid deployment artifact" in capsys.readouterr().out


def test_deployment_explain_command(tmp_path: Path, capsys) -> None:
    artifact_path = tmp_path / "deployment-artifact.yaml"
    _write_yaml(artifact_path, _artifact_payload())

    exit_code = main(
        [
            "deployment",
            "explain",
            str(artifact_path),
            "--block",
            "shared-services",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Shared Services (shared-services)" in output
    assert "grouping: The services share a target and lifecycle." in output


def test_deployment_inspect_strict_rejects_unpinned_git(tmp_path: Path, capsys) -> None:
    sources_path = tmp_path / "sources.yaml"
    _write_yaml(
        sources_path,
        {
            "system": "demo",
            "sources": [
                {
                    "id": "demo-repo",
                    "kind": "git",
                    "location": "https://example.test/demo.git",
                }
            ],
        },
    )

    output = tmp_path / "output"
    exit_code = main(
        [
            "deployment",
            "inspect",
            "--sources",
            str(sources_path),
            "--output",
            str(output),
            "--strict",
        ]
    )

    assert exit_code == 1
    assert "strict mode requires revisions" in capsys.readouterr().err
    assert "[failed] strict mode requires revisions" in (output / "inspection.log").read_text()
    assert not list(output.glob(".staging-*"))


def test_deployment_inspect_acquires_local_source_and_writes_sidecars(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("# Fixture deployment\n", encoding="utf-8")
    sources_path = tmp_path / "sources.yaml"
    _write_yaml(
        sources_path,
        {
            "system": "fixture",
            "sources": [
                {
                    "id": "fixture-source",
                    "kind": "local_directory",
                    "location": "source",
                }
            ],
        },
    )
    output = tmp_path / "output"

    exit_code = main(
        [
            "deployment",
            "inspect",
            "--sources",
            str(sources_path),
            "--output",
            str(output),
            "--strict",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "source acquisition complete: 1 source(s)" in captured.out
    assert "[1/8] Acquiring pinned deployment sources" in captured.err
    assert "[done] published" in captured.err
    assert (output / "source-manifest.json").is_file()
    assert (output / "repository-inventory.json").is_file()
    assert (output / "evidence.jsonl").read_text(encoding="utf-8")
    assert (output / "deterministic-findings.jsonl").is_file()
    assert (output / "facts.jsonl").is_file()
    assert (output / "unresolved-questions.yaml").is_file()
    assert (output / "fact-extraction.json").is_file()
    assert (output / "deployment-artifact.yaml").is_file()
    assert (output / "dependency-graph.json").is_file()
    assert (output / "inspection-report.md").is_file()
    assert (output / "output-manifest.json").is_file()
    assert "[8/8] Publishing generated outputs atomically" in (output / "inspection.log").read_text(
        encoding="utf-8"
    )
