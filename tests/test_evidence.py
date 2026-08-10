from __future__ import annotations

import json
from pathlib import Path

import pytest

from pheragent.deployment.enums import SourceKind
from pheragent.deployment.evidence import EvidenceStore
from pheragent.deployment.models import SourceManifestEntry, SourceSpec
from pheragent.deployment.source_manager import AcquiredSource


def _source(path: Path) -> AcquiredSource:
    return AcquiredSource(
        id="fixture-source",
        kind=SourceKind.LOCAL_DIRECTORY,
        path=path,
        spec=SourceSpec(
            id="fixture-source",
            kind=SourceKind.LOCAL_DIRECTORY,
            location="fixture",
        ),
        manifest=SourceManifestEntry(
            id="fixture-source",
            kind=SourceKind.LOCAL_DIRECTORY,
            location="fixture",
            content_hash="a" * 64,
        ),
    )


def test_evidence_capture_preserves_lines_and_redacts_secrets(tmp_path: Path) -> None:
    source_path = tmp_path / "source"
    source_path.mkdir()
    (source_path / "config.txt").write_text(
        "heading\npassword=actual-password\ntoken=${TOKEN}\n--client-secret actual-secret\n",
        encoding="utf-8",
    )
    store = EvidenceStore()

    first = store.capture(
        _source(source_path),
        path="config.txt",
        start_line=2,
        end_line=4,
        heading="Credentials",
    )
    second = store.capture(
        _source(source_path),
        path="config.txt",
        start_line=2,
        end_line=4,
        heading="Credentials",
    )

    assert first == second
    assert first.excerpt == ("password=[REDACTED]\ntoken=${TOKEN}\n--client-secret [REDACTED]")
    assert "actual-password" not in first.excerpt
    assert len(store.records) == 1


def test_evidence_capture_rejects_invalid_line_range(tmp_path: Path) -> None:
    source_path = tmp_path / "source"
    source_path.mkdir()
    (source_path / "README.md").write_text("one line\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid evidence line range"):
        EvidenceStore().capture(
            _source(source_path),
            path="README.md",
            start_line=1,
            end_line=2,
        )


def test_evidence_capture_rejects_path_escape(tmp_path: Path) -> None:
    source_path = tmp_path / "source"
    source_path.mkdir()
    (tmp_path / "outside.txt").write_text("outside\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must stay inside"):
        EvidenceStore().capture(
            _source(source_path),
            path="../outside.txt",
            start_line=1,
            end_line=1,
        )


def test_evidence_jsonl_is_deterministic(tmp_path: Path) -> None:
    source_path = tmp_path / "source"
    source_path.mkdir()
    (source_path / "README.md").write_text("first\nsecond\n", encoding="utf-8")
    store = EvidenceStore()
    source = _source(source_path)
    second = store.capture(source, path="README.md", start_line=2, end_line=2)
    first = store.capture(source, path="README.md", start_line=1, end_line=1)
    output = tmp_path / "evidence.jsonl"

    store.write(output)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert [record["id"] for record in records] == [first.id, second.id]
