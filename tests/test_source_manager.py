from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pheragent.deployment.enums import SourceKind
from pheragent.deployment.models import SourcesConfig, SourceSpec
from pheragent.deployment.source_manager import SourceManager


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _create_git_repository(path: Path) -> str:
    path.mkdir()
    _git(path, "init")
    _git(path, "config", "user.email", "fixture@example.test")
    _git(path, "config", "user.name", "Fixture")
    (path / "README.md").write_text("# Deployment\n\nRun ./install.sh\n", encoding="utf-8")
    (path / "install.sh").write_text("#!/bin/sh\necho install\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "fixture deployment")
    return _git(path, "rev-parse", "HEAD")


def test_git_source_is_resolved_to_detached_commit(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    commit = _create_git_repository(upstream)
    manager = SourceManager(
        cache_dir=tmp_path / "cache",
        config_dir=tmp_path,
        strict=True,
    )
    config = SourcesConfig(
        system="fixture",
        sources=[
            SourceSpec(
                id="fixture-git",
                kind=SourceKind.GIT,
                location=str(upstream),
                revision=commit,
            )
        ],
    )

    result = manager.acquire(config)
    acquired = result.sources[0]

    assert acquired.path != upstream
    assert acquired.manifest.requested_revision == commit
    assert acquired.manifest.resolved_revision == commit
    assert len(acquired.manifest.content_hash) == 64
    assert _git(acquired.path, "rev-parse", "HEAD") == commit
    assert _git(acquired.path, "status", "--porcelain") == ""
    assert _git(upstream, "status", "--porcelain") == ""


def test_git_tag_is_resolved_to_commit(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    commit = _create_git_repository(upstream)
    _git(upstream, "tag", "v1.0.0")
    manager = SourceManager(
        cache_dir=tmp_path / "cache",
        config_dir=tmp_path,
        strict=True,
    )
    config = SourcesConfig(
        system="fixture",
        sources=[
            SourceSpec(
                id="fixture-git",
                kind=SourceKind.GIT,
                location=str(upstream),
                revision="v1.0.0",
            )
        ],
    )

    result = manager.acquire(config)

    assert result.sources[0].manifest.requested_revision == "v1.0.0"
    assert result.sources[0].manifest.resolved_revision == commit


def test_strict_git_source_requires_revision(tmp_path: Path) -> None:
    manager = SourceManager(
        cache_dir=tmp_path / "cache",
        config_dir=tmp_path,
        strict=True,
    )
    config = SourcesConfig(
        system="fixture",
        sources=[
            SourceSpec(
                id="fixture-git",
                kind=SourceKind.GIT,
                location="https://example.test/fixture.git",
            )
        ],
    )

    with pytest.raises(ValueError, match="strict mode requires revisions"):
        manager.acquire(config)


def test_local_directory_is_copied_to_content_addressed_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "config" / "source"
    source.mkdir(parents=True)
    (source / "README.md").write_text("initial\n", encoding="utf-8")
    manager = SourceManager(
        cache_dir=tmp_path / "cache",
        config_dir=tmp_path / "config",
        strict=True,
    )
    config = SourcesConfig(
        system="fixture",
        sources=[
            SourceSpec(
                id="fixture-local",
                kind=SourceKind.LOCAL_DIRECTORY,
                location="source",
            )
        ],
    )

    result = manager.acquire(config)
    acquired = result.sources[0]
    (source / "README.md").write_text("changed\n", encoding="utf-8")

    assert acquired.path != source
    assert (acquired.path / "README.md").read_text(encoding="utf-8") == "initial\n"
    assert acquired.manifest.resolved_revision is None
    assert len(acquired.manifest.content_hash) == 64


def test_git_source_rejects_embedded_http_credentials(tmp_path: Path) -> None:
    manager = SourceManager(
        cache_dir=tmp_path / "cache",
        config_dir=tmp_path,
        strict=True,
    )
    config = SourcesConfig(
        system="fixture",
        sources=[
            SourceSpec(
                id="fixture-git",
                kind=SourceKind.GIT,
                location="https://user:token@example.test/fixture.git",
                revision="a" * 40,
            )
        ],
    )

    with pytest.raises(ValueError, match="may not contain embedded credentials"):
        manager.acquire(config)
