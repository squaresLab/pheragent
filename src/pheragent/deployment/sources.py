"""Acquire pinned sources and provide mutable run workspaces."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

from .models import SourceKind, SourcesConfig, SourceSpec
from .redaction import redact_secrets
from .source_manager import AcquiredSource, SourceManager
from .task import SourceLocation


def _source_parts(item: str | SourceLocation, base: Path) -> tuple[str, str]:
    location = item if isinstance(item, str) else item.location
    if location.startswith(("https://", "http://", "git@", "ssh://")):
        path = location.split(":", 1)[1] if location.startswith("git@") else urlsplit(location).path
        parts = [part for part in path.strip("/").split("/") if part]
        return (parts[-2] if len(parts) > 1 else "", parts[-1].removesuffix(".git"))
    path = Path(location)
    return "", ((base / path) if not path.is_absolute() else path).resolve().name


def _source_names(items: list[str | SourceLocation], base: Path) -> list[str]:
    parts = [_source_parts(item, base) for item in items]
    names = [
        f"{owner}/{name}" if sum(other == name for _, other in parts) > 1 and owner else name
        for owner, name in parts
    ]
    if len(names) != len(set(names)):
        raise ValueError("source names clash; use repositories with distinct names")
    return names


def _source_spec(item: str | SourceLocation, purpose: str, name: str, base: Path) -> SourceSpec:
    location = item if isinstance(item, str) else item.location
    revision = None if isinstance(item, str) else item.revision
    if location.startswith(("https://", "http://", "git@", "ssh://")):
        kind = SourceKind.GIT
    else:
        local = Path(location)
        local = local if local.is_absolute() else base / local
        kind = SourceKind.LOCAL_FILE if local.is_file() else SourceKind.LOCAL_DIRECTORY
    return SourceSpec(
        id=name,
        kind=kind,
        location=location,
        purpose=purpose,
        revision=revision,
    )


class DeploymentSources:
    """Acquire immutable source snapshots for one deployment run."""

    @classmethod
    def acquire(
        cls,
        task,
        *,
        base: Path,
        cache_dir: Path,
        workspace: Path,
    ) -> DeploymentSources:
        entries = [
            (purpose, item)
            for purpose, items in (
                ("repository", task.sources.repositories),
                ("documentation", task.sources.documentation),
            )
            for item in items
        ]
        names = _source_names([item for _, item in entries], base)
        specs = [
            _source_spec(item, purpose, name, base)
            for (purpose, item), name in zip(entries, names, strict=True)
        ]
        acquired = SourceManager(
            cache_dir=cache_dir,
            config_dir=base,
            strict=False,
        ).acquire(SourcesConfig(system="deployment", sources=specs))
        instance = cls()
        instance.manifest = acquired.manifest
        instance.tools = SourceWorkspace(acquired.sources)
        instance.tools.prepare(workspace)
        return instance


class SourceWorkspace:
    def __init__(self, sources: tuple[AcquiredSource, ...]):
        self.sources = {source.id: source for source in sources}

    def prepare(self, workspace: Path) -> None:
        workspace.mkdir(parents=True, exist_ok=True)
        for source in self.sources.values():
            target = workspace / source.id
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.path.is_file():
                shutil.copy2(source.path, target)
            elif source.kind == SourceKind.GIT:
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(source.path),
                        "worktree",
                        "add",
                        "--detach",
                        str(target),
                        source.manifest.resolved_revision or "HEAD",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                shutil.copytree(
                    source.path,
                    target,
                    symlinks=True,
                    ignore=shutil.ignore_patterns(".git", ".pheragent"),
                )

    def working_directory(self, reference: str | None, workspace: Path) -> Path:
        root = workspace.resolve()
        path = (root / (reference or ".")).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("working_directory escapes the source workspace") from exc
        if not path.is_dir():
            available = ", ".join(sorted(item.name for item in root.iterdir())) or "none"
            raise ValueError(
                f"working_directory does not exist: {reference}; available: {available}"
            )
        return path

    def workspace_diff(self, workspace: Path) -> str:
        diffs = []
        for source_id, source in self.sources.items():
            copy = workspace / source_id
            if not copy.exists() or source.path.is_file():
                continue
            result = subprocess.run(
                [
                    "diff",
                    "-ruN",
                    "--exclude=.git",
                    "--exclude=.pheragent",
                    str(source.path),
                    str(copy),
                ],
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
            if result.returncode == 1:
                diffs.append(result.stdout)
        return redact_secrets("".join(diffs))
