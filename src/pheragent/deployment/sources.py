"""Acquire pinned sources and provide mutable run workspaces."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .models import SourceKind, SourcesConfig, SourceSpec
from .redaction import redact_secrets
from .source_manager import AcquiredSource, SourceManager
from .task import SourceLocation


def _source_spec(item: str | SourceLocation, purpose: str, index: int, base: Path) -> SourceSpec:
    location = item if isinstance(item, str) else item.location
    revision = None if isinstance(item, str) else item.revision
    if location.startswith(("https://", "http://", "git@", "ssh://")):
        kind = SourceKind.GIT
    else:
        local = Path(location)
        local = local if local.is_absolute() else base / local
        kind = SourceKind.LOCAL_FILE if local.is_file() else SourceKind.LOCAL_DIRECTORY
    return SourceSpec(
        id=f"{purpose}-{index}",
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
    ) -> DeploymentSources:
        specs = [
            _source_spec(item, purpose, index, base)
            for purpose, items in (
                ("repository", task.sources.repositories),
                ("documentation", task.sources.documentation),
            )
            for index, item in enumerate(items, start=1)
        ]
        acquired = SourceManager(
            cache_dir=cache_dir,
            config_dir=base,
            strict=False,
        ).acquire(SourcesConfig(system="deployment", sources=specs))
        instance = cls()
        instance.manifest = acquired.manifest
        instance.tools = SourceWorkspace(acquired.sources)
        return instance


class SourceWorkspace:
    def __init__(self, sources: tuple[AcquiredSource, ...]):
        self.sources = {source.id: source for source in sources}

    def working_directory(self, reference: str | None, workspace: Path) -> Path | None:
        if reference is None:
            return None
        source_id, separator, relative = reference.partition(":")
        if not separator and source_id in self.sources:
            relative = "."
        if source_id not in self.sources:
            raise ValueError("working_directory must be source-id:relative/directory")
        source = self.sources[source_id]
        if source.path.is_file():
            raise ValueError("file sources cannot be command working directories")
        copy = workspace / source_id
        if not copy.exists():
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                source.path,
                copy,
                symlinks=True,
                ignore=shutil.ignore_patterns(".git", ".pheragent"),
            )
        root = copy.resolve()
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("working_directory escapes the source workspace") from exc
        if not path.is_dir():
            raise ValueError("working_directory does not exist")
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
