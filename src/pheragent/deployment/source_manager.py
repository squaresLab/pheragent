from __future__ import annotations

import hashlib
import os
import shutil
import stat
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from pheragent.models import CommandResult
from pheragent.process import run_command

from .enums import SourceKind
from .models import SourceManifest, SourceManifestEntry, SourcesConfig, SourceSpec

CommandRunner = Callable[[list[str], Path | None], CommandResult]
ProgressCallback = Callable[[str], None]

_DEFAULT_IGNORED_DIRECTORIES = {
    ".git",
    ".pheragent",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "target",
    "vendor",
}


@dataclass(frozen=True, slots=True)
class AcquiredSource:
    """A pinned source snapshot with containment-safe artifact access."""

    id: str
    kind: SourceKind
    path: Path
    spec: SourceSpec
    manifest: SourceManifestEntry

    def resolve_path(self, relative_path: str) -> Path:
        if self.path.is_file():
            if relative_path != self.path.name:
                raise ValueError(f"source {self.id} has no artifact: {relative_path}")
            return self.path.resolve()
        root = self.path.resolve()
        candidate = root.joinpath(*PurePosixPath(relative_path).parts).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"artifact path escapes source {self.id}: {relative_path}") from exc
        return candidate

    def contains_file(self, relative_path: str) -> bool:
        try:
            return self.resolve_path(relative_path).is_file()
        except OSError, ValueError:
            return False


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    manifest: SourceManifest
    sources: tuple[AcquiredSource, ...]


class SourceManager:
    def __init__(
        self,
        *,
        cache_dir: Path,
        config_dir: Path,
        strict: bool,
        timeout: float = 900.0,
        command_runner: CommandRunner | None = None,
        progress: ProgressCallback | None = None,
    ):
        self.cache_dir = cache_dir.expanduser().resolve()
        self.config_dir = config_dir.expanduser().resolve()
        self.strict = strict
        self.timeout = timeout
        self.command_runner = command_runner or self._run_command
        self.progress = progress or (lambda _message: None)

    def acquire(self, config: SourcesConfig) -> AcquisitionResult:
        if self.strict:
            unpinned = sorted(
                source.id
                for source in config.sources
                if source.kind == SourceKind.GIT and not source.revision
            )
            if unpinned:
                raise ValueError(
                    "strict mode requires revisions for Git sources: " + ", ".join(unpinned)
                )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        acquired_items = []
        for index, source in enumerate(config.sources, start=1):
            self.progress(f"acquiring source {index}/{len(config.sources)}: {source.id}")
            acquired_items.append(self._acquire_source(source))
            self.progress(f"acquired source {source.id}")
        acquired = tuple(acquired_items)
        manifest = SourceManifest(
            manifest_version="0.1",
            system=config.system,
            sources=[source.manifest for source in acquired],
        )
        return AcquisitionResult(manifest=manifest, sources=acquired)

    def _acquire_source(self, source: SourceSpec) -> AcquiredSource:
        if source.kind == SourceKind.GIT:
            return self._acquire_git(source)
        if source.kind in {SourceKind.LOCAL_DIRECTORY, SourceKind.LOCAL_FILE}:
            return self._acquire_local(source)
        raise ValueError(f"unsupported source kind: {source.kind}")

    def _acquire_git(self, source: SourceSpec) -> AcquiredSource:
        if self.strict and not source.revision:
            raise ValueError(f"strict mode requires a revision for Git source {source.id}")
        _reject_embedded_credentials(source.location)
        requested_revision = source.revision or "HEAD"
        cache_key = _cache_key(source.location, requested_revision)
        repository_path = self.cache_dir / "git" / f"{source.id}-{cache_key}"
        repository_path.parent.mkdir(parents=True, exist_ok=True)

        if repository_path.exists():
            self.progress(f"using cached Git repository for {source.id}")
            if not (repository_path / ".git").is_dir():
                raise ValueError(f"source cache is not a Git repository: {repository_path}")
            status = self._git(["status", "--porcelain"], cwd=repository_path)
            if status.stdout.strip():
                raise ValueError(f"source cache contains local changes: {repository_path}")
            self._git(
                ["remote", "set-url", "origin", source.location],
                cwd=repository_path,
            )
        else:
            self.progress(f"cloning Git source {source.id}")
            self._git(
                [
                    "clone",
                    "--no-checkout",
                    "--filter=blob:none",
                    source.location,
                    str(repository_path),
                ]
            )

        self.progress(f"resolving revision {requested_revision} for {source.id}")
        fetch = self._git(
            ["fetch", "--force", "origin", requested_revision],
            cwd=repository_path,
            check=False,
        )
        if fetch.ok:
            resolved_revision = self._git(
                ["rev-parse", "--verify", "FETCH_HEAD^{commit}"],
                cwd=repository_path,
            ).stdout.strip()
        else:
            self._git(["fetch", "--force", "--tags", "origin"], cwd=repository_path)
            resolved_revision = self._git(
                ["rev-parse", "--verify", f"{requested_revision}^{{commit}}"],
                cwd=repository_path,
            ).stdout.strip()

        self._git(["checkout", "--detach", resolved_revision], cwd=repository_path)
        self.progress(f"pinned {source.id} to {resolved_revision}")
        inspected_path = _resolve_root_path(repository_path, source.root_path)
        if not inspected_path.is_dir():
            raise ValueError(f"Git source root_path is not a directory: {source.root_path}")
        content_hash = self._git_tree_hash(
            repository_path,
            resolved_revision,
            source.root_path,
        )
        manifest = SourceManifestEntry(
            id=source.id,
            kind=source.kind,
            location=source.location,
            purpose=source.purpose,
            requested_revision=source.revision,
            resolved_revision=resolved_revision,
            root_path=source.root_path,
            include_patterns=source.include_patterns,
            exclude_patterns=source.exclude_patterns,
            content_hash=content_hash,
        )
        return AcquiredSource(
            id=source.id,
            kind=source.kind,
            path=inspected_path,
            spec=source,
            manifest=manifest,
        )

    def _acquire_local(self, source: SourceSpec) -> AcquiredSource:
        location = Path(source.location).expanduser()
        if not location.is_absolute():
            location = self.config_dir / location
        try:
            location = location.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"local source does not exist: {source.location}") from exc

        if source.kind == SourceKind.LOCAL_DIRECTORY:
            if not location.is_dir():
                raise ValueError(f"local_directory source is not a directory: {location}")
            inspected_path = _resolve_root_path(location, source.root_path)
            if not inspected_path.is_dir():
                raise ValueError(
                    f"local_directory root_path is not a directory: {source.root_path}"
                )
        else:
            if not location.is_file():
                raise ValueError(f"local_file source is not a file: {location}")
            if source.root_path != ".":
                raise ValueError("local_file sources require root_path='.'")
            if location.is_symlink():
                raise ValueError("local_file sources may not be symbolic links")
            inspected_path = location

        content_hash = _content_hash(inspected_path)
        snapshot_path = self._snapshot_local_source(source, inspected_path, content_hash)
        manifest = SourceManifestEntry(
            id=source.id,
            kind=source.kind,
            location=source.location,
            purpose=source.purpose,
            root_path=source.root_path,
            include_patterns=source.include_patterns,
            exclude_patterns=source.exclude_patterns,
            content_hash=content_hash,
        )
        return AcquiredSource(
            id=source.id,
            kind=source.kind,
            path=snapshot_path,
            spec=source,
            manifest=manifest,
        )

    def _snapshot_local_source(
        self,
        source: SourceSpec,
        inspected_path: Path,
        content_hash: str,
    ) -> Path:
        snapshot_root = self.cache_dir / "local" / f"{source.id}-{content_hash[:16]}"
        if inspected_path.is_file():
            target = snapshot_root / inspected_path.name
            if target.exists():
                if _content_hash(target) != content_hash:
                    raise ValueError(f"local source cache was modified: {target}")
                return target
        elif snapshot_root.exists():
            if _content_hash(snapshot_root) != content_hash:
                raise ValueError(f"local source cache was modified: {snapshot_root}")
            return snapshot_root

        snapshot_root.parent.mkdir(parents=True, exist_ok=True)
        temporary = snapshot_root.with_name(f".{snapshot_root.name}.tmp-{uuid.uuid4().hex}")
        try:
            if inspected_path.is_file():
                temporary.mkdir(parents=True)
                shutil.copy2(inspected_path, temporary / inspected_path.name, follow_symlinks=False)
            else:
                shutil.copytree(
                    inspected_path,
                    temporary,
                    symlinks=True,
                    ignore=_copy_ignore,
                )
            os.replace(temporary, snapshot_root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        if inspected_path.is_file():
            return snapshot_root / inspected_path.name
        return snapshot_root

    def _git_tree_hash(self, repository: Path, revision: str, root_path: str) -> str:
        command = ["ls-tree", "-r", "--full-tree", revision]
        if root_path != ".":
            command.extend(["--", PurePosixPath(root_path).as_posix()])
        tree = self._git(command, cwd=repository).stdout.encode()
        return hashlib.sha256(tree).hexdigest()

    def _git(
        self,
        arguments: list[str],
        *,
        cwd: Path | None = None,
        check: bool = True,
    ) -> CommandResult:
        result = self.command_runner(["git", *arguments], cwd)
        if check and not result.ok:
            detail = result.combined_output.strip() or "no command output"
            raise RuntimeError(f"Git command failed: git {' '.join(arguments)}\n{detail}")
        return result

    def _run_command(self, command: list[str], cwd: Path | None) -> CommandResult:
        return run_command(command, timeout=self.timeout, cwd=cwd)


def _resolve_root_path(source_path: Path, root_path: str) -> Path:
    normalized = PurePosixPath(root_path)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise ValueError(f"source root_path must stay inside the source: {root_path}")
    candidate = source_path.joinpath(*normalized.parts).resolve(strict=True)
    try:
        candidate.relative_to(source_path.resolve())
    except ValueError as exc:
        raise ValueError(f"source root_path escapes the source: {root_path}") from exc
    return candidate


def _cache_key(location: str, revision: str) -> str:
    return hashlib.sha256(f"{location}\0{revision}".encode()).hexdigest()[:16]


def _reject_embedded_credentials(location: str) -> None:
    parsed = urlsplit(location)
    if parsed.scheme in {"http", "https"} and (parsed.username or parsed.password):
        raise ValueError("Git source URLs may not contain embedded credentials")


def _content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        _hash_file(digest, path, path.name)
        return digest.hexdigest()

    for candidate in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        relative = candidate.relative_to(path)
        if any(part in _DEFAULT_IGNORED_DIRECTORIES for part in relative.parts):
            continue
        relative_text = relative.as_posix()
        mode = candidate.lstat().st_mode
        if stat.S_ISLNK(mode):
            _update_hash(digest, "symlink", relative_text, os.readlink(candidate))
        elif stat.S_ISREG(mode):
            _hash_file(digest, candidate, relative_text)
        elif stat.S_ISDIR(mode):
            _update_hash(digest, "directory", relative_text, "")
    return digest.hexdigest()


def _hash_file(digest: Any, path: Path, relative_path: str) -> None:
    _update_hash(digest, "file", relative_path, "")
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)


def _update_hash(digest: Any, kind: str, relative_path: str, value: str) -> None:
    for item in (kind, relative_path, value):
        encoded = item.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in _DEFAULT_IGNORED_DIRECTORIES}
