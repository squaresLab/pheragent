from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pheragent.utils import slugify


class AtomicOutputTransaction:
    """Stage generated files and atomically replace each published file on commit."""

    def __init__(self, output_dir: Path, *, write_manifest: bool = True) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        self.write_manifest = write_manifest
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir = self.output_dir / f".staging-{uuid.uuid4().hex}"
        self.staging_dir.mkdir()
        self._committed = False

    def commit(self) -> tuple[Path, ...]:
        manifest_name = "output-manifest.json"
        staged_files = sorted(
            path
            for path in self.staging_dir.rglob("*")
            if path.is_file() and path.name != manifest_name
        )
        manifest_payload = {
            "manifest_version": "0.1",
            "files": {
                path.relative_to(self.staging_dir).as_posix(): _sha256(path)
                for path in staged_files
            },
        }
        manifest_path = self.staging_dir / manifest_name
        if self.write_manifest:
            manifest_path.write_text(
                json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        published: list[Path] = []
        files_to_publish = (*staged_files, manifest_path) if self.write_manifest else staged_files
        for staged in files_to_publish:
            relative = staged.relative_to(self.staging_dir)
            target = self.output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, target)
            published.append(target)
        self._cleanup()
        self._committed = True
        return tuple(published)

    def _cleanup(self) -> None:
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)

    def __enter__(self) -> AtomicOutputTransaction:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        if not self._committed:
            self._cleanup()


def create_timestamped_run_directory(
    output_root: Path,
    *,
    name: str,
    now: datetime | None = None,
) -> Path:
    """Allocate a named, immutable UTC run directory without touching earlier runs."""
    root = output_root.expanduser().resolve()
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_slug = slugify(name, fallback="run")
    base_name = f"{timestamp}-{run_slug}"
    for suffix in range(1000):
        directory_name = base_name if suffix == 0 else f"{base_name}-{suffix:02d}"
        candidate = runs_dir / directory_name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"could not allocate a unique run directory under {runs_dir}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
