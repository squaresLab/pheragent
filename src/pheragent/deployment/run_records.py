from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .redaction import redact_secrets
from .serialization import write_json


class RunRecorder:
    """Own the lifecycle and integrity metadata of one immutable analysis run."""

    def __init__(self, run_dir: Path, manifest: dict[str, Any]) -> None:
        self.run_dir = run_dir.expanduser().resolve()
        self.manifest = manifest
        self.started_at = time.monotonic()

    @classmethod
    def start(
        cls,
        run_dir: Path,
        *,
        run_kind: str,
        analysis_method: str,
        inputs: dict[str, Any],
    ) -> RunRecorder:
        resolved = run_dir.expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        if (resolved / "run-manifest.json").exists():
            raise ValueError(f"run directory is already recorded: {resolved}")
        recorder = cls(
            resolved,
            {
                "manifest_version": "0.2",
                "run_id": resolved.name,
                "run_kind": run_kind,
                "analysis_method": analysis_method,
                "status": "running",
                "started_at": _timestamp(),
                "code": _code_state(),
                "inputs": _redact(inputs),
                "artifacts": {},
            },
        )
        recorder._write_manifest()
        recorder.record_event("run", "started")
        return recorder

    def record_event(
        self,
        stage: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "timestamp": _timestamp(),
            "stage": stage,
            "message": redact_secrets(message),
            "details": _redact(details or {}),
        }
        path = self.run_dir / "events.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")

    def complete(
        self,
        *,
        metrics: dict[str, Any],
        sources: dict[str, Any],
        llm: dict[str, Any],
    ) -> None:
        write_json(self.run_dir / "metrics.json", metrics)
        self.manifest.update(
            {
                "status": "completed",
                "completed_at": _timestamp(),
                "duration_seconds": round(time.monotonic() - self.started_at, 6),
                "sources": _redact(sources),
                "llm": _redact(llm),
                "artifacts": _artifact_hashes(self.run_dir),
            }
        )
        self.record_event("run", "completed")
        self._write_manifest()

    def fail(self, error: BaseException) -> None:
        self.manifest.update(
            {
                "status": "failed",
                "completed_at": _timestamp(),
                "duration_seconds": round(time.monotonic() - self.started_at, 6),
                "error": {
                    "type": type(error).__name__,
                    "message": redact_secrets(str(error)),
                },
                "artifacts": _artifact_hashes(self.run_dir),
            }
        )
        self.record_event("run", "failed", self.manifest["error"])
        self._write_manifest()

    def _write_manifest(self) -> None:
        write_json(self.run_dir / "run-manifest.json", self.manifest)


def _artifact_hashes(run_dir: Path) -> dict[str, str]:
    ignored = {"events.jsonl", "run-manifest.json"}
    return {
        path.relative_to(run_dir).as_posix(): _sha256(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in ignored
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _code_state() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    commit = _git(project_root, "rev-parse", "HEAD")
    status = _git(project_root, "status", "--porcelain")
    return {"commit": commit or None, "dirty": bool(status)}


def _git(project_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, Path):
        resolved = value.expanduser().resolve()
        return {
            "path": str(resolved),
            "sha256": _sha256(resolved) if resolved.is_file() else None,
        }
    if isinstance(value, dict):
        return {str(key): _redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value
