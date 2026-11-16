"""Append-only deployment history and artifacts derived from it."""

from __future__ import annotations

import json
import os
import shlex
from datetime import UTC, datetime
from pathlib import Path

from .redaction import redact_secrets
from .telemetry import record_event


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunHistory:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "events.jsonl"
        self._sequence = sum(1 for _ in self._events())

    def append(self, event: str, **data) -> dict:
        self._sequence += 1
        record = {"sequence": self._sequence, "time": _now(), "event": event, **data}
        encoded = redact_secrets(json.dumps(record, ensure_ascii=False, default=str))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        record_event(event, data)
        return record

    def events(self) -> list[dict]:
        return list(self._events())

    def usage(self) -> dict[str, int]:
        total: dict[str, int] = {}
        for event in self._events():
            for key, value in event.get("usage", {}).items():
                total[key] = total.get(key, 0) + value
        return total

    def runbook(self) -> str:
        events = self.events()
        evidence = {
            sequence
            for item in events
            if item["event"] == "component_recorded" and item.get("deployed") is True
            for sequence in item.get("evidence_events", [])
        }
        lines = [
            "#!/bin/sh",
            "set -eu",
            'RUN_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)',
        ]
        for event in events:
            if event["event"] == "sources_acquired":
                for source in event["sources"]["sources"]:
                    revision = source.get("resolved_revision") or source["content_hash"]
                    lines.append(f"# source {source['id']}: {source['location']} @ {revision}")
            if event["event"] != "shell_result" or event["sequence"] not in evidence:
                continue
            if not event.get("state_changing") or event.get("exit_code") != 0:
                continue
            lines.append("")
            if event.get("working_directory"):
                source_id, _, relative = event["working_directory"].partition(":")
                path = Path("workspace") / source_id / (relative or ".")
                lines.append(
                    f'(cd "$RUN_DIR" && cd {shlex.quote(str(path))} && '
                    f'{shlex.join(event["command"])})'
                )
            else:
                lines.append(shlex.join(event["command"]))
        return "\n".join(lines) + "\n"

    def _events(self):
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
