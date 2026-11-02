"""Append-only deployment history and derived context."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from .redaction import redact_secrets
from .telemetry import record_event


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunHistory:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
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

    def checkpoint(self, state: dict) -> None:
        self.append("checkpoint", state=state)

    def restore(self) -> dict:
        for event in reversed(self.events()):
            if event["event"] == "checkpoint":
                return event["state"]
        raise ValueError("--resume requires a run with a checkpoint event")

    def events(self) -> list[dict]:
        return list(self._events())

    def context(self, *, window: int, step_id: str | None = None) -> list[dict]:
        context = []
        decision_step = None
        for event in self._events():
            if event["event"] == "decision":
                decision_step = event.get("value", {}).get("step_id")
            if event["event"] not in {
                "decision",
                "tool",
                "human_choice",
                "human_input",
                "overview_revised",
            }:
                continue
            if step_id and decision_step != step_id:
                continue
            context.append(_event_summary(event))
        return context[-window:]

    def usage(self) -> dict[str, int]:
        total: dict[str, int] = {}
        for event in self._events():
            for key, value in event.get("usage", {}).items():
                total[key] = total.get(key, 0) + value
        return total

    def summary(self, report: dict) -> dict:
        events = self.events()
        summary = {
            "status": report["status"],
            "reason": report["reason"],
            "iterations": report["cycles"],
            "mutating_actions": report["mutating_actions"],
            "usage": report["usage"],
            "verified_outcomes": [
                item["id"] for item in report["state"].get("verified_outcomes", [])
            ],
            "events": len(events) + 1,
        }
        self.append("stop", result=summary)
        return summary

    def _events(self):
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _event_summary(event: dict) -> dict:
    if event["event"] == "decision":
        value = event.get("value", {})
        fields = ("kind", "tool", "focus", "reason", "step_id", "source_path", "command")
        return {"event": "decision", **{key: value[key] for key in fields if value.get(key)}}
    result = event.get("result", {})
    fields = ("status", "reason", "source", "error", "repeated_result_count")
    return {
        "event": event["event"],
        **({"tool": event["tool"]} if event.get("tool") else {}),
        **{key: result[key] for key in fields if result.get(key) is not None},
    }
