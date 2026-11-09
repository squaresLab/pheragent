"""Append-only deployment history and derived context."""

from __future__ import annotations

import json
import os
import shlex
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from .redaction import redact_secrets
from .retrieval import normalize_terms
from .telemetry import record_event

_SOURCE_TOOLS = {"read_file", "search_sources"}


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
                "action_started",
                "action_finished",
                "human_choice",
                "human_input",
                "overview_revised",
            }:
                continue
            if step_id and decision_step != step_id:
                continue
            context.append(_event_summary(event))
        return context[-window:]

    def recall_source(self, decision, *, source_signature: str) -> dict | None:
        """Reuse a prior source result when the source set and request still match."""
        records = self._source_records(source_signature)
        match = None
        if decision.tool == "read_file":
            match = next(
                (
                    record
                    for record in reversed(records)
                    if record["tool"] == "read_file"
                    and record.get("source_path") == decision.source_path
                    and record.get("start_line") == decision.start_line
                    and record.get("end_line") == decision.end_line
                ),
                None,
            )
        elif decision.tool == "search_sources" and decision.query:
            requested = set(normalize_terms(decision.query))
            candidates = [
                record
                for record in records
                if record["tool"] == "search_sources" and record.get("step_id") == decision.step_id
            ]
            match = next(
                (
                    record
                    for record in reversed(candidates)
                    if set(normalize_terms(record.get("query", ""))) == requested
                ),
                None,
            )
            if match is None:
                match = next(
                    (
                        record
                        for record in reversed(candidates)
                        if _term_overlap(
                            requested,
                            set(normalize_terms(record.get("query", ""))),
                        )
                        >= 0.75
                    ),
                    None,
                )
        if match is None:
            return None
        result = deepcopy(match["result"])
        result["memory_hit"] = True
        result["memory_event_sequence"] = match["sequence"]
        return result

    def usage(self) -> dict[str, int]:
        total: dict[str, int] = {}
        for event in self._events():
            for key, value in event.get("usage", {}).items():
                total[key] = total.get(key, 0) + value
        return total

    def runbook(self) -> str:
        events = self.events()
        finished = {
            event.get("iteration"): event.get("result", {})
            for event in events
            if event["event"] == "action_finished"
        }
        lines = [
            "#!/bin/sh",
            "# Observed commands. Replay assumes the same initial state and source revisions.",
            "set -u",
        ]
        for event in events:
            if event["event"] == "sources_acquired":
                for source in event["sources"]["sources"]:
                    revision = source.get("resolved_revision") or source["content_hash"]
                    lines.append(f"# source {source['id']}: {source['location']} @ {revision}")
            if event["event"] != "action_started":
                continue
            result = finished.get(event.get("iteration"), {})
            lines.extend(
                [
                    "",
                    f"# step: {event.get('step_id') or 'unassigned'}",
                    f"# result: {result.get('status', 'unfinished')}",
                ]
            )
            if event.get("working_directory"):
                lines.append(f"# working directory: {event['working_directory']}")
            command = shlex.join(event["command"])
            if event.get("stdin_input"):
                name = event["stdin_input"]
                command = f'printf "%s\\n" "${{{name}:?set {name}}}" | {command}'
            lines.append(command)
            for check in result.get("after", []):
                lines.append(f"# verify: {shlex.join(check['command'])}")
        return "\n".join(lines) + "\n"

    def _events(self):
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def _source_records(self, source_signature: str) -> list[dict]:
        records = []
        for event in self._events():
            if (
                event["event"] != "tool"
                or event.get("tool") not in _SOURCE_TOOLS
                or event.get("source_signature") != source_signature
            ):
                continue
            result = event.get("result", {})
            if result.get("memory_hit") or result.get("error"):
                continue
            if event["tool"] == "search_sources" and not result.get("hits"):
                continue
            if event["tool"] == "read_file" and "text" not in result:
                continue
            records.append(
                {
                    "sequence": event["sequence"],
                    "tool": event["tool"],
                    "result": result,
                    **event.get("source_request", {}),
                }
            )
        return records


def _event_summary(event: dict) -> dict:
    if event["event"] == "decision":
        value = event.get("value", {})
        fields = ("kind", "tool", "focus", "reason", "step_id", "source_path", "command")
        return {"event": "decision", **{key: value[key] for key in fields if value.get(key)}}
    if event["event"] == "action_started":
        fields = ("tool", "command", "working_directory", "step_id", "expected_change")
        return {"event": "action_started", **{key: event[key] for key in fields if event.get(key)}}
    result = event.get("result", {})
    fields = ("status", "reason", "source", "error", "repeated_result_count")
    return {
        "event": event["event"],
        **({"tool": event["tool"]} if event.get("tool") else {}),
        **{key: result[key] for key in fields if result.get(key) is not None},
    }


def _term_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))
