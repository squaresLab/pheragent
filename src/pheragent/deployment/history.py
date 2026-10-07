"""Append-only deployment history and derived context."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from .redaction import redact_secrets
from .sources import SourceTools
from .task import Decision
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

    def context(self, *, mode: str, window: int) -> list[dict] | dict:
        useful = [
            event
            for event in self._events()
            if event["event"] in {"decision", "tool", "human_choice", "overview_revised"}
        ]
        if mode == "recent":
            return useful[-window:]
        sources = {}
        for event in useful:
            result = event.get("result", {})
            if event.get("tool") == "read_file" and result.get("source"):
                text = result.get("text", "")
                sources[result["source"]] = {
                    "source": result["source"],
                    "total_lines": result.get("total_lines"),
                    "text": text if len(text) <= 3000 else text[:1500] + "\n[…]\n" + text[-1500:],
                }
        return {
            "decisions": sum(event["event"] == "decision" for event in useful),
            "tools": sum(event["event"] == "tool" for event in useful),
            "source_notes": list(sources.values())[-window:],
            "recent": [
                {
                    "event": event["event"],
                    "tool": event.get("tool"),
                    "focus": event.get("value", {}).get("focus"),
                    "status": event.get("result", {}).get("status"),
                    "reason": event.get("result", {}).get("reason"),
                }
                for event in useful[-min(window, 4) :]
            ],
        }

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


def update_state(state: dict, decision: Decision, result: dict, sources: SourceTools) -> None:
    state["focus"] = decision.focus
    if decision.selected_route:
        evidence = sorted(sources.existing_refs(decision.evidence))
        if evidence:
            state["selected_route"] = {"choice": decision.selected_route, "evidence": evidence}
    state["unresolved"] = sorted(
        (set(state["unresolved"]) | set(decision.add_questions)) - set(decision.resolve_questions)
    )
    gaps = [gap for gap in state.get("goal_stack", []) if gap not in decision.resolve_gaps]
    gaps.extend(gap for gap in decision.add_gaps if gap not in gaps)
    state["goal_stack"] = gaps
    state["gaps"] = gaps
    state["last_action"] = {
        "tool": decision.tool,
        "reason": decision.reason,
        "outcome": result.get("status", result.get("exit_code")),
    }
    if decision.command:
        state["last_action"]["command"] = decision.command
    if decision.working_directory:
        state["last_action"]["working_directory"] = decision.working_directory


def record_outcome(state: dict, decision: Decision, result: dict) -> None:
    name = (decision.outcome_id or "").strip()
    if not name or result["status"] != "validated" or not result["before_unsatisfied"]:
        return
    checks = [check.model_dump() for check in decision.validation]
    signature = sorted(json.dumps(check, sort_keys=True) for check in checks)
    if any(
        item["id"].casefold() == name.casefold()
        or sorted(json.dumps(check, sort_keys=True) for check in item["validation"]) == signature
        for item in state["verified_outcomes"]
    ):
        return
    state["verified_outcomes"].append(
        {"id": name, "validation": checks, "evidence": decision.evidence}
    )
    result["new_outcome"] = name
