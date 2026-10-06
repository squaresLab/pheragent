"""Record durable deployment progress and compact working memory."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from .redaction import redact_secrets
from .sources import SourceTools
from .task import Decision


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _append(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(redact_secrets(json.dumps(item, ensure_ascii=False, default=str)) + "\n")


def _update_state(state: dict, decision: Decision, result: dict, sources: SourceTools) -> None:
    state["focus"] = decision.focus
    if decision.selected_route:
        evidence = sorted(sources.existing_refs(decision.evidence))
        if evidence:
            state["selected_route"] = {"choice": decision.selected_route, "evidence": evidence}
    state["unresolved"] = sorted(
        (set(state["unresolved"]) | set(decision.add_questions)) - set(decision.resolve_questions)
    )
    state["gaps"] = sorted(
        (set(state["gaps"]) | set(decision.add_gaps)) - set(decision.resolve_gaps)
    )
    state["last_action"] = {
        "tool": decision.tool,
        "reason": decision.reason,
        "outcome": result.get("status", result.get("exit_code")),
    }
    if decision.command:
        state["last_action"]["command"] = decision.command
    if decision.working_directory:
        state["last_action"]["working_directory"] = decision.working_directory
    if decision.working_memory:
        state["working_memory"] = decision.working_memory


def _fingerprint(state: dict) -> str:
    relevant = {
        key: state[key]
        for key in (
            "gaps",
            "unresolved",
            "focus",
            "milestones",
            "verified_outcomes",
            "evidence",
            "selected_route",
            "working_memory",
        )
    }
    relevant["health"] = {
        name: [" ".join(line.split()[:4]) for line in value["summary"].splitlines()[1:]]
        for name, value in state.get("environment", {}).items()
        if name in {"workloads", "storage"}
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()


def _record_outcome(state: dict, decision: Decision, result: dict) -> None:
    """Count a named capability only when read-only checks proved a state transition."""
    name = (decision.outcome_id or "").strip()
    if not name or result["status"] != "validated":
        return
    if not result["before_absent"]:
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
