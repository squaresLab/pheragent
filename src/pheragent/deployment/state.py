"""Apply deployment decisions to the resumable working state."""

from __future__ import annotations

import json

from .sources import SourceTools
from .task import Decision


def apply_result(state: dict, decision: Decision, result: dict, sources: SourceTools) -> None:
    state["focus"] = decision.focus
    if decision.selected_route:
        evidence = sorted(sources.existing_refs(decision.evidence))
        if evidence:
            state["selected_route"] = {"choice": decision.selected_route, "evidence": evidence}
    state["unresolved"] = sorted(
        (set(state["unresolved"]) | set(decision.add_questions))
        - set(decision.resolve_questions)
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


def record_verified_outcome(state: dict, decision: Decision, result: dict) -> None:
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
