"""A small, resumable deployment agent loop."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .history import RunHistory
from .interaction import input_state, input_value, missing_inputs, prompt_for_inputs
from .llm import LLMClient
from .redaction import redact_secrets
from .runtime import execute_command
from .serialization import load_yaml, write_json, write_yaml
from .sources import DeploymentSources
from .task import Decision, DeploymentTask

_INSTRUCTIONS = """You are a senior DevOps engineer progressively deploying a system.

The task objective is authoritative. The context supplies source workspace paths, your current
step, the latest progress summary, prior component records, note index, and recent decisions with
their results. The recent history is a window, not the complete run record.
Start by finding the repository's deployment entrypoints and observing the target. Choose one
useful step at a time; revise your next step when source or runtime evidence changes the route.
Do not mark an outcome complete until runtime evidence verifies it.

Make one decision per turn:
- shell: run one ordinary argv command such as ls, find, rg, wc, sed, git, docker, kubectl,
  helm, terraform, ansible, make, or a repository script. Commands are argv, not shell pipelines.
- ask: request one atomic external value. Use a stable input name and mark secrets sensitive.
- record: record one component or outcome as deployed or not deployed. A deployed record must
  cite every successful shell event needed to reproduce the change, including file repairs.
  Put actual post-install checks in verification_events and include them in evidence_events.
  A not-deployed record must state the exact human action needed. A record is a checkpoint,
  not a final report. Immediately after runtime evidence verifies or rules out an outcome, record
  it before more investigation. Preserve relevant event sequence numbers in notes until recorded.
- finish: use after no safe, independent deployment work remains, or during budget finalization
  after the current outcome has been recorded.

Source inspection: use normal file commands. Locate root guidance and follow relevant links.
Use ["wc", "-l", "file"] to learn a file's length and ["sed", "-n", "1,120p", "file"]
to read a range. Do not include shell quote characters in argv. A range is not the whole file.
If output is truncated, the feedback gives the full log path and omitted range;
read what is missing before deciding that information is absent. Prefer complete high-signal
files when practical, but do not reread unchanged sections without a new question. Original
sources remain untouched; commands run in a per-run workspace. working_directory is an ordinary
path inside it, such as repo-name/deploy, and defaults to the workspace root.

Notes: use short, stable facts worth reusing, such as selected endpoints, commands, and diagnosed
repairs. Give each note a stable key and replace obsolete facts under that key. Update the
current-step note when your focus changes. After a meaningful milestone or change of course,
update the progress-summary note with verified state, remaining work, and essential references.
Choose those milestones yourself. Do not save raw output, private reasoning, or secrets. The
context includes the current step and latest summary; read run_notes.md only for other relevant
notes. Each note version remains in the event log.
Record source-supported prerequisites as prerequisite-* notes, citing a successful check when
available. Use task invariants to guide actions independently of the installation route. Record
route-independent system outcomes as invariant-* notes and cite successful end-to-end checks.
Check task invariants before finishing; never claim an untested one passed.

Verification must use executable checks whose success directly proves the required runtime
condition. A command that only lists or reports state is not proof merely because it exits zero;
interpret its output and prefer a predicate that exits nonzero when the condition is absent.
Before recording deployed, let transient starting, pending, or unknown states settle, verify each
required component is ready, and verify intended user-visible or health behavior when available.
For an aggregate outcome, verify every included component or record unverified components
separately. One healthy endpoint does not prove that an aggregate system is deployed.

Complete the task's deliverables; repository guidance explains how but does not replace the
task. Prefer repository-defined entrypoints. Preserve successful repairs as short reusable
patterns. If runtime behavior contradicts source, inspect the running artifact, then prefer
the smallest source-defined rebuild, reconfiguration, or redeployment.

A failed command is feedback. Diagnose it in the next turn, change the state or approach, and
continue. Once evidence establishes the root cause, stop investigating and attempt the smallest
safe source-supported repair. Make another observation only when one unresolved question could
change the next action. Before recording an outcome as not deployed, check notes for an
analogous successful repair, check the sources for a local build, configuration, or deployment
action, and attempt it when safe. Work is blocked only after an actual policy denial, missing
required input, failed repair, or confirmed absence of a source-supported action. A local build
using a repository-declared artifact name is not an external version-policy change. If one
component cannot progress, record it as not deployed and continue independent work. Do not stop
merely because one component is blocked. Unknown commands are permitted; destructive, host,
cloud, target, namespace, secret, and public-boundary changes may require human approval.

The last two requests are reserved for finalization. With two requests left, do not run another
shell command: record the current verified or not-deployed outcome. On the final request, finish
if that outcome is recorded; otherwise record it so the generated report preserves the result.
"""


def _restored_state(history: RunHistory) -> tuple[dict, dict, int]:
    components: dict[str, dict] = {}
    latest_feedback: dict = {}
    iteration = 0
    for event in history.events():
        iteration = max(iteration, event.get("iteration", 0))
        if event["event"] == "component_recorded":
            components[event["component"]] = {
                "name": event["component"],
                "deployed": event["deployed"],
                "reason": event["reason"],
                "evidence_events": event.get("evidence_events", []),
                "verification_events": event.get("verification_events", []),
                "human_action": event.get("human_action"),
            }
        elif event["event"] in {"shell_result", "input_result", "record_result"}:
            latest_feedback = event.get("result", {})
            if event["event"] == "shell_result":
                latest_feedback = {**latest_feedback, "event_sequence": event["sequence"]}
    return components, latest_feedback, iteration


def _successful_shell_events(history: RunHistory, sequences: list[int]) -> bool:
    successful = {
        event["sequence"]
        for event in history.events()
        if event["event"] == "shell_result" and event.get("exit_code") == 0
    }
    return bool(sequences) and set(sequences) <= successful


def _repeat_key(decision: Decision, result: dict) -> str:
    execution = result.get("execution", {})
    value = [
        decision.command,
        decision.working_directory,
        result.get("status"),
        execution.get("exit_code"),
        execution.get("stdout"),
        execution.get("stderr"),
    ]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _repeat_counts(history: RunHistory) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in history.events():
        if event["event"] == "shell_result" and event.get("repeat_key"):
            key = event["repeat_key"]
            counts[key] = counts.get(key, 0) + 1
    return counts


def _component_report(components: dict[str, dict]) -> list[dict]:
    return [components[name] for name in sorted(components)]


def _finished_status(components: dict[str, dict]) -> str:
    deployed = [item for item in components.values() if item["deployed"]]
    if not components:
        return "FAILED"
    if len(deployed) == len(components):
        return "SUCCESS"
    return "PARTIAL_SUCCESS" if deployed else "FAILED"


def run_deployment_agent(
    task_path: Path,
    output: Path,
    *,
    model: str = "gpt-5.6-terra",
    execute: bool = False,
    resume: bool = False,
    decide=None,
) -> dict:
    started = time.monotonic()
    task_path = task_path.resolve()
    task = DeploymentTask.model_validate(load_yaml(task_path))
    if decide is None and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required before acquiring deployment sources")
    if task.environment.kubeconfig:
        task.environment.kubeconfig = (task_path.parent / task.environment.kubeconfig).resolve()
        if not task.environment.kubeconfig.is_file():
            raise ValueError("environment.kubeconfig does not exist")
    if resume:
        if not (output / "events.jsonl").is_file():
            raise ValueError("--resume requires an existing run with events.jsonl")
        previous = DeploymentTask.model_validate(json.loads((output / "task.json").read_text()))
        task.inputs = {**previous.inputs, **task.inputs}
    else:
        output.mkdir(parents=True, exist_ok=False)
    write_json(output / "task.json", task)

    acquired = DeploymentSources.acquire(
        task,
        base=task_path.parent,
        cache_dir=output.parent / ".source-cache",
        workspace=output / "workspace",
    )
    write_json(output / "sources.json", acquired.manifest)
    history = RunHistory(output)
    if not resume:
        history.append("sources_acquired", sources=acquired.manifest.model_dump(mode="json"))
    sources = acquired.tools
    client = LLMClient(model=model)
    components, latest_feedback, start_iteration = _restored_state(history)
    (output / "run_notes.md").write_text(history.render_notes(), encoding="utf-8")
    repeats = _repeat_counts(history)
    inputs = input_state(task, task_path.parent)
    status, reason = "BUDGET_EXHAUSTED", "request budget exhausted"
    iteration = start_iteration

    for iteration in range(start_iteration + 1, task.budgets.max_cycles + 1):
        elapsed = time.monotonic() - started
        remaining_seconds = task.budgets.max_runtime_minutes * 60 - elapsed
        if remaining_seconds <= 0:
            reason = "runtime budget exhausted"
            break
        task_context = {
            "objective": task.task.objective,
            "invariants": task.task.invariants,
            "target": task.environment.model_dump(exclude={"kubeconfig"}),
            "constraints": task.constraints.model_dump(),
            "available_inputs": inputs,
            "sources": list(sources.sources),
            "workspace": str(output / "workspace"),
        }
        requests_left = task.budgets.max_cycles - iteration + 1
        notes = history.notes()
        note_index = history.note_index()
        note_index.pop("current-step", None)
        note_index.pop("progress-summary", None)
        payload = {
            "task": task_context,
            "current_step": notes.get("current-step", {}).get(
                "text", "Inspect source guidance and the target to choose the next step."
            ),
            "progress_summary": notes.get("progress-summary", {}).get("text", ""),
            "recorded_components": {
                name: component["deployed"] for name, component in components.items()
            },
            "run_notes": {
                "path": str(output / "run_notes.md"),
                "index": note_index,
            },
            "recent_history": history.recent_decisions(),
            "remaining_budget": {
                "requests": requests_left,
                "seconds": int(remaining_seconds),
            },
        }
        if requests_left == 2:
            payload["finalization"] = (
                "Record the current outcome now. Do not run another shell command."
            )
        elif requests_left == 1:
            payload["finalization"] = (
                "Final request: finish if the current outcome is recorded; otherwise record it. "
                "Do not run another shell command."
            )
        try:
            decision, call_usage = (
                decide(payload, iteration)
                if decide
                else client.complete(Decision, instructions=_INSTRUCTIONS, payload=payload)
            )
        except Exception as exc:
            status, reason = "FAILED", f"decision failed: {exc}"
            break
        for note in decision.notes:
            note.text = redact_secrets(note.text)
        print(f"agent: {decision.action}: {decision.reason}", flush=True)
        history.append(
            "decision",
            iteration=iteration,
            value=decision.model_dump(),
            usage=call_usage,
        )
        current_notes = history.notes()
        for note in decision.notes:
            previous = current_notes.get(note.key)
            if previous and (previous["text"], previous["evidence_events"]) == (
                note.text,
                note.evidence_events,
            ):
                continue
            history.append(
                "note_recorded",
                iteration=iteration,
                key=note.key,
                text=note.text,
                evidence_events=note.evidence_events,
            )
        if decision.notes:
            (output / "run_notes.md").write_text(history.render_notes(), encoding="utf-8")

        if decision.action == "finish":
            status, reason = _finished_status(components), decision.reason
            break

        if decision.action == "ask":
            state = {"inputs": inputs}
            provided = prompt_for_inputs(task, state, decision)
            inputs = state["inputs"]
            missing = missing_inputs(state, decision.required_inputs)
            if provided:
                write_json(output / "task.json", task)
                history.append("human_input", iteration=iteration, inputs=provided)
            if missing:
                request = {
                    "status": "WAITING_FOR_INPUT",
                    "reason": decision.reason,
                    "required_inputs": [item.model_dump() for item in missing],
                }
                write_yaml(output / "human-request.yaml", request)
                status, reason, latest_feedback = "WAITING_FOR_INPUT", decision.reason, request
                history.append("input_result", iteration=iteration, result=request)
                break
            latest_feedback = {"status": "inputs_provided", "inputs": provided}
            history.append("input_result", iteration=iteration, result=latest_feedback)
            continue

        if decision.action == "record":
            valid_evidence = _successful_shell_events(history, decision.evidence_events)
            valid_checks = not decision.verification_events or _successful_shell_events(
                history, decision.verification_events
            )
            if decision.deployed and (not valid_evidence or not valid_checks):
                latest_feedback = {
                    "status": "record_rejected",
                    "reason": "deployed requires successful shell evidence and verification events",
                }
            else:
                component = {
                    "name": decision.component,
                    "deployed": decision.deployed,
                    "reason": decision.reason,
                    "evidence_events": decision.evidence_events,
                    "verification_events": decision.verification_events,
                    "human_action": decision.human_action,
                }
                components[decision.component or ""] = component
                history.append(
                    "component_recorded",
                    iteration=iteration,
                    component=decision.component,
                    deployed=decision.deployed,
                    reason=decision.reason,
                    evidence_events=decision.evidence_events,
                    verification_events=decision.verification_events,
                    human_action=decision.human_action,
                )
                latest_feedback = {"status": "recorded", **component}
            history.append("record_result", iteration=iteration, result=latest_feedback)
            continue

        stdin_value = None
        stdin_sensitive = False
        if decision.stdin_input:
            if not inputs.get(decision.stdin_input, {}).get("available"):
                latest_feedback = {
                    "status": "needs_revision",
                    "reason": f"stdin input is unavailable: {decision.stdin_input}",
                }
                history.append("shell_result", iteration=iteration, result=latest_feedback)
                continue
            stdin_value, stdin_sensitive = input_value(task, decision.stdin_input, task_path.parent)
        result = execute_command(
            decision,
            task,
            sources,
            output / "workspace",
            enabled=execute,
            timeout=max(1, min(1800, int(remaining_seconds))),
            stdin_value=stdin_value,
            stdin_sensitive=stdin_sensitive,
            log_path=output / "actions" / f"{iteration:04d}.log",
        )
        repeat_key = _repeat_key(decision, result)
        repeats[repeat_key] = repeats.get(repeat_key, 0) + 1
        if repeats[repeat_key] >= 3:
            result["repeat_limit"] = (
                "This command produced the same result three times. Change the state or approach, "
                "or record the affected component as not deployed and continue."
            )
        execution = result.get("execution", {})
        event = history.append(
            "shell_result",
            iteration=iteration,
            command=result.get("command", decision.command),
            working_directory=decision.working_directory,
            state_changing=result.get("state_changing", False),
            exit_code=execution.get("exit_code"),
            repeat_key=repeat_key,
            result=result,
        )
        latest_feedback = {**result, "event_sequence": event["sequence"]}

    workspace_diff = sources.workspace_diff(output / "workspace")
    if workspace_diff:
        (output / "workspace.diff").write_text(workspace_diff, encoding="utf-8")
    report = {
        "status": status,
        "reason": reason,
        "cycles": iteration,
        "duration_seconds": round(time.monotonic() - started, 2),
        "usage": history.usage(),
        "model": model,
        "components": _component_report(components),
        "run_notes": "run_notes.md",
        "latest_feedback": latest_feedback,
        **({"workspace_diff": "workspace.diff"} if workspace_diff else {}),
    }
    write_json(output / "final-report.json", report)
    history.append("stop", result=report)
    runbook = output / "runbook.sh"
    runbook.write_text(history.runbook(task), encoding="utf-8")
    runbook.chmod(0o755)
    return report
