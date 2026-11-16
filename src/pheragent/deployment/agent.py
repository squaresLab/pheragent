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
from .runtime import execute_command
from .serialization import load_yaml, write_json, write_yaml
from .sources import DeploymentSources
from .task import Decision, DeploymentTask

_INSTRUCTIONS = """You are a senior DevOps engineer progressively deploying a system.

Maintain one compact working memory. It is the deployment state you will receive on the next
turn. It must contain a rough high-level overview checklist, the current outcome, confirmed
source facts and exact commands, confirmed runtime facts, the current problem, blocked work,
and the next action. Update the overview when evidence changes your understanding. Do not mark
an outcome complete until runtime evidence verifies it. Before the first state-changing command,
inspect the supplied sources and create a rough overview. The overview can be incomplete.

Make one decision per turn:
- shell: run one ordinary argv command such as find, rg, cat, git, docker, kubectl, helm,
  terraform, ansible, make, or a repository script.
- ask: request one atomic external value. Use a stable input name and mark secrets sensitive.
- record: record one component or outcome as deployed or not deployed. A deployed record must
  cite the successful shell events that changed and verified it. A not-deployed record must
  state the exact human action needed.
- finish: use only after the overview has no remaining safe, independent deployment work.

Use source-id:relative/directory for working_directory. Sources are immutable; commands run in
a per-run copy. Prefer repository-defined entrypoints. Read complete high-signal files when
practical. Keep facts already learned in memory instead of rereading unchanged files. If runtime
behavior contradicts source, inspect the artifact actually running before rereading source.

A failed command is feedback. Diagnose it in the next turn, change the state or approach, and
continue. If one component cannot progress, record it as not deployed and continue independent
work. Do not stop merely because one component is blocked. Unknown commands are permitted;
destructive, host, cloud, target, namespace, secret, and public-boundary changes may require
human approval.

Return memory as a complete replacement, not an addition. Keep it concise and do not include
raw file contents, command output, secrets, or private chain-of-thought.
"""


def _compact(value, limit: int = 12_000):
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    if len(encoded) <= limit:
        return value
    return {
        "status": value.get("status") if isinstance(value, dict) else "truncated",
        "summary": encoded[: limit // 2] + "\n[... omitted ...]\n" + encoded[-limit // 2 :],
    }


def _restored_state(history: RunHistory) -> tuple[str, dict, dict, int, int]:
    memory = ""
    components: dict[str, dict] = {}
    latest_feedback: dict = {}
    iteration = actions = 0
    for event in history.events():
        iteration = max(iteration, event.get("iteration", 0))
        if event["event"] == "memory_updated":
            memory = event["memory"]
        elif event["event"] == "component_recorded":
            components[event["component"]] = {
                "name": event["component"],
                "deployed": event["deployed"],
                "reason": event["reason"],
                "evidence_events": event.get("evidence_events", []),
                "human_action": event.get("human_action"),
            }
        elif event["event"] in {"shell_result", "input_result", "record_result"}:
            latest_feedback = event.get("result", {})
            if event["event"] == "shell_result":
                latest_feedback = {**latest_feedback, "event_sequence": event["sequence"]}
            if event["event"] == "shell_result" and event.get("state_changing"):
                actions += 1
    return memory, components, latest_feedback, iteration, actions


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
    )
    write_json(output / "sources.json", acquired.manifest)
    history = RunHistory(output)
    if not resume:
        history.append("sources_acquired", sources=acquired.manifest.model_dump(mode="json"))
    sources = acquired.tools
    client = LLMClient(model=model)
    memory, components, latest_feedback, start_iteration, actions = _restored_state(history)
    repeats = _repeat_counts(history)
    inputs = input_state(task, task_path.parent)
    usage = history.usage()
    status, reason = "BUDGET_EXHAUSTED", "request budget exhausted"
    iteration = start_iteration

    for iteration in range(start_iteration + 1, task.budgets.max_cycles + 1):
        elapsed = time.monotonic() - started
        remaining_seconds = task.budgets.max_runtime_minutes * 60 - elapsed
        if remaining_seconds <= 0:
            reason = "runtime budget exhausted"
            break
        payload = {
            "task": {
                "objective": task.task.objective,
                "target": task.environment.model_dump(exclude={"kubeconfig"}),
                "constraints": task.constraints.model_dump(),
                "available_inputs": inputs,
                "sources": list(sources.sources),
            },
            "memory": memory or "No overview exists yet. Inspect the sources and create one.",
            "latest_feedback": _compact(latest_feedback),
            "remaining_budget": {
                "requests": task.budgets.max_cycles - iteration + 1,
                "seconds": int(remaining_seconds),
            },
        }
        try:
            decision, call_usage = (
                decide(payload, iteration)
                if decide
                else client.complete(Decision, instructions=_INSTRUCTIONS, payload=payload)
            )
        except Exception as exc:
            status, reason = "FAILED", f"decision failed: {exc}"
            break
        for key, value in call_usage.items():
            usage[key] = usage.get(key, 0) + value
        print(f"agent: {decision.action}: {decision.reason}", flush=True)
        history.append(
            "decision",
            iteration=iteration,
            value=decision.model_dump(),
            usage=call_usage,
        )
        if decision.memory != memory:
            memory = decision.memory
            history.append("memory_updated", iteration=iteration, memory=memory)
            (output / "memory.md").write_text(memory + "\n", encoding="utf-8")

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
            if decision.deployed and not valid_evidence:
                latest_feedback = {
                    "status": "record_rejected",
                    "reason": "deployed requires a successful shell evidence event",
                }
            else:
                component = {
                    "name": decision.component,
                    "deployed": decision.deployed,
                    "reason": decision.reason,
                    "evidence_events": decision.evidence_events,
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
        if result.get("state_changing") and execution:
            actions += 1
        latest_feedback = {**result, "event_sequence": event["sequence"]}

    workspace_diff = sources.workspace_diff(output / "workspace")
    if workspace_diff:
        (output / "workspace.diff").write_text(workspace_diff, encoding="utf-8")
    report = {
        "status": status,
        "reason": reason,
        "cycles": iteration,
        "mutating_actions": actions,
        "duration_seconds": round(time.monotonic() - started, 2),
        "usage": usage,
        "model": model,
        "components": _component_report(components),
        "memory": memory,
        "latest_feedback": latest_feedback,
        **({"workspace_diff": "workspace.diff"} if workspace_diff else {}),
    }
    write_json(output / "final-report.json", report)
    history.append("stop", result=report)
    runbook = output / "runbook.sh"
    runbook.write_text(history.runbook(), encoding="utf-8")
    runbook.chmod(0o755)
    return report
