"""Resolve deployment inputs and terminal responses."""

from __future__ import annotations

import os
import sys
from getpass import getpass
from pathlib import Path

from .task import Decision, DeploymentTask, InputRequest, TaskInput


def input_state(task: DeploymentTask, base: Path) -> dict[str, dict]:
    result = {}
    for name, item in task.inputs.items():
        if item.value is not None:
            available, source = True, "task"
        elif item.from_env is not None:
            available, source = bool(os.getenv(item.from_env)), f"env:{item.from_env}"
        else:
            path = item.from_file or Path()
            path = path if path.is_absolute() else base / path
            available, source = path.is_file(), f"file:{path}"
        result[name] = {
            "available": available,
            "sensitive": item.sensitive,
            "source": source,
        }
        if item.value is not None and not item.sensitive:
            result[name]["value"] = item.value
    return result


def missing_input_names(state: dict, names: list[str]) -> list[str]:
    return sorted(name for name in names if not state["inputs"].get(name, {}).get("available"))


def missing_inputs(state: dict, requests: list[InputRequest]) -> list[InputRequest]:
    missing = set(missing_input_names(state, [request.name for request in requests]))
    return [request for request in requests if request.name in missing]


def input_value(task: DeploymentTask, name: str, base: Path) -> tuple[str, bool]:
    item = task.inputs[name]
    if item.value is not None:
        value = item.value
    elif item.from_env is not None:
        value = os.environ[item.from_env]
    else:
        path = item.from_file or Path()
        path = path if path.is_absolute() else base / path
        value = path.read_text(encoding="utf-8")
    return value if value.endswith("\n") else value + "\n", item.sensitive


def prompt_for_inputs(
    task: DeploymentTask,
    state: dict,
    decision: Decision,
) -> list[str]:
    if not sys.stdin.isatty():
        return []
    provided = []
    print(decision.reason, flush=True)
    for request in missing_inputs(state, decision.required_inputs):
        name = request.name
        configured = task.inputs.get(name)
        sensitive = request.sensitive or bool(configured and configured.sensitive)
        try:
            prompt = f"{request.prompt} [{name}]"
            value = getpass(f"{prompt} (hidden): ") if sensitive else input(f"{prompt}: ").strip()
        except (EOFError, KeyboardInterrupt):
            value = ""
        if not value:
            continue
        if sensitive:
            variable = configured.from_env if configured and configured.from_env else name
            os.environ[variable] = value
            task.inputs[name] = TaskInput(from_env=variable, sensitive=True)
            state["inputs"][name] = {
                "available": True,
                "sensitive": True,
                "source": f"env:{variable}",
            }
        else:
            task.inputs[name] = TaskInput(value=value)
            state["inputs"][name] = {
                "available": True,
                "sensitive": False,
                "source": "interactive",
                "value": value,
            }
        provided.append(name)
    return provided


def choose_option(reason: str, options: list[str]) -> str | None:
    if not sys.stdin.isatty():
        return None
    print(f"{reason}\nChoose a route:", flush=True)
    for index, option in enumerate(options, 1):
        print(f"  {index}. {option}", flush=True)
    try:
        answer = input("Choice [number, or n to stop]: ").strip()
    except EOFError:
        return None
    if answer not in {str(index) for index in range(1, len(options) + 1)}:
        return None
    return options[int(answer) - 1]


def confirm_source(location: str, evidence: list[str], reason: str) -> bool:
    print(
        "Approve deployment source?\n"
        f"Source: {location}\n"
        f"Referenced by: {', '.join(evidence)}\n"
        f"Reason: {reason}\n"
        "Access: read-only clone; commands still require separate approval\n"
        "[y/N] ",
        end="",
        flush=True,
    )
    try:
        return input().strip().casefold() == "y"
    except EOFError:
        return False
