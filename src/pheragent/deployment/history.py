"""Append-only deployment history and artifacts derived from it."""

from __future__ import annotations

import json
import os
import re
import shlex
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .redaction import redact_secrets
from .task import DeploymentTask, TaskInput
from .telemetry import record_event


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _preview_output(output: str, reference: str) -> str:
    if len(output) <= 1_200:
        return output
    lines = output.splitlines(keepends=True)
    if len(lines) > 7:
        head = "".join(lines[:2])[:400]
        tail = "".join(lines[-5:])[-800:]
        omitted_lines = len(lines) - 7
    else:
        head, tail = output[:400], output[-800:]
        omitted_lines = 0
    omitted_characters = len(output) - len(head) - len(tail)
    return (
        head + f"\n[{omitted_lines} complete lines and {omitted_characters} characters "
        f"omitted from history preview; read {reference}]\n" + tail
    )


def _comment(text: str) -> list[str]:
    return [f"# {line}" for line in redact_secrets(text).splitlines()]


def _source_location(location: str) -> str:
    parsed = urlsplit(location)
    if "@" in parsed.netloc:
        return urlunsplit(parsed._replace(netloc=parsed.netloc.rsplit("@", 1)[1]))
    return location


def _runbook_command(
    event: dict, run_dir: Path, stdin_name: str | None, inputs: dict[str, TaskInput]
) -> str | None:
    directory = event.get("working_directory") or "."
    if Path(directory).is_absolute() or ".." in Path(directory).parts:
        return None
    command = event.get("command") or []
    if not command or any(str(run_dir) in part or "[REDACTED]" in part for part in command):
        return None
    prefix = ""
    if stdin_name:
        item = inputs.get(stdin_name)
        if item and item.from_file:
            prefix = f'cat "${stdin_name}_FILE" | '
        else:
            variable = item.from_env if item and item.from_env else stdin_name
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable):
                return None
            prefix = f'printf "%s\\n" "${variable}" | '
    line = f'(cd "$WORKSPACE"/{shlex.quote(directory)} && {prefix}{shlex.join(command)})'
    return line if redact_secrets(line) == line else None


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

    def recent_decisions(self, limit: int = 10) -> list[dict]:
        """Pair recent decisions with their recorded results."""
        turns: dict[int, dict] = {}
        for event in self._events():
            iteration = event.get("iteration")
            if event["event"] == "decision":
                turns[iteration] = {
                    "iteration": iteration,
                    "decision_event": event["sequence"],
                    "decision": {
                        key: value
                        for key, value in event["value"].items()
                        if key != "notes" and value is not None and value != []
                    },
                    "result": {"status": "not_recorded"},
                }
            elif (
                event["event"] in {"shell_result", "input_result", "record_result"}
                and iteration in turns
            ):
                turns[iteration]["result"] = {
                    "event_sequence": event["sequence"],
                    **event["result"],
                }
        recent = list(turns.values())[-limit:]
        seen: dict[tuple, int] = {}
        for turn in recent[:-1]:
            result = turn["result"]
            execution = result.get("execution")
            if not execution:
                continue
            output = execution.get("stdout", "")
            key = (
                tuple(turn["decision"].get("command") or []),
                turn["decision"].get("working_directory"),
                result.get("status"),
                execution.get("exit_code"),
                output,
                execution.get("stderr", ""),
            )
            if output and key in seen:
                execution["stdout"] = f"[same output as result event {seen[key]}]"
            else:
                seen[key] = result["event_sequence"]
                reference = execution.get("output_log") or (
                    f"events.jsonl event {result['event_sequence']}"
                )
                execution["stdout"] = _preview_output(output, reference)
        return recent

    def usage(self) -> dict[str, int]:
        total: dict[str, int] = {}
        for event in self._events():
            for key, value in event.get("usage", {}).items():
                total[key] = total.get(key, 0) + value
        return total

    def notes(self) -> dict[str, dict]:
        latest: dict[str, dict] = {}
        for event in self._events():
            if event["event"] == "note_recorded":
                latest.pop(event["key"], None)
                latest[event["key"]] = event
        return latest

    def note_index(self) -> dict[str, str]:
        return {key: note["text"][:120] for key, note in list(self.notes().items())[-40:]}

    def render_notes(self) -> str:
        lines = ["# Run notes", ""]
        for note in self.notes().values():
            lines.extend([f"## {note['key']}", "", note["text"], ""])
        return "\n".join(lines)

    def runbook(self, task: DeploymentTask) -> str:
        events = self.events()
        shell_events = {
            event["sequence"]: event for event in events if event["event"] == "shell_result"
        }
        decisions = {
            event["iteration"]: event["value"]
            for event in events
            if event["event"] == "decision" and "iteration" in event
        }
        inputs = task.inputs
        notes = self.notes()
        gaps: list[str] = []
        lines = [
            "#!/bin/sh",
            "set -eu",
            'RUN_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)',
            'WORKSPACE=${RUNBOOK_WORKSPACE:-"$RUN_DIR/workspace"}',
            "# Candidate runbook: replay on a clean target before relying on it.",
            "",
            "# Prerequisites",
            "# Provide a clean target and the pinned sources under WORKSPACE.",
        ]
        if task.environment.type == "kubernetes":
            lines.extend(
                [
                    "# HUMAN INPUT KUBECONFIG: credentials for the declared Kubernetes target.",
                    ': "${KUBECONFIG:?Set KUBECONFIG for the declared target}"',
                    'test -r "$KUBECONFIG"',
                    f'test "$(kubectl config current-context)" = '
                    f"{shlex.quote(task.environment.context or '')}",
                ]
            )
        source_event = next((e for e in events if e["event"] == "sources_acquired"), None)
        if not source_event:
            gaps.append("No pinned source manifest was recorded")
        for source in source_event["sources"]["sources"] if source_event else []:
            revision = source.get("resolved_revision") or source["content_hash"]
            path = f'"$WORKSPACE"/{shlex.quote(source["id"])}'
            lines.extend(
                _comment(
                    f"Source {source['id']}: {_source_location(source['location'])} @ {revision}"
                )
            )
            lines.append(f"test -e {path}")
            if source.get("resolved_revision"):
                lines.append(
                    f'test "$(git -C {path} rev-parse HEAD)" = '
                    f"{shlex.quote(source['resolved_revision'])}"
                )
        prompts = {
            item["name"]: item["prompt"]
            for event in events
            if event["event"] == "decision" and event["value"].get("action") == "ask"
            for item in event["value"].get("required_inputs", [])
        }
        used_stdin = {
            decision["stdin_input"]
            for decision in decisions.values()
            if decision.get("stdin_input")
        }
        for name in sorted(set(inputs) | used_stdin):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                gaps.append(f"Input name cannot be used in a shell script: {name}")
                continue
            prompt = prompts.get(name, "Provide the value used in the original run.")
            lines.extend(_comment(f"HUMAN INPUT {name}: {prompt}"))
            item = inputs.get(name)
            if item and item.value is not None:
                value = f"export {name}={shlex.quote(item.value)}"
                if redact_secrets(value) != value:
                    gaps.append(f"Input {name} may contain a secret; provide it at replay time")
                else:
                    lines.append(value)
            elif item and item.from_file:
                lines.extend(
                    [
                        f': "${{{name}_FILE:?Set {name}_FILE to a readable input file}}"',
                        f'test -r "${name}_FILE"',
                    ]
                )
            else:
                variable = item.from_env if item and item.from_env else name
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", variable):
                    gaps.append(f"Invalid environment variable for {name}")
                    continue
                lines.append(f': "${{{variable}:?Provide {name}}}"')
                lines.append(f"export {variable}")

        def add_command(sequence: int, label: str) -> None:
            event = shell_events.get(sequence)
            if not event or event.get("exit_code") != 0:
                gaps.append(f"{label} has no successful command at event {sequence}")
                return
            stdin_name = decisions.get(event.get("iteration"), {}).get("stdin_input")
            command = _runbook_command(event, self.path.parent, stdin_name, inputs)
            if command is None:
                gaps.append(f"{label} command at event {sequence} is not portable")
            else:
                if event.get("result", {}).get("policy", "").startswith("high_risk_approval"):
                    lines.extend(
                        [
                            *_comment(f"HUMAN APPROVAL required for {label} (event {sequence})"),
                            "printf 'Type approve to continue: ' >&2",
                            "IFS= read -r RUNBOOK_APPROVAL </dev/tty || exit 1",
                            'test "$RUNBOOK_APPROVAL" = approve',
                        ]
                    )
                lines.append(command)

        for key, note in notes.items():
            if key.startswith("prerequisite-"):
                lines.extend(_comment(f"Required condition: {note['text']}"))
                if not note["evidence_events"]:
                    gaps.append(f"{key} has no recorded check")
                for sequence in note["evidence_events"]:
                    if shell_events.get(sequence, {}).get("state_changing"):
                        gaps.append(f"{key} preflight check changes state")
                    else:
                        add_command(sequence, key)

        lines.extend(["", "# Deployment and component checks"])
        components: dict[str, dict] = {}
        for event in events:
            if event["event"] == "component_recorded":
                components.pop(event["component"], None)
                components[event["component"]] = event
        emitted_actions: set[int] = set()
        previous_record = 0
        for component in components.values():
            lines.extend(["", *_comment(f"Component: {component['component']}")])
            if not component["deployed"]:
                reason = component.get("human_action") or component.get("reason", "")
                lines.extend(_comment(f"Not deployed: {reason}"))
                previous_record = component["sequence"]
                continue
            evidence = set(component.get("evidence_events", []))
            uncited = [
                event["sequence"]
                for event in shell_events.values()
                if previous_record < event["sequence"] < component["sequence"]
                and event.get("exit_code") == 0
                and event.get("state_changing")
                and event["sequence"] not in evidence
            ]
            if uncited:
                gaps.append(f"{component['component']} omits successful changes {uncited}")
            checks = set(component.get("verification_events", []))
            actions = sorted(evidence - checks)
            if not checks <= evidence:
                gaps.append(f"{component['component']} has verification outside its evidence")
            if actions and checks and min(checks) <= max(actions):
                gaps.append(f"{component['component']} check predates its last action")
            for sequence in actions:
                event = shell_events.get(sequence)
                if not event or event.get("exit_code") != 0:
                    gaps.append(f"{component['component']} has invalid action event {sequence}")
                    continue
                if not event.get("state_changing") or sequence in emitted_actions:
                    continue
                emitted_actions.add(sequence)
                lines.extend(
                    _comment(
                        decisions.get(event.get("iteration"), {}).get(
                            "reason", "Apply recorded change"
                        )
                    )
                )
                add_command(sequence, component["component"])
            if not checks:
                gaps.append(f"{component['component']} has no recorded component check")
            for sequence in sorted(checks):
                lines.extend(_comment(f"Verify {component['component']} (event {sequence})"))
                if shell_events.get(sequence, {}).get("state_changing"):
                    gaps.append(f"{component['component']} check changes state")
                add_command(sequence, component["component"])
            previous_record = component["sequence"]

        lines.extend(["", "# Post-deployment system checks"])
        invariants = dict(task.task.invariants)
        for key, note in notes.items():
            if key.startswith("invariant-"):
                invariants.setdefault(key.removeprefix("invariant-"), note["text"])
        if not invariants:
            gaps.append("No end-to-end system check was recorded")
        for key, description in invariants.items():
            lines.extend(_comment(f"System invariant {key}: {description}"))
            note = notes.get(f"invariant-{key}")
            if not note or not note["evidence_events"]:
                gaps.append(f"System invariant {key} has no recorded check")
                continue
            for sequence in note["evidence_events"]:
                if shell_events.get(sequence, {}).get("state_changing"):
                    gaps.append(f"System invariant {key} check changes state")
                add_command(sequence, f"system invariant {key}")
        if gaps:
            warning = [line for gap in gaps for line in _comment(f"INCOMPLETE: {gap}")]
            lines[4:4] = warning + ["exit 1  # Resolve runbook gaps before replaying changes."]
        return "\n".join(lines) + "\n"

    def _events(self):
        if not self.path.is_file():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
