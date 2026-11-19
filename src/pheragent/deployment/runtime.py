"""Run one model-selected command inside the declared deployment boundary."""

from __future__ import annotations

import os
import queue
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path

from .redaction import redact_secrets
from .sources import SourceWorkspace
from .task import Decision, DeploymentTask

_READ_COMMANDS = {
    "cat",
    "df",
    "du",
    "env",
    "find",
    "free",
    "grep",
    "head",
    "id",
    "ls",
    "lsblk",
    "lscpu",
    "nl",
    "printenv",
    "ps",
    "pwd",
    "rg",
    "stat",
    "tail",
    "test",
    "uname",
    "wc",
    "whoami",
}
_READ_VERBS = {
    "check",
    "config",
    "describe",
    "diff",
    "events",
    "explain",
    "get",
    "history",
    "images",
    "info",
    "inspect",
    "list",
    "logs",
    "output",
    "plan",
    "ps",
    "show",
    "status",
    "template",
    "top",
    "validate",
    "version",
}
_DESTRUCTIVE = {
    "cleanup",
    "delete",
    "destroy",
    "disable",
    "drain",
    "drop",
    "erase",
    "kill",
    "mkfs",
    "prune",
    "purge",
    "reboot",
    "remove",
    "reset",
    "rm",
    "rollback",
    "shutdown",
    "stop",
    "truncate",
    "uninstall",
    "wipe",
}


def _run(command: list[str], task: DeploymentTask, *, timeout: int = 30) -> dict:
    env = os.environ.copy()
    if task.environment.kubeconfig:
        env["KUBECONFIG"] = str(task.environment.kubeconfig.resolve())
    try:
        result = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "exit_code": result.returncode,
            "stdout": redact_secrets(result.stdout),
            "stderr": redact_secrets(result.stderr),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"exit_code": None, "stdout": "", "stderr": redact_secrets(str(exc))}


def _stream_command(
    command: list[str],
    task: DeploymentTask,
    *,
    cwd: Path | None,
    timeout: int,
    stdin_value: str | None,
    stdin_sensitive: bool,
    log_path: Path,
    output_limit: int = 8_000,
) -> dict:
    env = os.environ.copy()
    if task.environment.kubeconfig:
        env["KUBECONFIG"] = str(task.environment.kubeconfig.resolve())
    secret = stdin_value.rstrip("\n") if stdin_sensitive and stdin_value else ""

    def safe(text: str) -> str:
        text = redact_secrets(text)
        return text.replace(secret, "[REDACTED INPUT]") if secret else text

    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE if stdin_value is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
    except OSError as exc:
        return {
            "exit_code": None,
            "stdout": "",
            "stderr": safe(str(exc)),
            "duration_seconds": round(time.monotonic() - started, 2),
            "timed_out": False,
        }
    if process.stdin:
        process.stdin.write(stdin_value or "")
        process.stdin.close()

    chunks: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            chunks.put(line)
        chunks.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    head = tail = ""
    output_characters = output_lines = 0
    finished = timed_out = False
    with log_path.open("a", encoding="utf-8") as log:
        while not finished:
            try:
                chunk = chunks.get(timeout=0.1)
                if chunk is None:
                    finished = True
                    continue
                chunk = safe(chunk)
                print(chunk, end="", flush=True)
                log.write(chunk)
                log.flush()
                output_characters += len(chunk)
                output_lines += 1
                head = (head + chunk)[: output_limit // 2]
                tail = (tail + chunk)[-output_limit:]
            except queue.Empty:
                pass
            if process.poll() is None and time.monotonic() - started >= timeout:
                timed_out = True
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    process.wait()
    truncated = output_characters > output_limit
    if truncated:
        end = output_characters - output_limit // 2
        tail = (
            head
            + f"\n[output characters {len(head) + 1}-{end} omitted; full output: {log_path}]\n"
            + tail[-output_limit // 2 :]
        )
    return {
        "exit_code": None if timed_out else process.returncode,
        "stdout": tail,
        "stderr": f"command timed out after {timeout} seconds" if timed_out else "",
        "duration_seconds": round(time.monotonic() - started, 2),
        "timed_out": timed_out,
        "output_log": str(log_path),
        "output_truncated": truncated,
        "output_characters": output_characters,
        "output_lines": output_lines,
    }


def _without_target_context(command: list[str], task: DeploymentTask) -> list[str]:
    if task.environment.type != "kubernetes" or command[0] not in {"kubectl", "helm"}:
        return command
    flag = "--context" if command[0] == "kubectl" else "--kube-context"
    arguments = command[1:]
    matches = [
        index for index, part in enumerate(arguments) if part == flag or part.startswith(flag + "=")
    ]
    if len(matches) > 1:
        raise ValueError(f"specify {flag} only once")
    if not matches:
        return command
    index = matches[0]
    supplied = (
        arguments[index + 1]
        if arguments[index] == flag and index + 1 < len(arguments)
        else arguments[index].partition("=")[2]
    )
    if supplied != task.environment.context:
        raise ValueError(f"{flag} must match the declared target context")
    width = 2 if arguments[index] == flag else 1
    return [command[0], *arguments[:index], *arguments[index + width :]]


def target_command(command: list[str], task: DeploymentTask) -> list[str]:
    command = _without_target_context(command, task)
    if task.environment.type == "kubernetes" and command[0] in {"kubectl", "helm"}:
        flag = "--context" if command[0] == "kubectl" else "--kube-context"
        return [command[0], flag, task.environment.context or "", *command[1:]]
    return command


def _read_only(command: list[str]) -> bool:
    executable = Path(command[0]).name
    if executable == "sed":
        return (
            len(command) == 4
            and command[1] == "-n"
            and re.fullmatch(r"\d+(?:,\d+)?p", command[2]) is not None
        )
    if executable in _READ_COMMANDS:
        return True
    words = [part.casefold().replace("_", "-") for part in command[1:] if not part.startswith("-")]
    if executable == "git":
        return bool(words and words[0] in {"diff", "log", "rev-parse", "show", "status"})
    if executable == "docker" and words[:1] == ["compose"]:
        words = words[1:]
    if executable in {"aws", "az", "gcloud"}:
        return any(
            word in _READ_VERBS
            or word.startswith(("describe-", "get-", "list-"))
            for word in words[:3]
        )
    if executable == "terraform" and words[:1] == ["state"]:
        return len(words) > 1 and words[1] in {"list", "show"}
    return bool(words and words[0] in _READ_VERBS)


def _risk(command: list[str]) -> str | None:
    executable = Path(command[0]).name.casefold()
    tokens = {
        token.replace("_", "-")
        for token in re.findall(r"[a-z0-9_-]+", " ".join(command).casefold())
    }
    if tokens & _DESTRUCTIVE:
        return "potentially destructive operation"
    if tokens & {"dns", "expose", "ingress", "loadbalancer", "nodeport", "public"}:
        return "public endpoint change"
    host_commands = {
        "sudo", "apt", "apt-get", "dnf", "yum", "zypper", "snap", "systemctl", "service"
    }
    if executable in host_commands or tokens & host_commands:
        return "host-level change"
    cloud_commands = {"aws", "az", "gcloud", "terraform"}
    if executable in cloud_commands or tokens & cloud_commands:
        return "cloud or infrastructure change"
    system_path = any(
        part.startswith(("/boot/", "/dev/", "/etc/", "/opt/", "/usr/", "/var/"))
        for part in command[1:]
    )
    file_commands = {
        "chmod", "chown", "cp", "install", "ln", "mkdir", "mv", "sed", "tee", "touch"
    }
    if executable in file_commands and system_path:
        return "host-level change"
    return None


def _exposes_secret(command: list[str]) -> bool:
    return Path(command[0]).name in {"env", "printenv"} or (
        command[0] == "kubectl"
        and {part.casefold() for part in command} & {"secret", "secrets"}
        and any(part.startswith(("json", "yaml", "go-template")) for part in command)
    )


def classify_command(command: list[str], task: DeploymentTask, *, mutating: bool = True) -> str:
    if not command or any(not item or "\x00" in item for item in command):
        return "denied: command must be a nonempty argv list"
    try:
        normalized = _without_target_context(command, task)
    except ValueError as exc:
        return f"denied: {exc}"
    if normalized[:2] == ["kubectl", "config"] and normalized[2:3] not in (
        ["current-context"],
        ["get-contexts"],
    ):
        return "high_risk_approval: Kubernetes target change"
    if _exposes_secret(normalized):
        return "high_risk_approval: command can expose secrets"
    if _read_only(normalized):
        return "allowed"
    if mutating and normalized[0] in {"kubectl", "helm"}:
        namespace = next(
            (
                normalized[index + 1]
                for index, part in enumerate(normalized[:-1])
                if part in {"-n", "--namespace"}
            ),
            None,
        )
        allowed = task.constraints.allowed_namespaces
        if allowed and "*" not in allowed and namespace not in allowed:
            return "high_risk_approval: namespace outside declared scope or not explicit"
        if normalized[:3] == ["kubectl", "create", "namespace"]:
            return "high_risk_approval: namespace boundary change"
    allowed = task.constraints.allowed_namespaces
    opaque_kubernetes = (
        task.environment.type == "kubernetes"
        and allowed
        and "*" not in allowed
        and normalized[0] not in {"kubectl", "helm"}
        and not _read_only(normalized)
    )
    if opaque_kubernetes:
        return "high_risk_approval: command can cross the declared namespace boundary"
    risk = _risk(normalized)
    return f"high_risk_approval: {risk}" if risk else "allowed"


def execute_command(
    decision: Decision,
    task: DeploymentTask,
    sources: SourceWorkspace,
    workspace: Path,
    *,
    enabled: bool,
    timeout: int,
    stdin_value: str | None = None,
    stdin_sensitive: bool = False,
    log_path: Path,
) -> dict:
    policy = classify_command(decision.command, task)
    if policy.startswith("denied"):
        return {"status": "policy_denied", "reason": policy}
    if not enabled and not _read_only(decision.command):
        return {"status": "dry_run", "reason": "pass --execute", "policy": policy}
    if policy.startswith("high_risk_approval"):
        if not sys.stdin.isatty():
            return {"status": "approval_required", "reason": policy}
        print(
            f"Approve HIGH-RISK action on {task.environment.context or task.environment.type}?\n"
            f"Command: {shlex.join(target_command(decision.command, task))}\n"
            f"Reason: {decision.reason}\nPolicy: {policy}\n[type approve/N] ",
            end="",
            flush=True,
        )
        if input().strip().casefold() != "approve":
            return {"status": "declined", "reason": "human declined action"}
    if task.environment.type == "kubernetes" and decision.command[0] not in {"kubectl", "helm"}:
        current = _run(["kubectl", "config", "current-context"], task)
        if current["exit_code"] != 0 or current["stdout"].strip() != task.environment.context:
            return {
                "status": "policy_denied",
                "reason": "source command would use a different Kubernetes context",
            }
    try:
        cwd = sources.working_directory(decision.working_directory, workspace)
        command = target_command(decision.command, task)
    except ValueError as exc:
        return {"status": "needs_revision", "reason": str(exc)}
    execution = _stream_command(
        command,
        task,
        cwd=cwd,
        timeout=timeout,
        stdin_value=stdin_value,
        stdin_sensitive=stdin_sensitive,
        log_path=log_path,
    )
    return {
        "status": "succeeded" if execution["exit_code"] == 0 else "command_failed",
        "policy": policy,
        "state_changing": not _read_only(decision.command),
        "command": command,
        "execution": execution,
    }
