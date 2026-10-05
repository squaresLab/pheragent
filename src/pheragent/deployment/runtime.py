"""Observe and change the declared deployment target safely."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .redaction import redact_secrets
from .sources import SourceTools, _source_cwd, _source_grounded
from .task import Check, Decision, DeploymentTask

_READ_KUBECTL = {
    "get",
    "describe",
    "logs",
    "cluster-info",
    "api-resources",
    "version",
    "rollout",
    "wait",
    "explain",
    "top",
}
_READ_HELM = {"list", "status", "show", "get", "version", "history", "env"}
_READ_AWS = {
    ("sts", "get-caller-identity"),
    ("ec2", "describe-instances"),
    ("ec2", "describe-instance-status"),
    ("ec2", "describe-volumes"),
    ("ec2", "describe-vpcs"),
    ("ec2", "describe-subnets"),
    ("ec2", "describe-security-groups"),
    ("ec2", "describe-route-tables"),
    ("eks", "list-clusters"),
    ("eks", "describe-cluster"),
    ("eks", "list-nodegroups"),
    ("eks", "describe-nodegroup"),
}
_DENIED = {"rm", "sudo", "shutdown", "reboot", "mkfs", "dd", "terraform", "az", "gcloud"}


def _command(
    command: list[str],
    task: DeploymentTask,
    *,
    cwd: Path | None = None,
    timeout: int = 120,
    output_limit: int | None = 20000,
) -> dict:
    if not command or any(not item or "\x00" in item for item in command):
        raise ValueError("command must be a nonempty argv list")
    env = os.environ.copy()
    if task.environment.kubeconfig:
        env["KUBECONFIG"] = str(task.environment.kubeconfig.resolve())
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "exit_code": result.returncode,
            "stdout": redact_secrets(result.stdout[:output_limit]),
            "stderr": redact_secrets(result.stderr[:output_limit]),
            "duration_seconds": round(time.monotonic() - started, 2),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "exit_code": None,
            "stdout": "",
            "stderr": redact_secrets(str(exc)),
            "duration_seconds": round(time.monotonic() - started, 2),
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
    if matches:
        index = matches[0]
        supplied = (
            arguments[index + 1]
            if arguments[index] == flag and index + 1 < len(arguments)
            else arguments[index].partition("=")[2]
        )
        if supplied != task.environment.context:
            raise ValueError(f"{flag} must match the declared target context")
        width = 2 if arguments[index] == flag else 1
        arguments = arguments[:index] + arguments[index + width :]
    return [command[0], *arguments]


def _target_command(command: list[str], task: DeploymentTask) -> list[str]:
    command = _without_target_context(command, task)
    if task.environment.type == "kubernetes" and command[0] in {"kubectl", "helm"}:
        flag = "--context" if command[0] == "kubectl" else "--kube-context"
        return [command[0], flag, task.environment.context or "", *command[1:]]
    return command


def _read_only(command: list[str]) -> bool:
    if not command:
        return False
    if command[0] == "kubectl":
        verb = command[1] if len(command) > 1 else ""
        if verb == "auth":
            return command[2:3] == ["can-i"]
        if verb == "config":
            return command[2:3] in (["current-context"], ["get-contexts"])
        return verb in _READ_KUBECTL and not (verb == "rollout" and "status" not in command)
    if command[0] == "helm":
        verb = command[1] if len(command) > 1 else ""
        return verb in _READ_HELM or command[1:3] in (
            ["repo", "list"],
            ["search", "repo"],
            ["search", "hub"],
            ["dependency", "list"],
        )
    if command[0] == "aws":
        return command[1:2] == ["--version"] or tuple(command[1:3]) in _READ_AWS
    if command[0] in {"ansible", "ansible-playbook"}:
        return command[1:] == ["--version"] or (
            command[0] == "ansible-playbook"
            and command[1:2] == ["--syntax-check"]
            and len(command) == 3
            and not command[2].startswith("-")
        )
    if command[0] in {"bash", "sh"}:
        return command[1:2] == ["-n"] and len(command) == 3 and not command[2].startswith("-")
    if command[0] == "git":
        return len(command) > 1 and command[1] in {"status", "log", "show", "rev-parse"}
    if command[0] == "docker":
        return len(command) > 1 and (
            command[1] in {"ps", "info", "version"}
            or command[1:] == ["compose", "version"]
        )
    return command[0] in {"ls", "pwd", "uname", "df", "free", "ps", "stat"}


def _policy(command: list[str], task: DeploymentTask, *, mutating: bool) -> str:
    if not command or Path(command[0]).name in _DENIED:
        return "denied: forbidden command"
    if Path(command[0]).name in {"sh", "bash", "zsh", "python", "python3", "perl", "ruby"} and any(
        part in {"-c", "-e"} for part in command[1:]
    ):
        return "denied: inline interpreter commands are not allowed"
    if any(
        any(mark in part for mark in (";", "&&", "||", "|", "$(", "`", ">", "<"))
        for part in command
    ):
        return "denied: shell operators are not allowed"
    try:
        command = _without_target_context(command, task)
    except ValueError as exc:
        return f"denied: {exc}"
    if Path(command[0]).name == "aws":
        if mutating:
            return "denied: AWS changes are outside this agent's scope"
        if any(
            part.split("=", 1)[0] in {"--profile", "--endpoint-url", "--no-verify-ssl", "--debug"}
            for part in command[1:]
        ):
            return "denied: AWS profile, endpoint, and debug overrides are not allowed"
    if mutating and _read_only(command):
        return "denied: execute must propose a mutating command"
    if not mutating and not _read_only(command):
        return "denied: observation and validation must be read-only"
    if mutating and task.environment.type == "kubernetes" and command[0] not in {"kubectl", "helm"}:
        return "approval_required: source script or host command can change the target"
    if command[0] == "kubectl" and any(
        part in {"delete", "replace", "patch", "drain"} for part in command
    ):
        return "denied: potentially destructive Kubernetes operation"
    if command[0] == "helm" and any(part in {"uninstall", "rollback"} for part in command):
        return "denied: potentially destructive Helm operation"
    if mutating and (
        command[0] == "kubectl" or command[:2] in (["helm", "install"], ["helm", "upgrade"])
    ):
        namespace = next(
            (
                command[i + 1]
                for i, part in enumerate(command[:-1])
                if part in {"-n", "--namespace"}
            ),
            None,
        )
        if (
            "*" not in task.constraints.allowed_namespaces
            and namespace not in task.constraints.allowed_namespaces
        ):
            return "denied: namespace outside allowed scope or not explicit"
    if not mutating or (task.environment.type == "shell" and task.environment.sandbox):
        return "allowed"
    return "approval_required: deployment changes require human review"


def _observe(task: DeploymentTask) -> dict:
    if task.environment.type == "shell":
        return {
            "host": _command(["uname", "-a"], task),
            "disk": _command(["df", "-h"], task, output_limit=2000),
            "processes": _command(["ps", "-eo", "comm"], task, output_limit=2000),
        }
    context = task.environment.context or ""
    checks = {
        "cluster": ["kubectl", "--context", context, "cluster-info"],
        "workloads": [
            "kubectl",
            "--context",
            context,
            "get",
            "deploy,statefulset,daemonset,pods",
            "-A",
            "-o",
            "wide",
        ],
        "storage": ["kubectl", "--context", context, "get", "storageclass,csidriver", "-o", "wide"],
        "releases": ["helm", "--kube-context", context, "list", "-A", "--short"],
    }
    return {
        name: _command(argv, task, timeout=30, output_limit=4000) for name, argv in checks.items()
    }


def _checks(task: DeploymentTask, checks: list[Check]) -> list[dict]:
    results = []
    for check in checks:
        command = _target_command(check.command, task)
        policy = _policy(check.command, task, mutating=False)
        result = (
            _command(command, task)
            if policy == "allowed"
            else {"exit_code": None, "stderr": policy}
        )
        results.append(
            {
                "command": command,
                "passed": result["exit_code"] == 0
                and (check.contains is None or check.contains in result.get("stdout", "")),
                "result": result,
            }
        )
    return results


def _absent(check: dict) -> bool:
    """Only a completed read or a known not-found response proves absence."""
    if check["passed"]:
        return False
    result = check["result"]
    if result["exit_code"] == 0:
        return True
    command = check["command"]
    error = result.get("stderr", "").casefold()
    return (
        (command[0] == "kubectl" and "(notfound)" in error)
        or (command[0] == "helm" and "not found" in error)
        or (command[0] == "ls" and "no such file or directory" in error)
    )


def _execute(
    decision: Decision,
    task: DeploymentTask,
    sources: SourceTools,
    workspace: Path,
    *,
    enabled: bool,
    approve: bool,
    timeout: int,
) -> dict:
    if not decision.command or not decision.expected_change or not decision.validation:
        return {
            "status": "blocked",
            "reason": "mutation needs command, expected change, and validation",
        }
    if not sources.existing_refs(decision.evidence):
        return {
            "status": "blocked",
            "reason": "mutation needs an inventoried source file as evidence",
        }
    policy = _policy(decision.command, task, mutating=True)
    if decision.command[0].startswith("./") and not decision.working_directory:
        return {"status": "blocked", "reason": "source script needs a working directory"}
    if policy == "allowed" and not _source_grounded(decision, sources):
        policy = "approval_required: command is not shown by its cited source"
    if policy.startswith("denied"):
        return {"status": "policy_denied", "reason": policy}
    if not enabled:
        return {
            "status": "blocked",
            "reason": "dry run; pass --execute",
            "command": decision.command,
            "policy": policy,
        }
    if task.environment.type == "kubernetes" and decision.command[0] not in {"kubectl", "helm"}:
        current = _command(["kubectl", "config", "current-context"], task)
        if current["exit_code"] != 0 or current["stdout"].strip() != task.environment.context:
            return {
                "status": "blocked",
                "reason": "source script would use a different Kubernetes context",
            }
    cwd = _source_cwd(decision, sources, workspace)
    before = _checks(task, decision.validation)
    if all(item["passed"] for item in before):
        return {
            "status": "already_satisfied",
            "reason": "the proposed outcome already passes; choose another gap or a stronger check",
            "before": before,
            "command": _target_command(decision.command, task),
        }
    if not any(_absent(item) for item in before):
        return {
            "status": "needs_better_check",
            "reason": "checks could not prove absence; inspect and refine them",
            "before": before,
            "command": _target_command(decision.command, task),
        }
    if policy.startswith("approval_required") and not approve:
        if not sys.stdin.isatty():
            return {"status": "blocked", "reason": policy, "command": decision.command}
        print(
            f"Approve deployment action on {task.environment.context or task.environment.type}?\n"
            f"Change: {decision.expected_change}\n"
            f"Source: {decision.working_directory or ', '.join(decision.evidence)}\n"
            f"Command: {shlex.join(_target_command(decision.command, task))}\n"
            f"Check: {'; '.join(shlex.join(check.command) for check in decision.validation)}\n"
            f"Reason: {decision.reason}\nPolicy: {policy}\n[y/N] ",
            end="",
            flush=True,
        )
        if input().strip().casefold() != "y":
            return {
                "status": "blocked",
                "reason": "human declined action",
                "command": decision.command,
            }
    command = _target_command(decision.command, task)
    deadline = time.monotonic() + timeout
    result = _command(command, task, cwd=cwd, timeout=timeout, output_limit=None)
    after = _checks(task, decision.validation)
    deadline = min(deadline, time.monotonic() + 300)
    while result["exit_code"] == 0 and not all(item["passed"] for item in after):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(5, remaining))
        after = _checks(task, decision.validation)
    return {
        "status": (
            "command_failed"
            if result["exit_code"] != 0
            else "validated"
            if all(item["passed"] for item in after)
            else "verification_failed"
        ),
        "expected_change": decision.expected_change,
        "policy": policy,
        "command": command,
        "before": before,
        "before_absent": any(_absent(item) for item in before),
        "execution": result,
        "after": after,
    }
