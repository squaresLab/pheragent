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
    "events",
}
_READ_HELM = {"list", "status", "show", "get", "version", "history", "env", "lint", "template"}
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
_DENIED = {"rm", "shutdown", "reboot", "mkfs", "dd"}
_COMPOSE_OPTIONS_WITH_VALUE = {
    "-f",
    "--file",
    "--env-file",
    "-p",
    "--project-name",
    "--profile",
    "--project-directory",
}
_KUBECTL_OPTIONS_WITH_VALUE = {"-n", "--namespace", "--request-timeout"}
_HELM_OPTIONS_WITH_VALUE = {"-n", "--namespace"}


def _subcommand(arguments: list[str], options_with_value: set[str]) -> tuple[str | None, int]:
    index = 0
    while index < len(arguments) and arguments[index].startswith("-"):
        option = arguments[index].split("=", 1)[0]
        if option not in options_with_value:
            return None, index
        index += 1 if "=" in arguments[index] else 2
    return (arguments[index] if index < len(arguments) else None), index


def _compose_subcommand(arguments: list[str]) -> str | None:
    index = 0
    while index < len(arguments) and arguments[index].startswith("-"):
        option = arguments[index].split("=", 1)[0]
        if option in _COMPOSE_OPTIONS_WITH_VALUE and "=" not in arguments[index]:
            index += 2
        elif option in {"--compatibility", "--dry-run"}:
            index += 1
        else:
            return None
    return arguments[index] if index < len(arguments) else None


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
        verb, index = _subcommand(command[1:], _KUBECTL_OPTIONS_WITH_VALUE)
        verb = verb or ""
        offset = index + 1
        if verb == "auth":
            return command[offset + 1 : offset + 2] == ["can-i"]
        if verb == "config":
            return command[offset + 1 : offset + 2] in (["current-context"], ["get-contexts"])
        return verb in _READ_KUBECTL and not (verb == "rollout" and "status" not in command)
    if command[0] == "helm":
        verb, index = _subcommand(command[1:], _HELM_OPTIONS_WITH_VALUE)
        verb = verb or ""
        offset = index + 1
        return verb in _READ_HELM or command[offset : offset + 2] in (
            ["repo", "list"],
            ["search", "repo"],
            ["search", "hub"],
            ["dependency", "list"],
        )
    if command[0] == "aws":
        return command[1:2] == ["--version"] or tuple(command[1:3]) in _READ_AWS
    if command[0] == "terraform":
        return command[1:2] in (["version"], ["validate"], ["show"], ["output"], ["plan"]) or (
            command[1:2] == ["state"] and command[2:3] in (["list"], ["show"])
        )
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
        return len(command) > 1 and command[1] in {"status", "log", "show", "rev-parse", "diff"}
    if command[0] == "docker":
        if len(command) < 2:
            return False
        if command[1] in {"ps", "images", "inspect", "logs", "info", "version"}:
            return True
        if command[1:3] in (["image", "ls"], ["image", "inspect"]):
            return True
        if command[1] == "events":
            return any(part == "--until" or part.startswith("--until=") for part in command[2:])
        if command[1:2] != ["compose"]:
            return False
        subcommand = _compose_subcommand(command[2:])
        if subcommand == "config":
            return any(
                part in {"-q", "--quiet", "--images", "--profiles", "--services", "--volumes"}
                for part in command[2:]
            )
        return subcommand in {
            "images",
            "logs",
            "ls",
            "ps",
            "top",
            "version",
        }
    return command[0] in {
        "ls",
        "pwd",
        "uname",
        "df",
        "du",
        "free",
        "ps",
        "stat",
        "lsblk",
        "lscpu",
        "whoami",
        "id",
        "grep",
    }


def _inspect(
    decision: Decision,
    task: DeploymentTask,
    sources: SourceTools,
    workspace: Path,
    *,
    enabled: bool,
    approve: bool,
) -> dict:
    policy = _policy(decision.command, task, mutating=False)
    if policy.startswith("denied"):
        return {"error": policy}
    if policy != "allowed":
        if not enabled:
            return {
                "status": "blocked",
                "reason": "unclassified observation; pass --execute for human review",
            }
        high_risk = policy.startswith("high_risk_approval")
        accepted = approve and not high_risk
        if not accepted and sys.stdin.isatty():
            confirmation = "approve" if high_risk else "y"
            print(
                f"Approve {'HIGH-RISK ' if high_risk else ''}unclassified observation?\n"
                f"Command: {shlex.join(_target_command(decision.command, task))}\n"
                f"Reason: {decision.reason}\nPolicy: {policy}\n"
                f"[{'type approve' if high_risk else 'y'}/N] ",
                end="",
                flush=True,
            )
            accepted = input().strip().casefold() == confirmation
        if not accepted:
            return {"status": "blocked", "reason": policy}
    cwd = _source_cwd(decision, sources, workspace)
    return _command(_target_command(decision.command, task), task, cwd=cwd)


def _high_risk(command: list[str]) -> str | None:
    executable = Path(command[0]).name
    if executable == "sudo":
        return "privileged host change"
    if executable == "kubectl" and any(
        part in {"delete", "replace", "patch", "drain"} for part in command
    ):
        return "potentially destructive Kubernetes operation"
    if executable == "helm" and any(part in {"uninstall", "rollback"} for part in command):
        return "potentially destructive Helm operation"
    if executable in {"aws", "az", "gcloud", "terraform"}:
        return "cloud or infrastructure change"
    if executable == "docker" and (
        command[1:3] in (["system", "prune"], ["volume", "prune"], ["volume", "rm"])
        or (
            command[1:2] == ["compose"]
            and _compose_subcommand(command[2:]) == "down"
            and any(part in {"-v", "--volumes"} for part in command)
        )
    ):
        return "potentially destructive container operation"
    return None


def _infrastructure_change(command: list[str]) -> bool:
    return Path(command[0]).name in {"aws", "az", "gcloud", "terraform"} and not _read_only(command)


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
    if Path(command[0]).name == "aws" and any(
        part.split("=", 1)[0] in {"--profile", "--endpoint-url", "--no-verify-ssl", "--debug"}
        for part in command[1:]
    ):
        return "denied: AWS profile, endpoint, and debug overrides are not allowed"
    if command[:2] == ["kubectl", "config"] and command[2:3] not in (
        ["current-context"],
        ["get-contexts"],
    ):
        return "denied: Kubernetes context changes are not allowed"
    if _read_only(command):
        return "allowed"
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
    risk = _high_risk(command)
    if risk and "destructive" in risk and not task.constraints.allow_destructive_actions:
        return "denied: destructive actions are outside task constraints"
    if _infrastructure_change(command) and not task.constraints.allow_new_infrastructure:
        return "denied: infrastructure changes are outside task constraints"
    if risk:
        return f"high_risk_approval: {risk}"
    if not mutating:
        return "approval_required: command is not known to be read-only; propose it as execute"
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


def _checks(task: DeploymentTask, checks: list[Check], cwd: Path | None = None) -> list[dict]:
    results = []
    for check in checks:
        command = _target_command(check.command, task)
        policy = _policy(check.command, task, mutating=False)
        result = (
            _command(command, task, cwd=cwd)
            if policy == "allowed"
            else {"exit_code": None, "stderr": policy}
        )
        results.append(
            {
                "command": command,
                "status": (
                    "satisfied"
                    if result["exit_code"] == 0
                    and (check.contains is None or check.contains in result.get("stdout", ""))
                    else "unsatisfied"
                    if result["exit_code"] == 0
                    or result["exit_code"] in check.unsatisfied_exit_codes
                    else "unknown"
                ),
                "result": result,
            }
        )
    return results


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
            "status": "needs_revision",
            "reason": "mutation needs command, expected change, and validation",
        }
    if not sources.existing_refs(decision.evidence):
        return {
            "status": "needs_revision",
            "reason": "mutation needs an inventoried source file as evidence",
        }
    policy = _policy(decision.command, task, mutating=True)
    if decision.command[0].startswith("./") and not decision.working_directory:
        return {"status": "needs_revision", "reason": "source script needs a working directory"}
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
    before = _checks(task, decision.validation, cwd)
    if all(item["status"] == "satisfied" for item in before):
        return {
            "status": "already_satisfied",
            "reason": "the proposed outcome already passes; choose another gap or a stronger check",
            "before": before,
            "command": _target_command(decision.command, task),
        }
    if any(item["status"] == "unknown" for item in before):
        return {
            "status": "needs_better_check",
            "reason": (
                "validation was inconclusive; use a source-defined check and declare "
                "its unsatisfied exit codes"
            ),
            "before": before,
            "command": _target_command(decision.command, task),
        }
    needs_review = policy.startswith("high_risk_approval") or (
        policy.startswith("approval_required") and not approve
    )
    if needs_review:
        if not sys.stdin.isatty():
            return {"status": "blocked", "reason": policy, "command": decision.command}
        high_risk = policy.startswith("high_risk_approval")
        confirmation = "approve" if high_risk else "y"
        print(
            f"Approve {'HIGH-RISK ' if high_risk else ''}deployment action on "
            f"{task.environment.context or task.environment.type}\n"
            f"Change: {decision.expected_change}\n"
            f"Source: {decision.working_directory or ', '.join(decision.evidence)}\n"
            f"Command: {shlex.join(_target_command(decision.command, task))}\n"
            f"Check: {'; '.join(shlex.join(check.command) for check in decision.validation)}\n"
            f"Reason: {decision.reason}\nPolicy: {policy}\n"
            f"[{'type approve' if high_risk else 'y'}/N] ",
            end="",
            flush=True,
        )
        if input().strip().casefold() != confirmation:
            return {
                "status": "blocked",
                "reason": "human declined action",
                "command": decision.command,
            }
    command = _target_command(decision.command, task)
    deadline = time.monotonic() + timeout
    result = _command(command, task, cwd=cwd, timeout=timeout, output_limit=None)
    after = _checks(task, decision.validation, cwd)
    deadline = min(deadline, time.monotonic() + 300)
    while result["exit_code"] == 0 and not all(
        item["status"] == "satisfied" for item in after
    ):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(5, remaining))
        after = _checks(task, decision.validation, cwd)
    return {
        "status": (
            "command_failed"
            if result["exit_code"] != 0
            else "validated"
            if all(item["status"] == "satisfied" for item in after)
            else "verification_failed"
        ),
        "expected_change": decision.expected_change,
        "policy": policy,
        "command": command,
        "before": before,
        "before_unsatisfied": any(item["status"] == "unsatisfied" for item in before),
        "execution": result,
        "after": after,
    }
