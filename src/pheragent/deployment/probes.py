from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .models import ContractModel
from .redaction import redact_secrets

ProbeCommandRunner = Callable[[list[str], Path, float], tuple[int, str]]
_OUTPUT_LIMIT = 3_000


class ProbeName(StrEnum):
    SHELL_FILE_EXCERPT = "shell.file_excerpt"
    SHELL_LIST_DIRECTORY = "shell.list_directory"
    SHELL_COMMAND_EXISTS = "shell.command_exists"
    SHELL_COMMAND_VERSION = "shell.command_version"
    SHELL_SYNTAX_CHECK = "shell.syntax_check"
    HELM_VERSION = "helm.version"
    HELM_REPO_LIST = "helm.repo_list"
    HELM_RELEASE_LIST = "helm.release_list"
    HELM_CHART_METADATA = "helm.chart_metadata"
    HELM_DEPENDENCY_LIST = "helm.dependency_list"
    HELM_LINT = "helm.lint"
    HELM_TEMPLATE = "helm.template"
    HELM_RELEASE_STATUS = "helm.release_status"
    HELM_RELEASE_HISTORY = "helm.release_history"
    KUBERNETES_CURRENT_CONTEXT = "kubernetes.current_context"
    KUBERNETES_VERSION = "kubernetes.version"
    KUBERNETES_NAMESPACES = "kubernetes.namespaces"
    KUBERNETES_STORAGE_CLASSES = "kubernetes.storage_classes"
    KUBERNETES_RESOURCES = "kubernetes.resources"
    KUBERNETES_EVENTS = "kubernetes.events"
    KUBERNETES_LOGS = "kubernetes.logs"
    KUBERNETES_CAN_I = "kubernetes.can_i"
    GIT_COMMIT = "git.commit"
    GIT_STATUS = "git.status"
    GIT_DIFF_STAT = "git.diff_stat"
    GIT_DIFF = "git.diff"
    GIT_RECENT_COMMITS = "git.recent_commits"
    GIT_FILE_AT_COMMIT = "git.file_at_commit"


class ProbeRequest(ContractModel):
    """One bounded inspection selected from HerAgent's read-only catalogue."""

    name: ProbeName
    path: str | None = None
    namespace: str | None = None
    release: str | None = None
    resource: str | None = None
    command: str | None = None
    verb: str | None = None
    pod: str | None = None
    container: str | None = None


@dataclass(frozen=True, slots=True)
class ProbeResult:
    request: ProbeRequest
    succeeded: bool
    output: str = ""
    error: str = ""


class ProbeRunner:
    """Run a small catalogue of inspections without accepting arbitrary shell text."""

    def __init__(
        self,
        *,
        command_runner: ProbeCommandRunner | None = None,
        timeout: float = 20.0,
    ) -> None:
        self._command_runner = command_runner or _run_command
        self._timeout = timeout

    def run(self, request: ProbeRequest, *, root: Path) -> ProbeResult:
        try:
            output = self._inspect(request, root.expanduser().resolve(strict=True))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return ProbeResult(request, False, error=redact_secrets(str(exc)))
        return ProbeResult(request, True, output=_compact(redact_secrets(output)))

    def _inspect(self, request: ProbeRequest, root: Path) -> str:
        if request.name == "shell.file_excerpt":
            return _compact(_safe_path(root, request.path).read_text(encoding="utf-8"))
        if request.name == "shell.list_directory":
            path = _safe_path(root, request.path or ".")
            return "\n".join(sorted(item.name for item in path.iterdir()))
        if request.name == "shell.command_exists":
            command = _safe_word(request.command, "command")
            return shutil.which(command) or "not found"

        arguments, cwd = _probe_command(request, root)
        return_code, output = self._command_runner(arguments, cwd, self._timeout)
        if return_code:
            raise ValueError(f"probe exited with {return_code}: {_compact(output)}")
        return output


def default_probes(executor: str, output: str) -> tuple[ProbeRequest, ...]:
    """Choose a cheap first evidence set; the recovery model may request more."""
    probes = [ProbeRequest(name="git.commit"), ProbeRequest(name="git.status")]
    normalized = f"{executor} {output}".casefold()
    if "helm" in normalized or "chart" in normalized:
        probes.extend(
            [ProbeRequest(name="helm.version"), ProbeRequest(name="helm.repo_list")]
        )
    if any(word in normalized for word in ("kubectl", "kubernetes", "namespace", "pod")):
        probes.extend(
            [
                ProbeRequest(name="kubernetes.current_context"),
                ProbeRequest(name="kubernetes.version"),
            ]
        )
    return tuple(probes)


def _probe_command(request: ProbeRequest, root: Path) -> tuple[list[str], Path]:
    commands: dict[str, list[str]] = {
        "helm.version": ["helm", "version", "--short"],
        "helm.repo_list": ["helm", "repo", "list"],
        "helm.release_list": ["helm", "list", "-A", "--all"],
        "kubernetes.current_context": ["kubectl", "config", "current-context"],
        "kubernetes.version": ["kubectl", "version"],
        "kubernetes.namespaces": ["kubectl", "get", "namespaces", "--show-labels"],
        "kubernetes.storage_classes": ["kubectl", "get", "storageclass"],
        "git.commit": ["git", "rev-parse", "HEAD"],
        "git.status": ["git", "status", "--short"],
        "git.diff_stat": ["git", "diff", "--stat"],
        "git.diff": ["git", "diff", "--"],
        "git.recent_commits": ["git", "log", "-5", "--oneline"],
    }
    if request.name in commands:
        return commands[request.name], root
    if request.name == "shell.command_version":
        return [_safe_word(request.command, "command"), "--version"], root
    if request.name == "shell.syntax_check":
        return ["bash", "-n", str(_safe_path(root, request.path))], root
    if request.name in {"helm.chart_metadata", "helm.dependency_list", "helm.lint"}:
        action = {
            "helm.chart_metadata": ["show", "chart"],
            "helm.dependency_list": ["dependency", "list"],
            "helm.lint": ["lint"],
        }[request.name]
        return ["helm", *action, str(_safe_path(root, request.path or "."))], root
    if request.name == "helm.template":
        chart = str(_safe_path(root, request.path or "."))
        return ["helm", "template", "heragent-check", chart], root
    if request.name in {"helm.release_status", "helm.release_history"}:
        action = "status" if request.name.endswith("status") else "history"
        command = ["helm", action, _safe_word(request.release, "release")]
        if request.namespace:
            command.extend(["-n", _safe_word(request.namespace, "namespace")])
        return command, root
    if request.name in {"kubernetes.resources", "kubernetes.events"}:
        resource = "events" if request.name.endswith("events") else _safe_resource(request.resource)
        command = ["kubectl", "get", resource]
        if request.namespace:
            command.extend(["-n", _safe_word(request.namespace, "namespace")])
        command.extend(["-o", "wide"])
        return command, root
    if request.name == "kubernetes.logs":
        command = ["kubectl", "logs", _safe_word(request.pod, "pod"), "--tail", "100"]
        if request.namespace:
            command.extend(["-n", _safe_word(request.namespace, "namespace")])
        if request.container:
            command.extend(["-c", _safe_word(request.container, "container")])
        return command, root
    if request.name == "kubernetes.can_i":
        return [
            "kubectl",
            "auth",
            "can-i",
            _safe_word(request.verb, "verb"),
            _safe_resource(request.resource),
        ], root
    if request.name == "git.file_at_commit":
        path = _safe_relative_path(request.path)
        return ["git", "show", f"HEAD:{path.as_posix()}"], root
    raise ValueError(f"unknown read-only probe: {request.name}")


def _safe_path(root: Path, value: str | None) -> Path:
    if not value:
        raise ValueError("probe requires a path")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("probe path is outside source root")
    return path


def _safe_relative_path(value: str | None) -> Path:
    if not value:
        raise ValueError("probe requires a path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("probe path is outside source root")
    return path


def _safe_word(value: str | None, label: str) -> str:
    if not value or value.startswith("-") or not all(
        character.isalnum() or character in "._-/" for character in value
    ):
        raise ValueError(f"probe requires a safe {label}")
    return value


def _safe_resource(value: str | None) -> str:
    resource = _safe_word(value, "resource")
    if "secret" in resource.casefold():
        raise ValueError("secret contents are not available to recovery probes")
    return resource


def _run_command(arguments: list[str], cwd: Path, timeout: float) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            arguments,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return 127, str(exc)
    return completed.returncode, completed.stdout + completed.stderr


def _compact(text: str) -> str:
    if len(text) <= _OUTPUT_LIMIT:
        return text
    return text[:500] + "\n...[truncated]...\n" + text[-2_470:]
