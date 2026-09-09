from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from pheragent.deployment.analysis_models import DeploymentContext
from pheragent.deployment.redaction import redact_secrets

_IGNORED_DIRECTORIES = {
    ".git",
    ".pheragent",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "target",
    "tests",
    "test",
    "vendor",
}
_WORKLOAD_KINDS = {"cronjob", "daemonset", "deployment", "job", "statefulset"}
_GENERIC_SCRIPT_DIRECTORIES = {
    "all",
    "ansible",
    "bin",
    "chart",
    "charts",
    "deploy",
    "deployment",
    "helm",
    "install",
    "k8s",
    "kubernetes",
    "manifests",
    "operator",
    "scripts",
    "setup",
    "templates",
    "tools",
}
_GENERIC_IDENTITY_WORDS = {
    "application",
    "component",
    "deployment",
    "module",
    "platform",
    "server",
    "service",
    "services",
    "system",
}
_DEPLOYMENT_COMMAND = re.compile(
    r"\b(?:ansible-playbook|docker\s+compose|helm|kubectl)\b",
    re.IGNORECASE,
)
_TEXT_LIMIT = 1_000_000


@dataclass(frozen=True, slots=True)
class DeploymentEntity:
    name: str
    kind: str
    evidence_refs: tuple[str, ...]


class SourceCatalog:
    """Provide containment-safe access to pinned sources and their declared entities."""

    def __init__(self, roots: dict[str, Path]) -> None:
        self._roots = {source_id: path.expanduser().resolve() for source_id, path in roots.items()}
        missing = sorted(source_id for source_id, path in self._roots.items() if not path.exists())
        if missing:
            raise ValueError(f"source roots do not exist: {', '.join(missing)}")

    def resolve(self, source_id: str, relative_path: str) -> Path | None:
        root = self._roots.get(source_id)
        if root is None:
            return None
        pure_path = PurePosixPath(relative_path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            return None
        if root.is_file():
            return root if pure_path.as_posix() in {root.name, "."} else None
        candidate = root.joinpath(*pure_path.parts).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def resolve_directory(self, source_id: str, relative_path: str | None) -> Path | None:
        root = self._roots.get(source_id)
        if root is None or root.is_file():
            return None
        raw_path = relative_path or "."
        pure_path = PurePosixPath(raw_path)
        if pure_path.is_absolute() or ".." in pure_path.parts:
            return None
        candidate = root.joinpath(*pure_path.parts).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate if candidate.is_dir() else None

    def relative_file_exists(
        self,
        source_id: str,
        working_directory: str | None,
        relative_path: str,
    ) -> bool:
        base = self.resolve_directory(source_id, working_directory)
        root = self._roots.get(source_id)
        pure_path = PurePosixPath(relative_path)
        if base is None or root is None or pure_path.is_absolute() or ".." in pure_path.parts:
            return False
        candidate = base.joinpath(*pure_path.parts).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        return candidate.is_file()

    def supports_identity(self, source_id: str, relative_path: str, *names: str | None) -> bool:
        path = self.resolve(source_id, relative_path)
        if path is None:
            return False
        haystacks = (path.as_posix(), self._read_text(path))
        return any(
            identity_matches(name, haystack) for name in names if name for haystack in haystacks
        )

    def excerpt(
        self,
        source_id: str,
        relative_path: str,
        *,
        names: tuple[str, ...] = (),
        start_line: int | None = None,
        end_line: int | None = None,
        max_characters: int = 800,
    ) -> str:
        path = self.resolve(source_id, relative_path)
        if path is None:
            return ""
        lines = self._read_text(path).splitlines()
        if not lines:
            return ""
        if start_line is not None:
            start = max(0, start_line - 1)
            stop = min(len(lines), end_line or start_line + 5)
        else:
            match = _matching_line(lines, names)
            start = max(0, match - 2)
            stop = min(len(lines), match + 5)
        redacted_lines = redact_secrets("\n".join(lines[start:stop])).splitlines()
        rendered = "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(redacted_lines, start=start + 1)
        )
        return rendered[:max_characters]

    def deployment_entities(
        self,
        context: DeploymentContext,
        selected_paths: dict[str, set[str]] | None = None,
    ) -> tuple[DeploymentEntity, ...]:
        mode = deployment_mode(context)
        selected_paths = selected_paths or {}
        observations: dict[str, tuple[set[str], set[str]]] = {}
        for source_id, root in sorted(self._roots.items()):
            for path in self._iter_files(root):
                relative_path = path.name if root.is_file() else path.relative_to(root).as_posix()
                evidence_ref = f"{source_id}:{relative_path}"
                suffix = path.suffix.casefold()
                if suffix in {".yaml", ".yml"} and path.stat().st_size <= _TEXT_LIMIT:
                    include_compose = _primary_compose_file(relative_path) or relative_path in (
                        selected_paths.get(source_id) or set()
                    )
                    for name, kind in self._yaml_entities(path, mode, include_compose):
                        identity = normalize_identity(name)
                        kinds, refs = observations.setdefault(identity, (set(), set()))
                        kinds.add(kind)
                        refs.add(evidence_ref)
                if path.name.casefold() in {"install.sh", "run.sh"}:
                    text = self._read_text(path)
                    if _script_matches_mode(text, mode):
                        name = _script_component_name(path, root)
                        if name:
                            identity = normalize_identity(name)
                            kinds, refs = observations.setdefault(identity, (set(), set()))
                            kinds.add("installer")
                            refs.add(evidence_ref)
        return tuple(
            DeploymentEntity(
                name=identity,
                kind="+".join(sorted(kinds)),
                evidence_refs=tuple(sorted(refs)),
            )
            for identity, (kinds, refs) in sorted(observations.items())
            if identity
        )

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace")[:_TEXT_LIMIT]
        except OSError:
            return ""

    @staticmethod
    def _iter_files(root: Path):
        if root.is_file():
            yield root
            return
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative_parts = path.relative_to(root).parts[:-1]
            if any(part.casefold() in _IGNORED_DIRECTORIES for part in relative_parts):
                continue
            yield path

    def _yaml_entities(
        self,
        path: Path,
        mode: str,
        include_compose: bool,
    ) -> list[tuple[str, str]]:
        try:
            documents = list(yaml.safe_load_all(self._read_text(path)))
        except yaml.YAMLError:
            return []
        entities: list[tuple[str, str]] = []
        for document in documents:
            if isinstance(document, list) and mode in {"kubernetes", "mixed"}:
                entities.extend(
                    entity
                    for item in document
                    if isinstance(item, dict)
                    for entity in _ansible_entities(item)
                )
                continue
            if not isinstance(document, dict):
                continue
            if mode in {"compose", "mixed"} and include_compose:
                services = document.get("services")
                if isinstance(services, dict) and "kind" not in document:
                    entities.extend((str(name), "compose_service") for name in services)
            if mode in {"kubernetes", "mixed"}:
                entities.extend(_kubernetes_entities(document))
                if path.name.casefold() == "chart.yaml":
                    entities.extend(_helm_entities(document))
                if "ansible" in {part.casefold() for part in path.parts}:
                    entities.extend(_ansible_entities(document))
        return entities


def deployment_mode(context: DeploymentContext) -> str:
    terms = " ".join(
        filter(
            None,
            [
                context.deployment.profile,
                *(block.subtype for block in context.provided_blocks),
                *(block.implementation for block in context.provided_blocks),
            ],
        )
    ).casefold()
    compose = "compose" in terms or "docker" in terms
    kubernetes = any(term in terms for term in ("kubernetes", "k8s", "rke2", "eks", "aks", "gke"))
    if compose and not kubernetes:
        return "compose"
    if kubernetes and not compose:
        return "kubernetes"
    return "mixed"


def identity_matches(left: str, right: str) -> bool:
    left_words = _identity_words(left)
    right_words = _identity_words(right)
    if not left_words or not right_words:
        return False
    left_compact = "".join(left_words)
    right_compact = "".join(right_words)
    if left_compact == right_compact:
        return True
    shorter, longer = sorted((left_compact, right_compact), key=len)
    if len(shorter) >= 5 and longer.startswith(shorter) and len(shorter) / len(longer) >= 0.7:
        return True
    return _words_match(left_words, right_words)


def normalize_identity(value: str) -> str:
    return "-".join(_identity_words(value))


def _identity_words(value: str) -> tuple[str, ...]:
    return tuple(
        word
        for word in re.findall(r"[a-z0-9]+", value.casefold())
        if word not in _GENERIC_IDENTITY_WORDS
    )


def _words_match(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    if set(left) <= set(right) or set(right) <= set(left):
        return True
    return any(
        len(shorter) >= 5 and longer.startswith(shorter) and len(shorter) / len(longer) >= 0.7
        for left_word in left
        for right_word in right
        for shorter, longer in [sorted((left_word, right_word), key=len)]
    )


def _kubernetes_entities(document: dict) -> list[tuple[str, str]]:
    if str(document.get("kind", "")).casefold() == "list":
        return [
            entity
            for item in document.get("items", [])
            if isinstance(item, dict)
            for entity in _kubernetes_entities(item)
        ]
    kind = str(document.get("kind", "")).casefold()
    metadata = document.get("metadata")
    if kind not in _WORKLOAD_KINDS or not isinstance(metadata, dict):
        return []
    name = metadata.get("name")
    if not isinstance(name, str) or not name.strip() or "{{" in name:
        return []
    return [(name, f"kubernetes_{kind}")]


def _helm_entities(document: dict) -> list[tuple[str, str]]:
    entities = []
    name = document.get("name")
    if isinstance(name, str) and name.strip() and "{{" not in name:
        entities.append((name, "helm_chart"))
    dependencies = document.get("dependencies", [])
    if isinstance(dependencies, list):
        entities.extend(
            (dependency["name"], "helm_dependency")
            for dependency in dependencies
            if isinstance(dependency, dict)
            and isinstance(dependency.get("name"), str)
            and "{{" not in dependency["name"]
        )
    return entities


def _ansible_entities(document: dict) -> list[tuple[str, str]]:
    roles = document.get("roles", [])
    if not isinstance(roles, list):
        return []
    result = []
    for role in roles:
        name = (
            role if isinstance(role, str) else role.get("role") if isinstance(role, dict) else None
        )
        if isinstance(name, str) and name.strip() and "{{" not in name:
            result.append((name, "ansible_role"))
    return result


def _script_matches_mode(text: str, mode: str) -> bool:
    commands = {match.casefold().replace(" ", "_") for match in _DEPLOYMENT_COMMAND.findall(text)}
    if mode == "compose":
        return "docker_compose" in commands
    if mode == "kubernetes":
        return bool(commands & {"ansible-playbook", "helm", "kubectl"})
    return bool(commands)


def _script_component_name(path: Path, root: Path) -> str | None:
    if root.is_file():
        return None
    current = path.parent
    while current != root and current.name.casefold() in _GENERIC_SCRIPT_DIRECTORIES:
        current = current.parent
    return current.name if current != root else None


def _primary_compose_file(relative_path: str) -> bool:
    return PurePosixPath(relative_path).name.casefold() in {
        "compose.yaml",
        "compose.yml",
        "docker-compose.yaml",
        "docker-compose.yml",
    }


def _matching_line(lines: list[str], names: tuple[str, ...]) -> int:
    identities = [normalize_identity(name).replace("-", "") for name in names if name]
    for index, line in enumerate(lines):
        normalized_line = normalize_identity(line).replace("-", "")
        if any(identity and identity in normalized_line for identity in identities):
            return index
    return 0
