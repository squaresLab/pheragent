from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from .enums import InventoryCategory
from .models import InventoryEntry, RepositoryInventory
from .source_manager import AcquiredSource

_IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "vendor",
}
_DOCUMENT_SUFFIXES = {".md", ".markdown", ".rst", ".adoc", ".asciidoc"}
_CONFIG_SUFFIXES = {".env", ".ini", ".json", ".properties", ".toml", ".yaml", ".yml"}
_YAML_SUFFIXES = {".yaml", ".yml"}
_COMPOSE_FILE = re.compile(
    r"^(?:docker-)?compose(?:[._-][a-z0-9][a-z0-9_.-]*)?\.ya?ml$",
    re.IGNORECASE,
)
_CATEGORY_TECHNOLOGY = {
    InventoryCategory.ANSIBLE: "ansible",
    InventoryCategory.COMPOSE: "docker-compose",
    InventoryCategory.CI_WORKFLOW: "github-actions",
    InventoryCategory.HELM: "helm",
    InventoryCategory.HELMSMAN: "helmsman",
    InventoryCategory.KUBERNETES: "kubernetes",
    InventoryCategory.KUSTOMIZE: "kustomize",
    InventoryCategory.SHELL: "shell",
    InventoryCategory.TERRAFORM: "terraform",
}


class RepositoryInventoryBuilder:
    def __init__(self, *, max_file_size: int = 2 * 1024 * 1024):
        self.max_file_size = max_file_size

    def build(self, sources: tuple[AcquiredSource, ...]) -> RepositoryInventory:
        entries = [entry for source in sources for entry in self._inventory_source(source)]
        entries.sort(key=lambda item: (item.source_id, item.path))
        technologies = sorted(
            {
                technology
                for entry in entries
                if entry.selected
                for technology in [_CATEGORY_TECHNOLOGY.get(entry.category)]
                if technology is not None
            }
        )
        return RepositoryInventory(detected_technologies=technologies, entries=entries)

    def _inventory_source(self, source: AcquiredSource) -> list[InventoryEntry]:
        if source.path.is_file():
            return [self._inventory_file(source, source.path, source.path.name)]

        entries: list[InventoryEntry] = []
        for current_root, directory_names, file_names in os.walk(
            source.path,
            followlinks=False,
        ):
            directory_names[:] = sorted(
                name for name in directory_names if name not in _IGNORED_DIRECTORIES
            )
            current = Path(current_root)
            for file_name in sorted(file_names):
                file_path = current / file_name
                relative_path = file_path.relative_to(source.path).as_posix()
                entries.append(self._inventory_file(source, file_path, relative_path))
        return entries

    def _inventory_file(
        self,
        source: AcquiredSource,
        file_path: Path,
        relative_path: str,
    ) -> InventoryEntry:
        try:
            size_bytes = file_path.lstat().st_size
        except OSError as exc:
            return self._skipped(source, relative_path, 0, "stat_failed", warnings=[str(exc)])

        selection_reason = self._selection_reason(source, relative_path, file_path, size_bytes)
        if selection_reason is not None:
            return self._skipped(source, relative_path, size_bytes, selection_reason)

        sample = _read_sample(file_path)
        category = classify_file(relative_path, sample)
        if category == InventoryCategory.UNKNOWN:
            return self._skipped(
                source,
                relative_path,
                size_bytes,
                "unsupported_file_type",
            )
        return InventoryEntry(
            source_id=source.id,
            path=relative_path,
            category=category,
            size_bytes=size_bytes,
            selected=True,
        )

    def _selection_reason(
        self,
        source: AcquiredSource,
        relative_path: str,
        file_path: Path,
        size_bytes: int,
    ) -> str | None:
        if file_path.is_symlink():
            return "symbolic_link"
        if source.spec.include_patterns and not _matches_any(
            relative_path,
            source.spec.include_patterns,
        ):
            return "not_included"
        if _matches_any(relative_path, source.spec.exclude_patterns):
            return "excluded_pattern"
        if size_bytes > self.max_file_size:
            return "file_too_large"
        try:
            with file_path.open("rb") as handle:
                if b"\0" in handle.read(4096):
                    return "binary_file"
        except OSError:
            return "read_failed"
        return None

    @staticmethod
    def _skipped(
        source: AcquiredSource,
        path: str,
        size_bytes: int,
        reason: str,
        *,
        warnings: list[str] | None = None,
    ) -> InventoryEntry:
        return InventoryEntry(
            source_id=source.id,
            path=path,
            category=InventoryCategory.UNKNOWN,
            size_bytes=size_bytes,
            selected=False,
            skip_reason=reason,
            warnings=warnings or [],
        )


def classify_file(relative_path: str, sample: str) -> InventoryCategory:
    path = Path(relative_path)
    name = path.name.lower()
    suffix = path.suffix.lower()
    parts = {part.lower() for part in path.parts}
    normalized_path = relative_path.lower()

    if normalized_path.startswith(".github/workflows/") and suffix in _YAML_SUFFIXES:
        return InventoryCategory.CI_WORKFLOW
    if _COMPOSE_FILE.fullmatch(name):
        return InventoryCategory.COMPOSE
    if name == "chart.yaml" or name.startswith("values") and suffix in _YAML_SUFFIXES:
        return InventoryCategory.HELM
    if "templates" in parts and suffix in _YAML_SUFFIXES:
        return InventoryCategory.HELM
    if name in {"kustomization.yaml", "kustomization.yml"}:
        return InventoryCategory.KUSTOMIZE
    if suffix in {".tf", ".tfvars"}:
        return InventoryCategory.TERRAFORM
    if {"playbooks", "roles", "inventory"} & parts or name in {
        "ansible.cfg",
        "site.yaml",
        "site.yml",
    }:
        return InventoryCategory.ANSIBLE
    if suffix == ".sh" or sample.startswith("#!") and re.search(r"\b(?:ba|z|k)?sh\b", sample[:100]):
        return InventoryCategory.SHELL
    if name.startswith("readme") or suffix in _DOCUMENT_SUFFIXES:
        return InventoryCategory.DOCUMENTATION
    if suffix in _YAML_SUFFIXES:
        if "apiversion:" in sample.lower() and re.search(r"(?m)^\s*kind\s*:", sample):
            return InventoryCategory.KUBERNETES
        if "apps:" in sample.lower() and (
            "namespaces:" in sample.lower() or "helmsman" in sample.lower()
        ):
            return InventoryCategory.HELMSMAN
        if "dsf" in name:
            return InventoryCategory.HELMSMAN
        return InventoryCategory.CONFIGURATION
    if suffix in _CONFIG_SUFFIXES or name.startswith(".env"):
        return InventoryCategory.CONFIGURATION
    return InventoryCategory.UNKNOWN


def mark_inspected(
    entry: InventoryEntry,
    *,
    parser: str,
    warnings: list[str] | None = None,
) -> InventoryEntry:
    payload = entry.model_dump(mode="python")
    payload.update(inspected=True, parser=parser, warnings=warnings or [])
    return InventoryEntry.model_validate(payload)


def _matches_any(relative_path: str, patterns: list[str]) -> bool:
    path = Path(relative_path)
    for pattern in patterns:
        normalized = pattern.replace("\\", "/")
        if "/" not in normalized:
            if fnmatch.fnmatchcase(path.name, normalized):
                return True
        elif fnmatch.fnmatchcase(relative_path, normalized):
            return True
    return False


def _read_sample(path: Path, limit: int = 128 * 1024) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return handle.read(limit)
    except OSError:
        return ""
