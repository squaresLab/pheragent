"""Acquire and read source evidence used by deployment decisions."""

from __future__ import annotations

import hashlib
import posixpath
import re
import shlex
import shutil
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from .inventory import RepositoryInventoryBuilder
from .models import SourceKind, SourceSpec
from .redaction import redact_secrets
from .retrieval import DeploymentRetrievalEngine, RetrievalQuery
from .source_manager import AcquiredSource
from .task import Decision, SourceLocation

_ENTRYPOINT_CATEGORIES = {
    "ansible",
    "build",
    "ci_workflow",
    "compose",
    "helm",
    "helmsman",
    "terraform",
}
_ENTRYPOINT_DIRECTORIES = {"ansible", "deploy", "deployment", "scripts"}
_MAX_COMPLETE_FILE_CHARACTERS = 100_000
_REFERENCE = re.compile(r"(?:\[[^\]]*\]\(([^)]+)\)|(?:^|[\s'\"`])([.\w/-]+\.[\w.-]+))")


def _source_spec(item: str | SourceLocation, purpose: str, index: int, base: Path) -> SourceSpec:
    location = item if isinstance(item, str) else item.location
    revision = None if isinstance(item, str) else item.revision
    if location.startswith(("https://", "http://", "git@", "ssh://")):
        kind = SourceKind.GIT
    else:
        local = Path(location)
        local = local if local.is_absolute() else base / local
        kind = SourceKind.LOCAL_FILE if local.is_file() else SourceKind.LOCAL_DIRECTORY
    return SourceSpec(
        id=f"{purpose}-{index}",
        kind=kind,
        location=location,
        purpose=purpose,
        revision=revision,
    )


def _source_tree(paths: set[str]) -> dict:
    roots: dict[str, dict] = {}
    for reference in paths:
        source_id, _, relative = reference.partition(":")
        branch = roots.setdefault(source_id, {})
        for part in PurePosixPath(relative).parts:
            branch = branch.setdefault(part, {})

    lines = []
    length = 0

    def append(line: str) -> bool:
        nonlocal length
        if length + len(line) + 1 > 5500:
            return False
        lines.append(line)
        length += len(line) + 1
        return True

    def visit(branch: dict, depth: int) -> bool:
        for name, children in sorted(branch.items(), key=lambda item: (bool(item[1]), item[0])):
            if not append(f"{'  ' * depth}{name}{'/' if children else ''}"):
                return False
            if children and not visit(children, depth + 1):
                return False
        return True

    complete = True
    for source_id, branch in sorted(roots.items()):
        if not append(f"{source_id}:") or not visit(branch, 1):
            complete = False
            break
    return {"tree": "\n".join(lines), "total": len(paths), "truncated": not complete}


class SourceTools:
    def __init__(self, sources: tuple[AcquiredSource, ...]):
        self.sources = {source.id: source for source in sources}
        inventory = RepositoryInventoryBuilder().build(sources)
        self.entries = [entry for entry in inventory.entries if entry.selected]
        documents = []
        for entry in self.entries:
            source = self.sources[entry.source_id]
            try:
                content = source.resolve_path(entry.path).read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                continue
            documents.extend(
                DeploymentRetrievalEngine.passages(
                    source_id=entry.source_id,
                    source_kind=source.spec.purpose,
                    path=entry.path,
                    text=redact_secrets(content),
                )
            )
        self.search_index = DeploymentRetrievalEngine(documents)
        self.paths = {f"{entry.source_id}:{entry.path}" for entry in self.entries}
        self.categories = {
            f"{entry.source_id}:{entry.path}": entry.category.value for entry in self.entries
        }
        self.readable_paths = {
            f"{entry.source_id}:{entry.path}"
            for entry in inventory.entries
            if entry.selected or entry.skip_reason == "unsupported_file_type"
        }

    def reference_graph(self) -> list[dict[str, str]]:
        edges = set()
        by_source = {
            source_id: {
                reference.partition(":")[2]
                for reference in self.readable_paths
                if reference.startswith(source_id + ":")
            }
            for source_id in self.sources
        }
        for reference in self.readable_paths:
            source_id, _, relative = reference.partition(":")
            try:
                text = self.sources[source_id].resolve_path(relative).read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                continue
            parent = PurePosixPath(relative).parent
            for match in _REFERENCE.finditer(text):
                target = (match.group(1) or match.group(2)).split("#", 1)[0].strip()
                if not target or "://" in target or target.startswith(("#", "/")):
                    continue
                candidates = (
                    posixpath.normpath((parent / target).as_posix()),
                    posixpath.normpath(PurePosixPath(target).as_posix()),
                )
                resolved = next((item for item in candidates if item in by_source[source_id]), None)
                if resolved:
                    edges.add((reference, f"{source_id}:{resolved}"))
        return [
            {"from": source, "to": target}
            for source, target in sorted(edges)[:500]
        ]

    def related_paths(self, references: list[str], limit: int = 100) -> list[str]:
        selected = set(references)
        neighbors = {
            endpoint
            for edge in self.reference_graph()
            if edge["from"] in selected or edge["to"] in selected
            for endpoint in edge.values()
            if endpoint not in selected
        }
        return sorted(neighbors)[:limit]

    def entrypoint_candidates(self) -> list[str]:
        candidates = []
        for reference in self.paths:
            _source_id, _, relative = reference.partition(":")
            path = PurePosixPath(relative)
            name = path.name.casefold()
            explicit = name.startswith(
                ("readme", "dockerfile", "docker-compose", "helmfile")
            ) or name in {"chart.yaml", "makefile"}
            if (
                explicit
                or self.categories[reference] in _ENTRYPOINT_CATEGORIES
                or _ENTRYPOINT_DIRECTORIES.intersection(
                    part.casefold() for part in path.parts[:-1]
                )
            ):
                root = len(path.parts) == 1
                rank = (
                    0
                    if root and name.startswith("readme")
                    else 1
                    if root and explicit
                    else 2
                    if name.startswith("readme")
                    else 3
                    if explicit
                    else 4
                    if self.categories[reference] in _ENTRYPOINT_CATEGORIES
                    else 5
                )
                candidates.append((rank, len(path.parts), reference.casefold(), reference))
        return [item[-1] for item in sorted(candidates)[:200]]

    def existing_refs(self, references: list[str]) -> set[str]:
        return {
            path
            for path in self.readable_paths
            if any(
                ref == path or ref.startswith((path + ":", path + " —", path + " lines "))
                for ref in references
            )
        }

    def supports_source(self, location: str, references: list[str]) -> bool:
        parsed = urlsplit(location)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            return False
        expected = location.removesuffix(".git").rstrip("/")
        for reference in self.existing_refs(references):
            source_id, _, relative = reference.partition(":")
            text = self.sources[source_id].resolve_path(relative).read_text(
                encoding="utf-8", errors="replace"
            )
            if expected in text.replace(".git", ""):
                return True
        return False

    def read_file(
        self, reference: str, start_line: int | None = None, end_line: int | None = None
    ) -> dict:
        identifier, separator, relative = reference.partition(":")
        if not separator or identifier not in self.sources:
            raise ValueError("source_path must be source-id:relative/path")
        if reference not in self.readable_paths:
            raise ValueError("source file is not in the readable inventory")
        content = self.sources[identifier].resolve_path(relative).read_text(
            encoding="utf-8", errors="replace"
        )
        lines = content.splitlines()
        start = max(1, start_line or 1)
        complete = start_line is None and end_line is None
        digest = hashlib.sha256(content.encode()).hexdigest()
        if complete and len(content) > _MAX_COMPLETE_FILE_CHARACTERS:
            return {
                "source": reference,
                "complete": False,
                "total_lines": len(lines),
                "sha256": digest,
                "reason": "file is too large for one model call; request a line range",
            }
        end = len(lines) if complete else min(len(lines), end_line or start + 119)
        return {
            "source": reference,
            "start_line": start,
            "end_line": end,
            "total_lines": len(lines),
            "complete": complete,
            "sha256": digest,
            "text": redact_secrets("\n".join(lines[start - 1 : end])),
        }

    def inventory(self) -> dict:
        return {
            **_source_tree(self.readable_paths),
            "entrypoint_candidates": self.entrypoint_candidates(),
            "references": self.reference_graph(),
        }

    def working_directory(self, decision: Decision, workspace: Path) -> Path | None:
        if not decision.working_directory:
            return None
        source_id, separator, relative = decision.working_directory.partition(":")
        if not separator and source_id in self.sources:
            relative = "."
        if source_id not in self.sources:
            raise ValueError("working_directory must be source-id:relative/directory")
        source = self.sources[source_id]
        path = source.resolve_path(relative)
        if not path.is_dir():
            raise ValueError("working_directory does not exist")
        copy = workspace / source_id
        if not copy.exists():
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                source.path,
                copy,
                symlinks=True,
                ignore=shutil.ignore_patterns(".git", ".pheragent"),
            )
        return copy / path.relative_to(source.path)

    def grounds(self, decision: Decision) -> bool:
        evidence = self.existing_refs(decision.evidence)
        if decision.working_directory and decision.command[0].startswith("./"):
            source_id, _, directory = decision.working_directory.partition(":")
            path = (PurePosixPath(directory or ".") / decision.command[0]).as_posix()
            if f"{source_id}:{path}" in evidence:
                return True
        rendered = shlex.join(decision.command)
        plain = " ".join(decision.command)
        for reference in evidence:
            source_id, _, path = reference.partition(":")
            if source_id not in self.sources or reference not in self.readable_paths:
                continue
            content = self.sources[source_id].resolve_path(path).read_text(
                encoding="utf-8", errors="replace"
            )
            if rendered in content or plain in content:
                return True
        return False

    def call(self, decision: Decision) -> dict:
        if decision.tool == "inventory_sources":
            return self.inventory()
        if decision.tool == "search_sources":
            hits = self.search_index.search(RetrievalQuery(terms=(decision.query or "",)), limit=10)
            result = []
            for hit in hits:
                lines = hit.document.text.splitlines()
                start = max(0, hit.line - hit.document.start_line - 2)
                end = min(len(lines), start + 11)
                result.append(
                    {
                        "source": hit.document.file_key,
                        "file_type": self.categories[hit.document.file_key],
                        "line": hit.line,
                        "start_line": hit.document.start_line + start,
                        "end_line": hit.document.start_line + end - 1,
                        "text": "\n".join(lines[start:end])[:2500],
                    }
                )
            return {"hits": result}
        if decision.tool in {"read_file", "list_directory"}:
            identifier, separator, relative = (decision.source_path or "").partition(":")
            if not separator or identifier not in self.sources:
                raise ValueError("source_path must be source-id:relative/path")
            source = self.sources[identifier]
            path = source.resolve_path(relative)
            if decision.tool == "list_directory":
                if not path.is_dir():
                    raise ValueError("source path is not a directory")
                return {"entries": sorted(item.name for item in path.iterdir())[:200]}
            return self.read_file(
                decision.source_path, decision.start_line, decision.end_line
            )
        raise ValueError("unknown source tool")
