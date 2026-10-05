"""Acquire and read source evidence used by deployment decisions."""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path, PurePosixPath

from .enums import SourceKind
from .inventory import RepositoryInventoryBuilder
from .models import SourceSpec
from .redaction import redact_secrets
from .retrieval import DeploymentRetrievalEngine, RetrievalQuery
from .source_manager import AcquiredSource
from .task import Decision, SourceLocation


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

    def existing_refs(self, references: list[str]) -> set[str]:
        return {
            path
            for path in self.readable_paths
            if any(ref == path or ref.startswith((path + ":", path + " —")) for ref in references)
        }

    def call(self, decision: Decision) -> dict:
        if decision.tool == "inventory_sources":
            return _source_tree(self.paths)
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
            if decision.source_path not in self.readable_paths:
                raise ValueError("source file is not in the readable inventory")
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(1, decision.start_line or 1)
            end = min(len(lines), decision.end_line or start + 119, start + 119)
            return {
                "source": decision.source_path,
                "start_line": start,
                "end_line": end,
                "text": redact_secrets("\n".join(lines[start - 1 : end])),
            }
        raise ValueError("unknown source tool")


def _source_cwd(decision: Decision, sources: SourceTools, workspace: Path) -> Path | None:
    if not decision.working_directory:
        return None
    source_id, separator, relative = decision.working_directory.partition(":")
    if not separator and source_id in sources.sources:
        relative = "."
    if source_id not in sources.sources:
        raise ValueError("working_directory must be source-id:relative/directory")
    source = sources.sources[source_id]
    path = source.resolve_path(relative)
    if not path.is_dir():
        raise ValueError("working_directory does not exist")
    copy = workspace / source_id
    if not copy.exists():
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source.path, copy, symlinks=True, ignore=shutil.ignore_patterns(".git", ".pheragent")
        )
    return copy / path.relative_to(source.path)


def _source_grounded(decision: Decision, sources: SourceTools) -> bool:
    evidence = sources.existing_refs(decision.evidence)
    if decision.working_directory and decision.command[0].startswith("./"):
        source_id, _, directory = decision.working_directory.partition(":")
        path = (PurePosixPath(directory or ".") / decision.command[0]).as_posix()
        if f"{source_id}:{path}" in evidence:
            return True
    rendered = shlex.join(decision.command)
    plain = " ".join(decision.command)
    for ref in evidence:
        source_id, _, path = ref.partition(":")
        if source_id not in sources.sources or ref not in sources.readable_paths:
            continue
        content = (
            sources.sources[source_id]
            .resolve_path(path)
            .read_text(encoding="utf-8", errors="replace")
        )
        if rendered in content or plain in content:
            return True
    return False
