from __future__ import annotations

import re

from ..enums import DeterministicFindingKind
from ..evidence import EvidenceStore
from ..models import InventoryEntry
from ..source_manager import AcquiredSource
from .base import ParseResult, make_finding, source_file_path

_HEADING = re.compile(r"^\s{0,3}(?P<marks>#{1,6})\s+(?P<title>.+?)\s*#*\s*$")
_ORDERED_STEP = re.compile(r"^\s*\d+[.)]\s+(?P<step>.+)$")
_FENCE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})\s*(?P<language>[\w+-]*)")
_FILE_REFERENCE = re.compile(r"`(?P<path>[^`\n]*(?:/|\.[A-Za-z0-9]{1,8})[^`\n]*)`")
_SHELL_LANGUAGES = {"bash", "console", "shell", "sh", "zsh"}


class MarkdownParser:
    name = "markdown"

    def parse(
        self,
        source: AcquiredSource,
        entry: InventoryEntry,
        evidence: EvidenceStore,
    ) -> ParseResult:
        lines = source_file_path(source, entry).read_text(encoding="utf-8").splitlines()
        result = ParseResult(parser=self.name)
        fence_start: int | None = None
        fence_marker = ""
        fence_language = ""

        for index, line in enumerate(lines, start=1):
            if fence_start is not None:
                if line.lstrip().startswith(fence_marker):
                    content = "\n".join(lines[fence_start : index - 1]).strip()
                    if content and _is_command_block(fence_language, content):
                        result.findings.append(
                            make_finding(
                                source=source,
                                entry=entry,
                                evidence=evidence,
                                kind=DeterministicFindingKind.COMMAND,
                                name=f"{fence_language or 'fenced'} command block",
                                start_line=fence_start,
                                end_line=index,
                                attributes={"language": fence_language, "command": content},
                            )
                        )
                    fence_start = None
                    fence_marker = ""
                    fence_language = ""
                continue

            fence_match = _FENCE.match(line)
            if fence_match:
                fence_start = index
                fence_marker = fence_match.group("fence")
                fence_language = fence_match.group("language").lower()
                continue

            heading = _HEADING.match(line)
            if heading:
                title = heading.group("title")
                result.findings.append(
                    make_finding(
                        source=source,
                        entry=entry,
                        evidence=evidence,
                        kind=DeterministicFindingKind.DOCUMENT_HEADING,
                        name=title,
                        start_line=index,
                        end_line=index,
                        attributes={"level": len(heading.group("marks"))},
                        heading=title,
                    )
                )

            step = _ORDERED_STEP.match(line)
            if step:
                result.findings.append(
                    make_finding(
                        source=source,
                        entry=entry,
                        evidence=evidence,
                        kind=DeterministicFindingKind.PROCEDURE_STEP,
                        name=step.group("step"),
                        start_line=index,
                        end_line=index,
                    )
                )

            for reference in _FILE_REFERENCE.finditer(line):
                path = reference.group("path").strip()
                if path:
                    result.findings.append(
                        make_finding(
                            source=source,
                            entry=entry,
                            evidence=evidence,
                            kind=DeterministicFindingKind.FILE_REFERENCE,
                            name=path,
                            start_line=index,
                            end_line=index,
                        )
                    )

        if fence_start is not None:
            result.warnings.append(f"unclosed fenced block starting at line {fence_start}")
        return result


def _is_command_block(language: str, content: str) -> bool:
    if language in _SHELL_LANGUAGES:
        return True
    return any(
        line.lstrip().startswith(("$ ", "kubectl ", "helm ", "./")) for line in content.splitlines()
    )
