from __future__ import annotations

import re

from ..enums import DeterministicFindingKind
from ..evidence import EvidenceStore
from ..models import InventoryEntry
from ..source_manager import AcquiredSource
from .base import ParseResult, make_finding, source_file_path

_CONTROL_ONLY = re.compile(r"^(?:then|do|done|fi|else|esac|\{|\})\s*;?$")
_VALIDATION_MARKERS = (" health", " ready", "curl ", "kubectl get", "kubectl wait", " test ")


class ShellParser:
    name = "shell"

    def parse(
        self,
        source: AcquiredSource,
        entry: InventoryEntry,
        evidence: EvidenceStore,
    ) -> ParseResult:
        lines = source_file_path(source, entry).read_text(encoding="utf-8").splitlines()
        result = ParseResult(parser=self.name)
        index = 0
        while index < len(lines):
            start = index + 1
            parts = [lines[index]]
            while parts[-1].rstrip().endswith("\\") and index + 1 < len(lines):
                index += 1
                parts.append(lines[index])
            command = "\n".join(parts).strip()
            end = index + 1
            index += 1
            if not command or command.startswith(("#", "#!")) or _CONTROL_ONLY.fullmatch(command):
                continue
            kind = (
                DeterministicFindingKind.VALIDATION
                if _looks_like_validation(command)
                else DeterministicFindingKind.COMMAND
            )
            result.findings.append(
                make_finding(
                    source=source,
                    entry=entry,
                    evidence=evidence,
                    kind=kind,
                    name=command.splitlines()[0][:120],
                    start_line=start,
                    end_line=end,
                    attributes={"command": command},
                )
            )
        return result


def _looks_like_validation(command: str) -> bool:
    normalized = f" {command.lower()} "
    return any(marker in normalized for marker in _VALIDATION_MARKERS)
