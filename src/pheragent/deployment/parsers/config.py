from __future__ import annotations

import re

from ..enums import DeterministicFindingKind
from ..evidence import EvidenceStore
from ..models import InventoryEntry
from ..source_manager import AcquiredSource
from .base import ParseResult, make_finding, source_file_path

_ASSIGNMENT = re.compile(r"^\s*(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_.-]*)\s*[:=]")


class ConfigurationParser:
    name = "configuration"

    def parse(
        self,
        source: AcquiredSource,
        entry: InventoryEntry,
        evidence: EvidenceStore,
    ) -> ParseResult:
        lines = source_file_path(source, entry).read_text(encoding="utf-8").splitlines()
        result = ParseResult(parser=self.name)
        for line_number, line in enumerate(lines, start=1):
            if not line.strip() or line.lstrip().startswith(("#", ";")):
                continue
            assignment = _ASSIGNMENT.match(line)
            if not assignment:
                continue
            result.findings.append(
                make_finding(
                    source=source,
                    entry=entry,
                    evidence=evidence,
                    kind=DeterministicFindingKind.CONFIGURATION,
                    name=assignment.group("key"),
                    start_line=line_number,
                    end_line=line_number,
                    attributes={"key": assignment.group("key")},
                )
            )
        return result
