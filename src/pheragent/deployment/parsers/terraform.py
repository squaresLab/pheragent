from __future__ import annotations

import re

from ..enums import DeterministicFindingKind
from ..evidence import EvidenceStore
from ..models import InventoryEntry
from ..source_manager import AcquiredSource
from .base import ParseResult, make_finding, source_file_path

_BLOCK = re.compile(
    r'^\s*(?P<kind>resource|data|module|output|variable)\s+"(?P<name>[^"\n]+)"'
    r'(?:\s+"(?P<instance>[^"\n]+)")?\s*\{'
)
_DEPENDS_ON = re.compile(r"\bdepends_on\s*=\s*(?P<value>.+)")


class TerraformParser:
    name = "terraform"

    def parse(
        self,
        source: AcquiredSource,
        entry: InventoryEntry,
        evidence: EvidenceStore,
    ) -> ParseResult:
        lines = source_file_path(source, entry).read_text(encoding="utf-8").splitlines()
        result = ParseResult(parser=self.name)
        for line_number, line in enumerate(lines, start=1):
            block = _BLOCK.match(line)
            if block:
                block_kind = block.group("kind")
                name = block.group("name")
                instance = block.group("instance")
                display_name = ".".join(value for value in (block_kind, name, instance) if value)
                finding_kind = (
                    DeterministicFindingKind.OUTPUT
                    if block_kind == "output"
                    else DeterministicFindingKind.RESOURCE
                )
                result.findings.append(
                    make_finding(
                        source=source,
                        entry=entry,
                        evidence=evidence,
                        kind=finding_kind,
                        name=display_name,
                        start_line=line_number,
                        end_line=line_number,
                        attributes={"terraform_kind": block_kind},
                    )
                )
            dependency = _DEPENDS_ON.search(line)
            if dependency:
                result.findings.append(
                    make_finding(
                        source=source,
                        entry=entry,
                        evidence=evidence,
                        kind=DeterministicFindingKind.DEPENDENCY,
                        name="terraform depends_on",
                        start_line=line_number,
                        end_line=line_number,
                        attributes={"depends_on": dependency.group("value").strip()},
                    )
                )
        return result
