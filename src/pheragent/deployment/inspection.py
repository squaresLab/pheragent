from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .evidence import EvidenceStore
from .inventory import RepositoryInventoryBuilder, mark_inspected
from .models import DeterministicFinding, EvidenceRecord, RepositoryInventory
from .parsers import DeterministicParserRegistry
from .serialization import load_sources_config, write_json, write_jsonl
from .source_manager import AcquisitionResult, SourceManager


@dataclass(frozen=True, slots=True)
class DeterministicInspectionResult:
    acquisition: AcquisitionResult
    inventory: RepositoryInventory
    findings: tuple[DeterministicFinding, ...]
    evidence: tuple[EvidenceRecord, ...]

    @property
    def evidence_count(self) -> int:
        return len(self.evidence)


def run_deterministic_inspection(
    *,
    sources_path: Path,
    output_dir: Path,
    strict: bool,
    timeout: float = 900.0,
    cache_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> DeterministicInspectionResult:
    resolved_sources_path = sources_path.expanduser().resolve()
    resolved_output_dir = output_dir.expanduser().resolve()
    config = load_sources_config(resolved_sources_path)
    notify = progress or (lambda _message: None)
    manager = SourceManager(
        cache_dir=cache_dir or resolved_output_dir / ".source-cache",
        config_dir=resolved_sources_path.parent,
        strict=strict,
        timeout=timeout,
        progress=notify,
    )
    acquisition = manager.acquire(config)
    notify("building repository inventory")
    inventory = RepositoryInventoryBuilder().build(acquisition.sources)
    evidence = EvidenceStore()
    registry = DeterministicParserRegistry()
    sources_by_id = {source.id: source for source in acquisition.sources}
    inspected_entries = []
    findings_by_id: dict[str, DeterministicFinding] = {}

    selected_count = sum(entry.selected for entry in inventory.entries)
    notify(f"inspecting {selected_count} selected file(s)")
    selected_seen = 0
    for entry in inventory.entries:
        if not entry.selected:
            inspected_entries.append(entry)
            continue
        selected_seen += 1
        source = sources_by_id[entry.source_id]
        try:
            parsed = registry.parse(source, entry, evidence)
        except (OSError, UnicodeError, ValueError) as exc:
            inspected_entries.append(mark_inspected(entry, parser="failed", warnings=[str(exc)]))
            if selected_seen % 50 == 0 or selected_seen == selected_count:
                notify(f"inspected selected file {selected_seen}/{selected_count}")
            continue
        inspected_entries.append(
            mark_inspected(entry, parser=parsed.parser, warnings=parsed.warnings)
        )
        for finding in parsed.findings:
            existing = findings_by_id.get(finding.id)
            if existing is not None and existing != finding:
                raise ValueError(f"deterministic finding ID collision: {finding.id}")
            findings_by_id[finding.id] = finding
        if selected_seen % 50 == 0 or selected_seen == selected_count:
            notify(f"inspected selected file {selected_seen}/{selected_count}")

    completed_inventory = RepositoryInventory(
        inventory_version=inventory.inventory_version,
        detected_technologies=inventory.detected_technologies,
        entries=inspected_entries,
    )
    findings = tuple(
        sorted(
            findings_by_id.values(),
            key=lambda item: (item.source_id, item.path, item.kind, item.name, item.id),
        )
    )
    write_json(resolved_output_dir / "source-manifest.json", acquisition.manifest)
    write_json(resolved_output_dir / "repository-inventory.json", completed_inventory)
    evidence.write(resolved_output_dir / "evidence.jsonl")
    write_jsonl(resolved_output_dir / "deterministic-findings.jsonl", findings)
    return DeterministicInspectionResult(
        acquisition=acquisition,
        inventory=completed_inventory,
        findings=findings,
        evidence=tuple(evidence.records),
    )
