from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import yaml

from pheragent.deployment.serialization import write_text


def summarize_study(study_root: Path) -> None:
    """Create a compact comparison without modifying any sealed run directory."""
    rows = _load_rows(study_root)
    _write_csv(study_root / "results.csv", rows)
    write_text(study_root / "results.md", _render_summary(rows))
    write_text(study_root / "error-analysis.md", _render_error_analysis(study_root, rows))


def _load_rows(study_root: Path) -> list[dict[str, Any]]:
    rows = []
    for manifest_path in sorted((study_root / "runs").glob("*/run-manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics_path = manifest_path.parent / "metrics.json"
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.is_file()
            else {}
        )
        inputs = manifest.get("inputs", {})
        usage = manifest.get("llm", {}).get("usage", {})
        rows.append(
            {
                "run_id": manifest["run_id"],
                "case": inputs.get("case", "unknown"),
                "treatment": manifest.get("analysis_method", manifest.get("treatment")),
                "repetition": inputs.get("repetition", 0),
                "status": manifest["status"],
                "component_count": metrics.get("component_count"),
                "component_precision": metrics.get("component_precision"),
                "component_recall": metrics.get("component_recall"),
                "entrypoint_accuracy": metrics.get("entrypoint_accuracy"),
                "executable_route_coverage": metrics.get("executable_route_coverage"),
                "edge_precision": metrics.get("edge_precision"),
                "edge_recall": metrics.get("edge_recall"),
                "hallucination_rate": metrics.get("hallucination_rate"),
                "evidence_characters": metrics.get("evidence_characters"),
                "graph_gaps_before_retrieval": metrics.get("graph_gaps_before_retrieval"),
                "llm_requests": usage.get("requests", metrics.get("llm_requests", 0)),
                "input_tokens": usage.get("input_tokens", metrics.get("llm_input_tokens", 0)),
                "output_tokens": usage.get("output_tokens", metrics.get("llm_output_tokens", 0)),
                "duration_seconds": manifest.get("duration_seconds"),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["run_id", "case", "treatment", "status"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_summary(rows: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["case"]), str(row["treatment"]))].append(row)
    lines = [
        "# Graph retrieval pilot results",
        "",
        "| Case | Treatment | Runs passed | Components | Route coverage | Recall | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (case, treatment), group in sorted(grouped.items()):
        passed = sum(row["status"] == "completed" for row in group)
        lines.append(
            f"| {case} | {treatment} | {passed}/{len(group)} | "
            f"{_median(group, 'component_count')} | "
            f"{_median(group, 'executable_route_coverage')} | "
            f"{_median(group, 'component_recall')} | "
            f"{_token_median(group)} |"
        )
    return "\n".join(lines) + "\n"


def _render_error_analysis(study_root: Path, rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Qualitative error analysis",
        "",
        "Review each A1/A2 difference against pinned source evidence before assigning a verdict.",
        "",
    ]
    by_case: dict[str, dict[str, Path]] = defaultdict(dict)
    for row in rows:
        if row["status"] != "completed":
            continue
        run_dir = study_root / "runs" / str(row["run_id"])
        by_case[str(row["case"])][str(row["treatment"])] = run_dir
    for case, treatments in sorted(by_case.items()):
        if "a1" not in treatments or "a2" not in treatments:
            continue
        a1 = _component_names(treatments["a1"] / "functional-blocks.yaml")
        a2 = _component_names(treatments["a2"] / "functional-blocks.yaml")
        lines.extend(
            [
                f"## {case}",
                "",
                f"- Added by A2: {', '.join(sorted(a2 - a1)) or 'none'}",
                f"- Removed by A2: {', '.join(sorted(a1 - a2)) or 'none'}",
                "- Verdict: pending human source review",
                "",
            ]
        )
    return "\n".join(lines)


def _component_names(path: Path) -> set[str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        str(component["name"])
        for block in payload.get("blocks", [])
        for component in block.get("components", [])
    }


def _median(rows: list[dict[str, Any]], field: str) -> str:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return "n/a"
    value = median(values)
    return f"{value:.3f}" if not value.is_integer() else str(int(value))


def _token_median(rows: list[dict[str, Any]]) -> str:
    values = [
        int(row.get("input_tokens") or 0) + int(row.get("output_tokens") or 0)
        for row in rows
    ]
    return str(int(median(values))) if values else "n/a"
