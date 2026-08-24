from __future__ import annotations

from pathlib import Path
from typing import Any

from .analyzer import AnalysisResult
from .output import AtomicOutputTransaction
from .report import render_analysis_report
from .serialization import write_json, write_text, write_yaml


def publish_analysis_artifacts(
    run_dir: Path,
    result: AnalysisResult,
    *,
    debug: bool = False,
) -> tuple[Path, ...]:
    """Publish the stable Phase 1 artifact contract for product and research callers."""
    with AtomicOutputTransaction(run_dir, write_manifest=False) as transaction:
        staging = transaction.staging_dir
        write_yaml(staging / "functional-blocks.yaml", result.document)
        write_text(staging / "analysis-report.md", render_analysis_report(result))
        write_yaml(staging / "deployment-workflow.yaml", result.workflow)
        if result.knowledge_graph is not None:
            write_json(staging / "knowledge-graph.json", result.knowledge_graph)
        if debug:
            _write_debug_artifacts(staging / "debug", result)
        return transaction.commit()


def analysis_metrics(result: AnalysisResult) -> dict[str, Any]:
    """Return comparison metrics derived from one completed analysis result."""
    metrics = result.document.evaluation.model_dump(mode="json")
    metrics.update(
        {
            "evidence_observations": len(result.investigation.observations),
            "evidence_characters": result.evidence_characters,
            "retrieval_queries": len(result.retrieval_queries),
            "graph_queries": len(result.graph_queries),
            "graph_nodes": len(result.knowledge_graph.nodes) if result.knowledge_graph else 0,
            "graph_edges": len(result.knowledge_graph.edges) if result.knowledge_graph else 0,
            "graph_gaps_before_retrieval": (
                len(result.knowledge_graph.gaps) if result.knowledge_graph else 0
            ),
        }
    )
    return metrics


def _write_debug_artifacts(debug: Path, result: AnalysisResult) -> None:
    write_json(debug / "repository-index.json", list(result.repository_files))
    write_json(
        debug / "reference-graph.json",
        {
            "nodes": [item.model_dump(mode="json") for item in result.reference_nodes],
            "relations": [item.model_dump(mode="json") for item in result.reference_relations],
        },
    )
    write_json(debug / "component-candidates.json", result.signals.candidate_components)
    write_json(debug / "deployment-signals.json", result.signals)
    write_yaml(debug / "investigation-plan-input.yaml", result.investigation.plan_input)
    write_yaml(debug / "investigation-plan.yaml", result.investigation.plan)
    write_json(debug / "investigation-evidence.json", list(result.investigation.observations))
    write_yaml(
        debug / "investigation-synthesis-input.yaml",
        result.investigation.synthesis_input,
    )
    if result.investigation.synthesis is not None:
        write_yaml(debug / "investigation-synthesis.yaml", result.investigation.synthesis)
