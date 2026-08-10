from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .analysis_llm import build_compact_classification_input
from .analyzer import AnalysisConfig, render_analysis_report, run_repository_analysis
from .artifact_writer import write_artifact_outputs
from .explain import find_block, render_block_explanation
from .fact_extractor import FactExtractorConfig, run_fact_extraction
from .inspection import run_deterministic_inspection
from .output import AtomicOutputTransaction, create_timestamped_run_directory
from .progress import InspectionProgress
from .serialization import load_deployment_artifact, write_json, write_text, write_yaml
from .synthesis import synthesize_artifact
from .validation import validate_artifact


def add_deployment_parser(subparsers: Any) -> None:
    deployment = subparsers.add_parser(
        "deployment",
        help="Inspect deployment sources and review deployment artifacts.",
    )
    commands = deployment.add_subparsers(dest="deployment_command", required=True)

    analyze = commands.add_parser(
        "analyze",
        help="Discover a compact functional deployment block graph without executing sources.",
    )
    analyze.add_argument("--repo", action="append", default=[])
    analyze.add_argument("--docs", action="append", default=[])
    analyze.add_argument("--context", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument(
        "--run-name",
        default=None,
        help="Label for the timestamped run folder; defaults to the context system name.",
    )
    analyze.add_argument("--gold", type=Path, default=None)
    analyze.add_argument("--node-budget", type=_positive_int, default=200)
    analyze.add_argument("--source-timeout", type=float, default=900.0)
    analyze.add_argument("--strict", action="store_true")
    analyze.add_argument(
        "--synthesizer",
        choices=("auto", "deterministic", "llm"),
        default="auto",
        help=(
            "Use one compact LLM classification request when available, or deterministic "
            "classification."
        ),
    )
    analyze.add_argument("--model", default=None)
    analyze.add_argument("--openai-base-url", default=None)
    analyze.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    analyze.add_argument("--openai-base-url-env", default="OPENAI_BASE_URL")
    analyze.add_argument("--llm-timeout", type=float, default=120.0)
    analyze.add_argument("--llm-max-tokens", type=_positive_int, default=3000)
    analyze.add_argument(
        "--retry-failed-llm",
        action="store_true",
        help="Retry an identical request recorded in the failed-LLM history.",
    )
    analyze.add_argument(
        "--debug",
        action="store_true",
        help="Write detailed repository and signal artifacts beneath debug/.",
    )

    inspect = commands.add_parser(
        "inspect",
        help="Inspect deployment sources without executing them.",
    )
    inspect.add_argument("--sources", required=True, type=Path)
    inspect.add_argument("--output", required=True, type=Path)
    inspect.add_argument("--strict", action="store_true")
    inspect.add_argument("--source-timeout", type=float, default=900.0)
    inspect.add_argument(
        "--extractor",
        choices=("auto", "deterministic", "llm"),
        default="auto",
        help="Fact extractor. auto uses the LLM only when an API key is present.",
    )
    inspect.add_argument(
        "--llm-api",
        choices=("responses", "chat-completions"),
        default="responses",
    )
    inspect.add_argument("--model", default=None)
    inspect.add_argument("--openai-base-url", default=None)
    inspect.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    inspect.add_argument("--openai-base-url-env", default="OPENAI_BASE_URL")
    inspect.add_argument("--llm-timeout", type=float, default=120.0)
    inspect.add_argument("--llm-max-tokens", type=int, default=4096)
    inspect.add_argument(
        "--llm-max-requests",
        type=_non_negative_int,
        default=25,
        help=(
            "Hard cap on OpenAI requests for fact extraction, including retries. "
            "Default: 25; use 0 for no LLM requests."
        ),
    )
    inspect.add_argument("--llm-retries", type=int, default=3)
    inspect.add_argument("--llm-retry-delay", type=float, default=1.0)
    inspect.add_argument("--chunk-max-chars", type=int, default=12_000)

    validate = commands.add_parser("validate", help="Validate a deployment artifact.")
    validate.add_argument("artifact", type=Path)
    validate.add_argument("--json", action="store_true")

    explain = commands.add_parser("explain", help="Explain one deployment artifact block.")
    explain.add_argument("artifact", type=Path)
    explain.add_argument("--block", required=True)
    explain.add_argument("--json", action="store_true")


def run_deployment_command(args: argparse.Namespace) -> int:
    try:
        if args.deployment_command == "analyze":
            return _run_analyze(args)
        if args.deployment_command == "inspect":
            return _run_inspect(args)
        if args.deployment_command == "validate":
            return _run_validate(args)
        if args.deployment_command == "explain":
            return _run_explain(args)
        raise ValueError(f"unsupported deployment command: {args.deployment_command}")
    except (OSError, RuntimeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        print(f"deployment error: {exc}", file=sys.stderr)
        return 1


def _run_analyze(args: argparse.Namespace) -> int:
    output_root = args.output.expanduser().resolve()
    run_name = args.run_name or _context_system_name(args.context)
    run_dir = create_timestamped_run_directory(output_root, name=run_name)

    def progress(message: str) -> None:
        print(f"analyze: {message}", file=sys.stderr, flush=True)

    try:
        result = run_repository_analysis(
            AnalysisConfig(
                repositories=args.repo,
                documentation=args.docs,
                context_path=args.context,
                cache_dir=output_root / ".source-cache",
                node_budget=args.node_budget,
                strict=args.strict,
                source_timeout=args.source_timeout,
                gold_path=args.gold,
                synthesizer=args.synthesizer,
                model=args.model
                or os.getenv("PHERAGENT_MODEL")
                or os.getenv("OPENAI_MODEL")
                or "gpt-4o-mini",
                api_key_env=args.openai_api_key_env,
                base_url_env=args.openai_base_url_env,
                base_url=args.openai_base_url,
                llm_timeout=args.llm_timeout,
                llm_max_output_tokens=args.llm_max_tokens,
                llm_cache_dir=output_root / ".llm-cache",
                retry_failed_llm=args.retry_failed_llm,
            ),
            progress=progress,
        )
        with AtomicOutputTransaction(run_dir, write_manifest=False) as transaction:
            write_yaml(transaction.staging_dir / "functional-blocks.yaml", result.document)
            write_text(
                transaction.staging_dir / "analysis-report.md",
                render_analysis_report(result),
            )
            if args.debug:
                debug = transaction.staging_dir / "debug"
                write_json(debug / "repository-index.json", list(result.repository_files))
                write_json(
                    debug / "reference-graph.json",
                    {
                        "nodes": [item.model_dump(mode="json") for item in result.reference_nodes],
                        "relations": [
                            item.model_dump(mode="json")
                            for item in result.reference_relations
                        ],
                    },
                )
                write_json(
                    debug / "component-candidates.json",
                    result.signals.candidate_components,
                )
                write_json(debug / "deployment-signals.json", result.signals)
                write_yaml(
                    debug / "llm-input.yaml",
                    build_compact_classification_input(result.signals),
                )
            published = transaction.commit()
    except Exception:
        progress(f"failed; run directory retained at {run_dir}")
        raise

    evaluation = result.document.evaluation
    print(f"run: {run_dir}")
    print(f"functional blocks: {run_dir / 'functional-blocks.yaml'}")
    print(f"analysis report: {run_dir / 'analysis-report.md'}")
    print(f"synthesizer: {result.used_synthesizer}")
    print(f"LLM requests this run: {evaluation.llm_requests}")
    if result.llm_failure_history_path:
        print(f"LLM failure history: {result.llm_failure_history_path}")
    print(
        f"components: {evaluation.component_count}; "
        f"deployability coverage: {evaluation.deployability_coverage:.1%}; "
        f"forbidden components: {evaluation.forbidden_component_count}"
    )
    print(f"published files: {len(published)}")
    return 0


def _context_system_name(context_path: Path) -> str:
    payload = yaml.safe_load(context_path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("system"), str):
        raise ValueError("deployment context must define a system name")
    system = payload["system"].strip()
    if not system:
        raise ValueError("deployment context system name cannot be empty")
    return system


def _run_inspect(args: argparse.Namespace) -> int:
    output_dir = args.output.expanduser().resolve()
    with InspectionProgress(output_dir) as progress:
        try:
            with AtomicOutputTransaction(output_dir) as transaction:
                progress.phase(1, "Acquiring pinned deployment sources")
                result = run_deterministic_inspection(
                    sources_path=args.sources,
                    output_dir=transaction.staging_dir,
                    cache_dir=output_dir / ".source-cache",
                    strict=args.strict,
                    timeout=args.source_timeout,
                    progress=progress.detail,
                )
                progress.phase(
                    2,
                    f"Inventory and deterministic inspection complete: "
                    f"{len(result.findings)} finding(s), "
                    f"{result.evidence_count} evidence record(s)",
                )
                progress.phase(3, "Preparing evidence chunks and deterministic facts")
                extraction = run_fact_extraction(
                    result,
                    output_dir=transaction.staging_dir,
                    config=FactExtractorConfig(
                        extractor=args.extractor,
                        model=args.model
                        or os.getenv("PHERAGENT_MODEL")
                        or os.getenv("OPENAI_MODEL")
                        or "gpt-5.5",
                        api_mode=args.llm_api,
                        api_key_env=args.openai_api_key_env,
                        base_url_env=args.openai_base_url_env,
                        base_url=args.openai_base_url,
                        timeout=args.llm_timeout,
                        max_tokens=args.llm_max_tokens,
                        max_requests=args.llm_max_requests,
                        max_retries=args.llm_retries,
                        retry_delay_s=args.llm_retry_delay,
                        chunk_max_chars=args.chunk_max_chars,
                    ),
                    progress=progress.detail,
                )
                progress.phase(
                    4,
                    f"Fact extraction complete: {len(extraction.facts)} normalized fact(s) "
                    f"using {extraction.report.used_extractor}; "
                    f"{extraction.report.llm_requests_made}/"
                    f"{extraction.report.max_llm_requests} LLM request(s) used",
                )
                progress.phase(5, "Synthesizing semantic deployment blocks")
                synthesis = synthesize_artifact(result, extraction)
                progress.detail(
                    f"synthesized {len(synthesis.artifact.blocks)} block(s) with "
                    f"{sum(len(block.components) for block in synthesis.artifact.blocks)} "
                    "component(s)"
                )
                progress.phase(
                    6,
                    f"Dependency graph built: {len(synthesis.graph.edges)} edge(s), "
                    f"{len(synthesis.graph.unmatched_capabilities)} unmatched capability(ies)",
                )
                progress.phase(7, "Validating artifact and rendering inspection report")
                written = write_artifact_outputs(
                    output_dir=transaction.staging_dir,
                    inspection=result,
                    extraction=extraction,
                    synthesis=synthesis,
                    strict=args.strict,
                )
                progress.detail(
                    f"artifact validation: "
                    f"{'valid' if written.validation.valid else 'invalid'} with "
                    f"{len(written.validation.findings)} finding(s)"
                )
                progress.phase(8, "Publishing generated outputs atomically")
                published = transaction.commit()
                if written.validation.valid:
                    progress.complete(f"published {len(published)} file(s) to {output_dir}")
                else:
                    progress.failed(
                        f"published {len(published)} review file(s), but artifact validation failed"
                    )
        except Exception as exc:
            progress.failed(str(exc))
            raise

    print(f"source acquisition complete: {len(result.acquisition.sources)} source(s)")
    for source in result.acquisition.sources:
        revision = source.manifest.resolved_revision or source.manifest.content_hash
        print(f"- {source.id}: {revision}")
    print(f"manifest: {output_dir / 'source-manifest.json'}")
    print(f"inventory: {output_dir / 'repository-inventory.json'}")
    print(f"evidence: {output_dir / 'evidence.jsonl'}")
    print(f"deterministic findings: {len(result.findings)}")
    print(
        f"facts: {len(extraction.facts)} "
        f"({extraction.report.used_extractor}; {extraction.report.chunk_count} chunk(s))"
    )
    print(
        f"LLM requests: {extraction.report.llm_requests_made}/"
        f"{extraction.report.max_llm_requests}; ranked chunks selected: "
        f"{extraction.report.selected_chunk_count}/{extraction.report.chunk_count}"
    )
    print(f"facts sidecar: {output_dir / 'facts.jsonl'}")
    print(f"deployment artifact: {output_dir / 'deployment-artifact.yaml'}")
    print(f"dependency graph: {output_dir / 'dependency-graph.json'}")
    print(f"inspection report: {output_dir / 'inspection-report.md'}")
    print(f"inspection log: {output_dir / 'inspection.log'}")
    for warning in extraction.report.warnings:
        print(f"extraction warning: {warning}", file=sys.stderr)
    return 0 if written.validation.valid else 1


def _run_validate(args: argparse.Namespace) -> int:
    artifact = load_deployment_artifact(args.artifact)
    result = validate_artifact(artifact)
    if args.json:
        print(json.dumps(result.model_dump(mode="json", exclude_none=True), indent=2))
    else:
        status = "valid" if result.valid else "invalid"
        print(f"{status} deployment artifact: {args.artifact}")
        print(f"artifact id: {artifact.metadata.artifact_id}")
        print(f"blocks: {len(artifact.blocks)}")
        for finding in result.findings:
            location = f" [{finding.location}]" if finding.location else ""
            print(f"{finding.severity}: {finding.code}{location}: {finding.message}")
    return 0 if result.valid else 1


def _run_explain(args: argparse.Namespace) -> int:
    artifact = load_deployment_artifact(args.artifact)
    block = find_block(artifact, args.block)
    if args.json:
        print(json.dumps(block.model_dump(mode="json", exclude_none=True), indent=2))
    else:
        print(render_block_explanation(block), end="")
    return 0


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed
