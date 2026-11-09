"""Build deployment overviews from repository structure and selected full files."""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from pheragent.deployment.llm import LLMClient
from pheragent.deployment.models import SourceKind, SourcesConfig, SourceSpec
from pheragent.deployment.overview import create_overview
from pheragent.deployment.serialization import write_json, write_yaml
from pheragent.deployment.source_manager import SourceManager
from pheragent.deployment.sources import SourceTools

DEFAULT_PROJECTS = {
    "mosip": (
        "https://github.com/mosip/infra.git",
        "Deploy MOSIP using the repository's supported deployment route.",
        "kubernetes",
    ),
    "otel": (
        "https://github.com/open-telemetry/opentelemetry-demo.git",
        "Deploy the OpenTelemetry Demo using its supported default route.",
        "shell",
    ),
    "airflow": (
        "https://github.com/apache/airflow.git",
        "Deploy Apache Airflow using its supported production route.",
        "kubernetes",
    ),
}


def _repositories(values: list[str]) -> dict[str, list[str]]:
    repositories: dict[str, list[str]] = defaultdict(list)
    for value in values:
        name, separator, url = value.partition("=")
        if not separator or not name or not url:
            raise argparse.ArgumentTypeError("repositories must use NAME=URL")
        repositories[name].append(url)
    return dict(repositories)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        action="append",
        default=[],
        metavar="NAME=URL",
        help="repository source; repeat NAME to add related repositories",
    )
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required")

    supplied = _repositories(args.repository)
    projects = supplied or {
        name: [url] for name, (url, _objective, _target) in DEFAULT_PROJECTS.items()
    }
    output = args.output or Path(".pheragent/research") / (
        "repository-overview-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    client = LLMClient(
        model=args.model,
        max_output_tokens=5000,
        reasoning_effort=args.reasoning_effort,
    )
    results = []
    for name, urls in projects.items():
        project_dir = output / name
        project_dir.mkdir()
        default = DEFAULT_PROJECTS.get(name)
        objective = default[1] if default else f"Deploy {name}."
        target = {"type": default[2] if default else "unspecified"}
        try:
            acquired = SourceManager(
                cache_dir=output.parent / ".source-cache",
                config_dir=Path.cwd(),
                strict=False,
                progress=lambda message, project=name: print(f"{project}: {message}"),
            ).acquire(
                SourcesConfig(
                    system=name,
                    sources=[
                        SourceSpec(
                            id=f"repository-{index}",
                            kind=SourceKind.GIT,
                            location=url,
                        )
                        for index, url in enumerate(urls, start=1)
                    ],
                )
            )
            sources = SourceTools(acquired.sources)
            inventory = sources.inventory()
            write_json(project_dir / "sources.json", acquired.manifest)
            write_json(project_dir / "inventory.json", inventory)
            overview, usage = create_overview(
                client,
                objective=objective,
                target=target,
                sources=sources,
            )
            documents = [sources.read_file(path) for path in overview.route_evidence]
            write_json(project_dir / "selected-files.json", documents)
            write_yaml(project_dir / "overview.yaml", overview)
            result = {
                "project": name,
                "status": "complete",
                "objective": objective,
                "target": target,
                "route": overview.route,
                "selected_files": overview.route_evidence,
                "selected_characters": sum(len(item.get("text", "")) for item in documents),
                "usage": usage,
            }
        except Exception as exc:
            result = {"project": name, "status": "failed", "error": str(exc), "usage": {}}
        write_json(project_dir / "result.json", result)
        results.append(result)
        print(f"{name}: {result['status']}; usage={result['usage']}")

    summary = {
        "treatment": "tree-links-select-full-files-overview",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "results": results,
    }
    write_json(output / "summary.json", summary)
    print(f"run: {output}")
    return 0 if all(result["status"] == "complete" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
