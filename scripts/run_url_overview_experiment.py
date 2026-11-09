"""Compare deployment overviews produced from repository URLs alone."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from pheragent.deployment.llm import LLMClient

REPOSITORIES = (
    "https://github.com/mosip/infra",
    "https://github.com/apache/airflow",
    "https://github.com/open-telemetry/opentelemetry-demo",
)

INSTRUCTIONS = """You are a senior DevOps engineer forming a tentative deployment overview.
You receive only a repository URL. You cannot browse, inspect the repository, or use tools.
Use prior knowledge and what the URL itself implies. Do not invent exact files, commands,
versions, or runtime state. Produce two to eight ordered semantic stages with a few high-level
steps each. State important uncertainties explicitly and use low confidence when appropriate.
This is an overview for later investigation, not an executable deployment plan.
"""


class Stage(BaseModel):
    name: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    steps: list[str] = Field(min_length=1, max_length=5)


class Overview(BaseModel):
    system: str = Field(min_length=1)
    likely_route: str = Field(min_length=1)
    confidence: Literal["low", "medium", "high"]
    stages: list[Stage] = Field(min_length=2, max_length=8)
    uncertainties: list[str] = Field(max_length=10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required")

    output = args.output or Path(".pheragent/experiments") / (
        "url-overview-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    client = LLMClient(
        model=args.model,
        max_output_tokens=4000,
        reasoning_effort=args.reasoning_effort,
    )
    records = []
    for url in REPOSITORIES:
        payload = {"repository_url": url}
        started = time.monotonic()
        overview, usage = client.complete(
            Overview,
            instructions=INSTRUCTIONS,
            payload=payload,
        )
        slug = url.rstrip("/").rsplit("/", 1)[-1]
        record = {
            "treatment": "url-only-no-tools",
            "repository_url": url,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "status": "complete",
            "usage": usage,
            "visible_input_characters": len(INSTRUCTIONS)
            + len(json.dumps(payload, ensure_ascii=False)),
            "payload_tokens_estimate": max(1, len(json.dumps(payload)) // 4),
            "duration_seconds": round(time.monotonic() - started, 6),
            "overview": overview.model_dump(mode="json"),
        }
        (output / f"{slug}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        records.append(record)
        print(
            f"{slug}: complete; "
            f"input={usage.get('input_tokens', 0)}; "
            f"output={usage.get('output_tokens', 0)}"
        )

    usage = {
        key: sum(int(record["usage"].get(key, 0)) for record in records)
        for key in (
            "requests",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        )
    }
    summary = {
        "treatment": "url-only-no-tools",
        "limitation": "The model could not inspect repository contents or current documentation.",
        "usage": usage,
        "results": records,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"run: {output}")
    print(f"usage: {usage}")
    return 0 if all(record["overview"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
