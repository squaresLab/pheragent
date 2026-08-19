from __future__ import annotations

from pathlib import Path

import pytest

from pheragent.cli import _build_parser


@pytest.fixture(autouse=True)
def _isolate_project_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def test_deployment_analyze_can_force_fresh_llm_requests(tmp_path: Path) -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "deployment",
            "analyze",
            "--context",
            str(tmp_path / "context.yaml"),
            "--output",
            str(tmp_path / "output"),
            "--refresh-llm",
            "--llm-reasoning-effort",
            "low",
        ]
    )

    assert args.refresh_llm is True
    assert args.llm_reasoning_effort == "low"
