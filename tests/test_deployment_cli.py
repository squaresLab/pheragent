from __future__ import annotations

from pathlib import Path

import pytest

from pheragent.cli import _build_parser
from pheragent.deployment.cli import _analysis_config
from pheragent.deployment.enums import AnalysisTreatment


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
    assert _analysis_config(args, tmp_path).treatment == AnalysisTreatment.HYBRID


@pytest.mark.parametrize("research_option", ["--treatment", "--gold"])
def test_deployment_analyze_hides_research_options(
    tmp_path: Path,
    research_option: str,
) -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "deployment",
                "analyze",
                "--context",
                str(tmp_path / "context.yaml"),
                "--output",
                str(tmp_path / "output"),
                research_option,
                "a2" if research_option == "--treatment" else "gold.yaml",
            ]
        )
