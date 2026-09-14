from __future__ import annotations

from pathlib import Path

import pytest

from pheragent.cli import _build_parser
from pheragent.deployment.analysis_llm import DEFAULT_ANALYSIS_MODEL
from pheragent.deployment.cli import _analysis_config, _terminal_approval
from pheragent.deployment.enums import AnalysisTreatment
from pheragent.deployment.recovery import RecoveryResolution, RecoveryStatus


@pytest.fixture(autouse=True)
def _isolate_project_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)


def test_deployment_analyze_can_force_fresh_llm_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHERAGENT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
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
    config = _analysis_config(args, tmp_path)
    assert config.treatment == AnalysisTreatment.HYBRID
    assert config.model == DEFAULT_ANALYSIS_MODEL


def test_deployment_run_accepts_one_block_scope(tmp_path: Path) -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "deployment",
            "run",
            str(tmp_path / "deployment-workflow.yaml"),
            "--block",
            "B6",
            "--allow-unready",
        ]
    )

    assert args.block == "B6"
    assert args.allow_unready is True


def test_terminal_approval_offers_only_yes_no_or_view(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    answers = iter(("v", "y"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    resolution = RecoveryResolution(
        failure_id="failure-1",
        step_id="S001",
        status=RecoveryStatus.NEEDS_HUMAN,
        reason="verify the exact image",
        patch="--- a/values.yaml\n+++ b/values.yaml\n",
        approval_items=("image: docker.io/example/minio@sha256:123",),
    )

    assert _terminal_approval(resolution)
    output = capsys.readouterr().out
    assert "docker.io/example/minio@sha256:123" in output
    assert resolution.patch in output


def test_deployment_runtime_inspection_options(tmp_path: Path) -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "deployment",
            "inspect-runtime",
            "--output",
            str(tmp_path / "runtime.json"),
            "--aws-region",
            "us-east-1",
            "--kube-context",
            "mosip",
        ]
    )

    assert args.deployment_command == "inspect-runtime"
    assert args.aws_region == "us-east-1"
    assert args.kube_context == "mosip"


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
