from __future__ import annotations

import json
from pathlib import Path

from pheragent.cli import main
from pheragent.evaluation.recovery import evaluate_recovery


def test_recovery_evaluation_uses_recorded_execution_evidence(tmp_path: Path) -> None:
    run = tmp_path / "execution-run"
    run.mkdir()
    (run / "execution.json").write_text(
        json.dumps(
            {
                "completed": ["S001", "S004"],
                "failed": [],
                "skipped": [],
                "attempts": [
                    {"step_id": "S001", "attempt": 1, "status": "failed"},
                    {"step_id": "S004", "attempt": 1, "status": "completed"},
                    {"step_id": "S001", "attempt": 2, "status": "completed"},
                ],
                "recoveries": [
                    {
                        "step_id": "S001",
                        "status": "resolved",
                        "usage": {"input_tokens": 100, "output_tokens": 20},
                        "validation": {"succeeded": True},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "failure-bundle.json").write_text(
        json.dumps({"failures": [{"id": "S001-attempt-1", "step_id": "S001"}]}),
        encoding="utf-8",
    )
    (run / "run-manifest.json").write_text(
        json.dumps({"run_id": "test-run", "duration_seconds": 12.5}),
        encoding="utf-8",
    )

    report = evaluate_recovery(run)

    assert report.failure_capture_rate == 1.0
    assert report.sandbox_validation_success_rate == 1.0
    assert report.live_repair_success_rate == 1.0
    assert report.mean_attempts_before_success == 2.0
    assert report.usage.total_tokens == 120
    assert report.usage.duration_seconds == 12.5
    assert report.usage.cost_usd is None


def test_phase_three_evaluation_command_writes_report(
    tmp_path: Path,
    capsys,
) -> None:
    run = tmp_path / "execution-run"
    run.mkdir()
    (run / "execution.json").write_text(
        json.dumps({"completed": [], "failed": [], "attempts": [], "recoveries": []}),
        encoding="utf-8",
    )
    (run / "failure-bundle.json").write_text(
        json.dumps({"failures": []}),
        encoding="utf-8",
    )
    (run / "run-manifest.json").write_text(
        json.dumps({"run_id": "test-run", "duration_seconds": 1}),
        encoding="utf-8",
    )
    output = tmp_path / "evaluation.json"

    exit_code = main(
        ["evaluation", "phase-three", "--run", str(run), "--output", str(output)]
    )

    assert exit_code == 0
    assert output.is_file()
    assert "capture=unavailable" in capsys.readouterr().out


def test_repair_rate_counts_exhausted_recovery_attempts(tmp_path: Path) -> None:
    run = tmp_path / "execution-run"
    run.mkdir()
    (run / "execution.json").write_text(
        json.dumps(
            {
                "completed": ["S001"],
                "failed": [{"step_id": "S002", "reason": "needs human"}],
                "attempts": [],
                "recoveries": [
                    {"step_id": "S001", "status": "resolved"},
                    {"step_id": "S002", "status": "exhausted"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "failure-bundle.json").write_text('{"failures": []}', encoding="utf-8")
    (run / "run-manifest.json").write_text('{"run_id": "test-run"}', encoding="utf-8")

    report = evaluate_recovery(run)

    assert report.live_repair_success_rate == 0.5
