import json
from pathlib import Path

from pheragent.deployment.run_records import RunRecorder


def test_run_recorder_seals_outputs_and_redacts_secrets(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    recorder = RunRecorder.start(
        run_dir,
        run_kind="research",
        analysis_method="a2",
        inputs={"context": "context.yaml", "password": "password=unsafe"},
    )
    (run_dir / "artifact.yaml").write_text("version: 1\n", encoding="utf-8")

    recorder.complete(
        metrics={"component_count": 2},
        sources={"sources": []},
        llm={"usage": {"requests": 0}},
    )

    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["inputs"]["password"] == "password=[REDACTED]"
    assert "artifact.yaml" in manifest["artifacts"]
    assert "metrics.json" in manifest["artifacts"]
    assert len((run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_run_recorder_preserves_failed_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "failed"
    recorder = RunRecorder.start(
        run_dir,
        run_kind="product",
        analysis_method="deployment-analysis-v1",
        inputs={},
    )

    recorder.fail(ValueError("token=unsafe"))

    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]["message"] == "token=[REDACTED]"
