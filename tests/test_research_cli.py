import json
from pathlib import Path

from pheragent.cli import main

STUDY = Path("research/studies/graph-retrieval-pilot.yaml").resolve()


def test_research_run_defaults_to_cost_preflight(capsys) -> None:
    exit_code = main(
        [
            "research",
            "run",
            "--study",
            str(STUDY),
            "--case",
            "multihop",
            "--treatment",
            "a0",
            "--repetitions",
            "1",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "planned runs: 1" in output
    assert "maximum LLM requests: 0" in output
    assert "preflight only" in output


def test_research_run_uses_product_analyzer_and_seals_results(
    tmp_path: Path,
    capsys,
) -> None:
    exit_code = main(
        [
            "research",
            "run",
            "--study",
            str(STUDY),
            "--case",
            "multihop",
            "--treatment",
            "a0",
            "--repetitions",
            "1",
            "--output",
            str(tmp_path),
            "--execute",
        ]
    )

    assert exit_code == 0
    study_root = tmp_path / "graph-retrieval-pilot"
    run_dir = next((study_root / "runs").iterdir())
    manifest = json.loads((run_dir / "run-manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_kind"] == "research"
    assert manifest["analysis_method"] == "a0"
    assert manifest["status"] == "completed"
    assert (run_dir / "functional-blocks.yaml").is_file()
    assert (study_root / "results.csv").is_file()
    assert "results:" in capsys.readouterr().out
