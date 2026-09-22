import json
from pathlib import Path

import yaml

from pheragent.cli import main


def _study(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "install.sh").write_text("#!/bin/sh\necho deploy\n", encoding="utf-8")
    (tmp_path / "context.yaml").write_text(
        yaml.safe_dump({"system": "fixture", "deployment": {}, "provided_blocks": []}),
        encoding="utf-8",
    )
    (tmp_path / "sources.yaml").write_text(
        yaml.safe_dump(
            {
                "system": "fixture",
                "sources": [
                    {
                        "id": "fixture",
                        "kind": "local_directory",
                        "location": str(repository),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    study = tmp_path / "study.yaml"
    study.write_text(
        yaml.safe_dump(
            {
                "version": "0.1",
                "id": "fixture-study",
                "max_total_llm_requests": 0,
                "treatments": ["a0"],
                "cases": [
                    {
                        "id": "fixture",
                        "sources": "sources.yaml",
                        "context": "context.yaml",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return study


def test_research_run_defaults_to_cost_preflight(tmp_path: Path, capsys) -> None:
    exit_code = main(["research", "run", "--study", str(_study(tmp_path))])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "planned runs: 1" in output
    assert "maximum LLM requests: 0" in output
    assert "preflight only" in output


def test_research_run_uses_product_analyzer_and_seals_results(
    tmp_path: Path,
    capsys,
) -> None:
    study = _study(tmp_path)
    output = tmp_path / "results"
    exit_code = main(
        ["research", "run", "--study", str(study), "--output", str(output), "--execute"]
    )

    assert exit_code == 0
    study_root = output / "fixture-study"
    run_dir = next((study_root / "runs").iterdir())
    manifest = json.loads(
        (run_dir / ".heragent" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_kind"] == "research"
    assert manifest["analysis_method"] == "a0"
    assert manifest["status"] == "completed"
    assert (run_dir / "functional-blocks.yaml").is_file()
    assert (study_root / "results.csv").is_file()
    assert "results:" in capsys.readouterr().out
