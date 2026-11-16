from pathlib import Path

from pheragent.deployment.history import RunHistory


def test_history_is_append_only_and_accumulates_usage(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    history.append("decision", usage={"requests": 1, "input_tokens": 20})
    history.append("decision", usage={"requests": 1, "input_tokens": 10})

    restored = RunHistory(tmp_path)

    assert restored.usage() == {"requests": 2, "input_tokens": 30}
    assert [event["sequence"] for event in restored.events()] == [1, 2]


def test_runbook_contains_only_successful_recorded_changes(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    history.append(
        "sources_acquired",
        sources={
            "sources": [
                {
                    "id": "repository-1",
                    "location": "https://example.invalid/repo.git",
                    "resolved_revision": "a" * 40,
                    "content_hash": "b" * 64,
                }
            ]
        },
    )
    changed = history.append(
        "shell_result",
        command=["./install.sh"],
        working_directory="repository-1:deploy",
        state_changing=True,
        exit_code=0,
    )
    history.append(
        "shell_result",
        command=["appctl", "status"],
        working_directory="repository-1:deploy",
        state_changing=False,
        exit_code=0,
    )
    history.append(
        "component_recorded",
        component="database",
        deployed=True,
        evidence_events=[changed["sequence"]],
    )

    runbook = history.runbook()

    assert "https://example.invalid/repo.git @ " + "a" * 40 in runbook
    assert "./install.sh" in runbook
    assert "appctl status" not in runbook
