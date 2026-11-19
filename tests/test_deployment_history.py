from pathlib import Path

from pheragent.deployment.history import RunHistory, _source_location
from pheragent.deployment.task import DeploymentTask


def test_history_is_append_only_and_accumulates_usage(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    history.append("decision", usage={"requests": 1, "input_tokens": 20})
    history.append("decision", usage={"requests": 1, "input_tokens": 10})

    restored = RunHistory(tmp_path)

    assert restored.usage() == {"requests": 2, "input_tokens": 30}
    assert [event["sequence"] for event in restored.events()] == [1, 2]


def test_runbook_source_location_omits_url_credentials() -> None:
    assert _source_location("https://user:secret@example.org/repo.git") == (
        "https://example.org/repo.git"
    )


def test_latest_notes_rebuild_from_append_only_events(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    history.append("note_recorded", key="database-host", text="old host", evidence_events=[1])
    history.append("note_recorded", key="database-host", text="new host", evidence_events=[2])
    history.append("note_recorded", key="progress-summary", text="Database verified")

    restored = RunHistory(tmp_path)
    assert list(restored.notes()) == ["database-host", "progress-summary"]
    assert "new host" in restored.render_notes()
    assert "old host" not in restored.render_notes()
    assert len(restored.events()) == 3


def test_note_index_keeps_recent_keys_only(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    for number in range(45):
        history.append("note_recorded", key=f"fact-{number}", text="x" * 200)

    index = history.note_index()
    assert len(index) == 40
    assert "fact-0" not in index
    assert index["fact-44"] == "x" * 120


def test_recent_decisions_pairs_results_and_keeps_ten(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    for iteration in range(12):
        history.append(
            "decision",
            iteration=iteration,
            value={"action": "shell", "reason": "inspect", "command": ["ls"], "notes": []},
            usage={"input_tokens": 20},
        )
        history.append("note_recorded", iteration=iteration, key="step", text="inspect")
        history.append(
            "shell_result",
            iteration=iteration,
            result={"status": "succeeded", "execution": {"stdout": "source file contents"}},
        )

    recent = RunHistory(tmp_path).recent_decisions()

    assert [turn["iteration"] for turn in recent] == list(range(2, 12))
    assert recent[-1]["decision"] == {"action": "shell", "reason": "inspect", "command": ["ls"]}
    assert recent[-1]["result"] == {
        "event_sequence": 36,
        "status": "succeeded",
        "execution": {"stdout": "source file contents"},
    }


def test_recent_decisions_includes_input_record_and_unfinished_actions(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    for iteration, action, result_event in (
        (1, "ask", "input_result"),
        (2, "record", "record_result"),
        (3, "shell", None),
    ):
        history.append("decision", iteration=iteration, value={"action": action, "reason": action})
        if result_event:
            history.append(result_event, iteration=iteration, result={"status": "recorded"})

    recent = RunHistory(tmp_path).recent_decisions()

    assert [turn["result"]["status"] for turn in recent] == [
        "recorded",
        "recorded",
        "not_recorded",
    ]
    assert recent[-1]["decision_event"] == 5


def test_recent_decisions_previews_old_output_without_changing_events(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    output = "first line\nsecond line\n" + "middle\n" * 300 + "last line\n"
    history.append("decision", iteration=1, value={"action": "shell", "command": ["inspect"]})
    history.append(
        "shell_result",
        iteration=1,
        result={
            "status": "succeeded",
            "execution": {
                "exit_code": 0,
                "stdout": output,
                "output_log": str(tmp_path / "actions/0001.log"),
                "output_lines": 303,
            },
        },
    )
    history.append("decision", iteration=2, value={"action": "shell", "command": ["verify"]})
    latest_output = "fresh error\n" * 200
    history.append(
        "shell_result",
        iteration=2,
        result={"status": "failed", "execution": {"stdout": latest_output, "exit_code": 1}},
    )

    recent = history.recent_decisions()

    preview = recent[0]["result"]["execution"]["stdout"]
    assert preview.startswith("first line\nsecond line\n")
    assert "296 complete lines" in preview
    assert "actions/0001.log" in preview
    assert preview.endswith("last line\n")
    assert recent[0]["result"]["execution"]["output_lines"] == 303
    assert recent[-1]["result"]["execution"]["stdout"] == latest_output
    assert history.events()[1]["result"]["execution"]["stdout"] == output


def test_recent_decisions_previews_long_line_and_deduplicates_exact_results(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    output = '{"details":"' + "x" * 3_000 + '"}'
    for iteration in range(1, 5):
        history.append(
            "decision", iteration=iteration, value={"action": "shell", "command": ["inspect"]}
        )
        history.append(
            "shell_result",
            iteration=iteration,
            result={
                "status": "failed" if iteration == 3 else "succeeded",
                "execution": {"exit_code": 1 if iteration == 3 else 0, "stdout": output},
            },
        )

    recent = history.recent_decisions()

    assert "0 complete lines" in recent[0]["result"]["execution"]["stdout"]
    assert "events.jsonl event 2" in recent[0]["result"]["execution"]["stdout"]
    assert recent[1]["result"]["execution"]["stdout"] == "[same output as result event 2]"
    assert "same output" not in recent[2]["result"]["execution"]["stdout"]
    assert recent[3]["result"]["execution"]["stdout"] == output


def test_runbook_contains_only_successful_recorded_changes(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    history.append(
        "sources_acquired",
        sources={
            "sources": [
                {
                    "id": "repo",
                    "location": "https://example.invalid/repo.git",
                    "resolved_revision": "a" * 40,
                    "content_hash": "b" * 64,
                }
            ]
        },
    )
    history.append(
        "shell_result",
        command=["sed", "-i", "s/old/new/", "install.sh"],
        working_directory="repo/deploy",
        state_changing=True,
        exit_code=0,
    )
    changed = history.append(
        "shell_result",
        command=["./install.sh"],
        working_directory="repo/deploy",
        state_changing=True,
        exit_code=0,
    )
    history.append(
        "shell_result",
        command=["appctl", "status"],
        working_directory="repo/deploy",
        state_changing=False,
        exit_code=0,
    )
    history.append(
        "component_recorded",
        component="database",
        deployed=True,
        evidence_events=[changed["sequence"]],
    )

    task = DeploymentTask.model_validate(
        {
            "task": {"objective": "Deploy the database"},
            "sources": {"repositories": ["https://example.invalid/repo.git"]},
            "environment": {"type": "shell"},
        }
    )
    runbook = history.runbook(task)

    assert "https://example.invalid/repo.git @ " + "a" * 40 in runbook
    assert "./install.sh" in runbook
    assert "appctl status" not in runbook
    assert "INCOMPLETE" in runbook
    assert "omits successful changes" in runbook
    assert runbook.index("exit 1") < runbook.index("./install.sh")


def test_runbook_names_sensitive_stdin_without_storing_its_value(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    history.append(
        "sources_acquired",
        sources={
            "sources": [
                {
                    "id": "repo",
                    "kind": "local_directory",
                    "location": "repo",
                    "content_hash": "a" * 64,
                }
            ]
        },
    )
    history.append(
        "decision",
        iteration=1,
        value={
            "action": "ask",
            "required_inputs": [
                {"name": "INSTALL_ANSWER", "prompt": "Confirm installation", "sensitive": True}
            ],
        },
    )
    history.append(
        "decision", iteration=2, value={"action": "shell", "stdin_input": "INSTALL_ANSWER"}
    )
    installed = history.append(
        "shell_result",
        iteration=2,
        command=["./install.sh"],
        working_directory="repo",
        state_changing=True,
        exit_code=0,
        result={"policy": "high_risk_approval: host-level change"},
    )
    checked = history.append(
        "shell_result",
        iteration=3,
        command=["test", "-f", "ready"],
        working_directory="repo",
        state_changing=False,
        exit_code=0,
    )
    history.append(
        "component_recorded",
        component="application",
        deployed=True,
        evidence_events=[installed["sequence"], checked["sequence"]],
        verification_events=[checked["sequence"]],
    )
    history.append(
        "note_recorded",
        key="invariant-ready",
        text="Application is ready",
        evidence_events=[checked["sequence"]],
    )
    task = DeploymentTask.model_validate(
        {
            "task": {
                "objective": "Deploy application",
                "invariants": {"ready": "Application is ready"},
            },
            "sources": {"repositories": ["repo"]},
            "environment": {"type": "shell"},
            "inputs": {"INSTALL_ANSWER": {"from_env": "SECRET_ANSWER", "sensitive": True}},
        }
    )

    runbook = history.runbook(task)

    assert "HUMAN INPUT INSTALL_ANSWER: Confirm installation" in runbook
    assert "${SECRET_ANSWER:?Provide INSTALL_ANSWER}" in runbook
    assert 'printf "%s\\n" "$SECRET_ANSWER" | ./install.sh' in runbook
    assert "HUMAN APPROVAL required for application" in runbook
    assert "INCOMPLETE" not in runbook

    history.append(
        "note_recorded",
        key="invariant-ready",
        text="Application is ready",
        evidence_events=[installed["sequence"]],
    )
    assert "System invariant ready check changes state" in history.runbook(task)

    history.append(
        "component_recorded",
        component="application",
        deployed=True,
        evidence_events=[installed["sequence"]],
        verification_events=[installed["sequence"]],
    )
    assert "application check changes state" in history.runbook(task)
