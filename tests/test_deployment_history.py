from __future__ import annotations

from pathlib import Path

from pheragent.deployment.history import RunHistory


def test_history_restores_checkpoint_and_accumulates_usage(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    history.append("decision", usage={"requests": 1, "input_tokens": 20})
    history.checkpoint({"focus": "database", "goal_stack": ["storage"]})
    history.append("decision", usage={"requests": 1, "input_tokens": 10})

    restored = RunHistory(tmp_path)

    assert restored.restore() == {"focus": "database", "goal_stack": ["storage"]}
    assert restored.usage() == {"requests": 2, "input_tokens": 30}
    assert [event["sequence"] for event in restored.events()] == [1, 2, 3]


def test_history_supports_recent_and_compact_source_context(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    history.append("decision", value={"focus": "find command"}, usage={})
    history.append(
        "tool",
        tool="read_file",
        result={
            "source": "repository-1:README.md",
            "total_lines": 200,
            "text": "start\n" + "x" * 4000 + "\nmake start",
        },
    )

    recent = history.context(mode="recent", window=1)
    summary = history.context(mode="summary", window=8)

    assert recent[0]["tool"] == "read_file"
    assert summary["source_notes"][0]["source"] == "repository-1:README.md"
    assert "make start" in summary["source_notes"][0]["text"]
    assert len(summary["source_notes"][0]["text"]) < 3100
