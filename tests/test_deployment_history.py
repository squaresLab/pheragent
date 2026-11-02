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


def test_history_returns_only_relevant_step_events(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    history.append(
        "decision",
        value={
            "kind": "ACT",
            "tool": "read_file",
            "focus": "find command",
            "step_id": "deploy",
        },
        usage={},
    )
    history.append(
        "tool",
        tool="read_file",
        result={
            "source": "repository-1:README.md",
            "total_lines": 200,
            "complete": True,
            "text": "start\n" + "x" * 4000 + "\nmake start",
        },
    )
    history.append(
        "decision",
        value={
            "kind": "ACT",
            "tool": "observe",
            "focus": "check runtime",
            "step_id": "deploy",
        },
        usage={},
    )

    context = history.context(window=8, step_id="deploy")

    assert [event["event"] for event in context] == ["decision", "tool", "decision"]
    assert context[0]["focus"] == "find command"
    assert context[1]["source"] == "repository-1:README.md"
    assert context[2]["focus"] == "check runtime"
    assert all("text" not in event for event in context)
