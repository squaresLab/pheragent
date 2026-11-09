from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_history_recalls_current_step_source_search(tmp_path: Path) -> None:
    history = RunHistory(tmp_path)
    history.append(
        "tool",
        iteration=1,
        tool="search_sources",
        source_signature="current",
        source_request={
            "step_id": "database",
            "focus": "find the PostgreSQL installer",
            "query": "postgres install helm",
        },
        result={
            "hits": [
                {
                    "source": "repository-1:postgres/install.sh",
                    "start_line": 1,
                    "end_line": 20,
                    "text": "helm install postgres",
                }
            ]
        },
    )

    recalled = history.recall_source(
        SimpleNamespace(
            tool="search_sources",
            step_id="database",
            query="helm postgres install",
        ),
        source_signature="current",
    )

    assert recalled["memory_hit"] is True
    assert recalled["hits"][0]["text"] == "helm install postgres"
    assert (
        history.recall_source(
            SimpleNamespace(
                tool="search_sources",
                step_id="database",
                query="helm postgres install",
            ),
            source_signature="changed",
        )
        is None
    )


def test_history_renders_an_observed_runbook(tmp_path: Path) -> None:
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
    history.append(
        "action_started",
        iteration=2,
        command=["./install.sh"],
        working_directory="repository-1:deploy",
        step_id="database",
        stdin_input="INSTALL_CONFIRMATION",
    )
    history.append(
        "action_finished",
        iteration=2,
        result={
            "status": "validated",
            "after": [{"command": ["appctl", "status"]}],
        },
    )

    runbook = history.runbook()

    assert "https://example.invalid/repo.git @ " + "a" * 40 in runbook
    assert 'printf "%s\\n" "${INSTALL_CONFIRMATION:?set INSTALL_CONFIRMATION}"' in runbook
    assert "./install.sh" in runbook
    assert "# verify: appctl status" in runbook
