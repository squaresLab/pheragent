from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pheragent.analyzer import RepoAnalyzer
from pheragent.llm_planner import (
    IncompleteLLMResponseError,
    OpenAIResponsesBlockPlanner,
    OpenAIResponsesPlannerConfig,
    _openai_client,
    _read_streamed_response_with_usage,
    make_planner,
)
from pheragent.planner import RuleBasedBlockPlanner


class FakeEvent:
    def __init__(self, event_type: str, **values: object):
        self.type = event_type
        for name, value in values.items():
            setattr(self, name, value)


class FakeStream:
    def __init__(self, events: list[FakeEvent]):
        self.events = events
        self.closed = False

    def __iter__(self):
        return iter(self.events)

    def close(self) -> None:
        self.closed = True


class FakeResponses:
    def __init__(self, events: list[FakeEvent], *, failures_before_success: int = 0):
        self.events = events
        self.failures_before_success = failures_before_success
        self.calls: list[dict[str, Any]] = []

    def create(self, **payload):
        self.calls.append(payload)
        if len(self.calls) <= self.failures_before_success:
            raise TimeoutError("temporary")
        return FakeStream(self.events)


class FakeOpenAI:
    clients: list[FakeOpenAI] = []
    events: list[FakeEvent] = []
    failures_before_success = 0

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.responses = FakeResponses(
            self.events,
            failures_before_success=self.failures_before_success,
        )
        type(self).clients.append(self)


def _planner_response_events() -> list[FakeEvent]:
    content = json.dumps(
        {
            "blocks": [
                {
                    "id": "00-custom",
                    "order": 0,
                    "title": "Custom",
                    "goal": "custom setup",
                    "script": "echo custom",
                    "validation_command": (
                        'uv run python -c "import flask; print(flask.__version__)"'
                    ),
                }
            ]
        }
    )
    midpoint = len(content) // 2
    return [
        FakeEvent("response.output_text.delta", delta=content[:midpoint]),
        FakeEvent("response.output_text.delta", delta=content[midpoint:]),
        FakeEvent(
            "response.completed",
            response={
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "output_tokens_details": {"reasoning_tokens": 3},
                    "total_tokens": 18,
                }
            },
        ),
    ]


def test_stream_reader_reports_incomplete_response_with_usage() -> None:
    stream = FakeStream(
        [
            FakeEvent("response.output_text.delta", delta='{"items":['),
            FakeEvent(
                "response.incomplete",
                response={
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "usage": {
                        "input_tokens": 120,
                        "output_tokens": 50,
                        "total_tokens": 170,
                    },
                },
            ),
        ]
    )

    with pytest.raises(IncompleteLLMResponseError, match="max_output_tokens") as error:
        _read_streamed_response_with_usage(stream, error_context="test synthesis")

    assert error.value.content == '{"items":['
    assert error.value.usage == {
        "requests": 1,
        "input_tokens": 120,
        "output_tokens": 50,
        "reasoning_tokens": 0,
        "total_tokens": 170,
    }
    assert stream.closed is True


def test_make_planner_auto_uses_rules_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    planner = make_planner(
        mode="auto",
        model=None,
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        base_url=None,
        timeout=120.0,
        max_tokens=4096,
        retries=3,
        retry_delay=1.0,
    )

    assert isinstance(planner, RuleBasedBlockPlanner)


def test_openai_responses_planner_uses_sdk_streaming(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    context = RepoAnalyzer().analyze(tmp_path)
    context.task_description = "Setup target: flask CLI smoke test."
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    FakeOpenAI.clients = []
    FakeOpenAI.events = _planner_response_events()
    FakeOpenAI.failures_before_success = 0
    monkeypatch.setattr("pheragent.llm_planner._openai_client", FakeOpenAI)

    planner = OpenAIResponsesBlockPlanner(OpenAIResponsesPlannerConfig(model="gpt-5.5"))

    blocks = planner.plan(context)

    assert blocks[0].id == "00-custom"
    assert blocks[0].script.startswith("#!/bin/sh")
    assert "echo custom" in blocks[0].script
    assert blocks[0].validation_command == 'uv run python -c "import flask; print(flask)"'
    assert planner.usage_summary()["total"] == {
        "requests": 1,
        "input_tokens": 11,
        "output_tokens": 7,
        "reasoning_tokens": 3,
        "total_tokens": 18,
    }

    client = FakeOpenAI.clients[0]
    assert client.kwargs["api_key"] == "test-key"
    assert client.kwargs["base_url"] == "https://example.test/v1"
    assert client.kwargs["timeout"] == 120.0

    payload = client.responses.calls[0]
    assert payload["model"] == "gpt-5.5"
    assert payload["stream"] is True
    assert payload["text"] == {"format": {"type": "json_object"}}
    assert "max_output_tokens" not in payload
    assert "temperature" not in payload
    assert "messages" not in payload
    assert "response_format" not in payload
    assert isinstance(payload["input"], list)
    assert payload["input"][0]["role"] == "user"
    input_text = payload["input"][0]["content"][0]
    assert input_text["type"] == "input_text"
    content = json.loads(input_text["text"])
    assert content["output_instructions"] == "Return JSON only."
    assert "json" in input_text["text"].lower()
    assert "repo_context" in content
    assert content["repo_context"]["task_description"] == "Setup target: flask CLI smoke test."


def test_openai_chat_completions_planner_uses_chat_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeChatCompletions:
        def __init__(self):
            self.calls: list[dict[str, Any]] = []

        def create(self, **payload):
            self.calls.append(payload)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "blocks": [
                                        {
                                            "id": "00-chat",
                                            "order": 0,
                                            "title": "Chat",
                                            "goal": "chat setup",
                                            "script": "echo chat",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 13,
                    "completion_tokens": 5,
                    "completion_tokens_details": {"reasoning_tokens": 2},
                    "total_tokens": 18,
                },
            }

    class FakeChat:
        def __init__(self):
            self.completions = FakeChatCompletions()

    class FakeChatOpenAI:
        clients: list[FakeChatOpenAI] = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = FakeChat()
            type(self).clients.append(self)

    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    context = RepoAnalyzer().analyze(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1/chat/completions")
    monkeypatch.setattr("pheragent.llm_planner._openai_client", FakeChatOpenAI)

    planner = make_planner(
        mode="llm",
        model="gpt-5.5",
        api_key_env="OPENAI_API_KEY",
        base_url_env="OPENAI_BASE_URL",
        base_url=None,
        timeout=120.0,
        max_tokens=4096,
        retries=3,
        retry_delay=1.0,
        api_mode="chat-completions",
    )

    blocks = planner.plan(context)

    assert blocks[0].id == "00-chat"
    assert "echo chat" in blocks[0].script
    assert planner.usage_summary()["total"] == {
        "requests": 1,
        "input_tokens": 13,
        "output_tokens": 5,
        "reasoning_tokens": 2,
        "total_tokens": 18,
    }

    client = FakeChatOpenAI.clients[0]
    assert client.kwargs["base_url"] == "https://example.test/v1"
    payload = client.chat.completions.calls[0]
    assert payload["model"] == "gpt-5.5"
    assert payload["response_format"] == {"type": "json_object"}
    assert "input" not in payload
    assert "text" not in payload
    assert "stream" not in payload
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert "json" in payload["messages"][1]["content"].lower()
    content = json.loads(payload["messages"][1]["content"])
    assert content["output_instructions"] == "Return JSON only."
    assert "repo_context" in content


def test_openai_client_disables_sdk_retries() -> None:
    client = _openai_client(
        base_url="https://example.test/v1",
        api_key="test-key",
        timeout=120.0,
    )

    assert client.max_retries == 0


def test_openai_responses_planner_retries_transient_sdk_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    context = RepoAnalyzer().analyze(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    FakeOpenAI.clients = []
    FakeOpenAI.events = _planner_response_events()
    FakeOpenAI.failures_before_success = 1
    monkeypatch.setattr("pheragent.llm_planner._openai_client", FakeOpenAI)
    planner = OpenAIResponsesBlockPlanner(
        OpenAIResponsesPlannerConfig(model="gpt-5.5", max_retries=2, retry_delay_s=0)
    )

    blocks = planner.plan(context)

    assert len(FakeOpenAI.clients[0].responses.calls) == 2
    assert blocks[0].id == "00-custom"


def test_openai_responses_planner_auto_falls_back_after_retries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FailingResponses:
        def create(self, **payload):
            del payload
            raise TimeoutError("temporary")

    class FailingOpenAI:
        def __init__(self, **kwargs):
            del kwargs
            self.responses = FailingResponses()

    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    context = RepoAnalyzer().analyze(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("pheragent.llm_planner._openai_client", FailingOpenAI)
    planner = OpenAIResponsesBlockPlanner(
        OpenAIResponsesPlannerConfig(
            model="gpt-5.5",
            max_retries=1,
            retry_delay_s=0,
            fallback_on_error=True,
        )
    )

    blocks = planner.plan(context)

    assert blocks
    assert any(block.id.endswith("python-deps") for block in blocks)


def test_openai_responses_planner_uses_safe_preflight_script() -> None:
    planner = OpenAIResponsesBlockPlanner(OpenAIResponsesPlannerConfig(model="gpt-5.5"))

    blocks = planner._parse_blocks(
        json.dumps(
            {
                "blocks": [
                    {
                        "id": "00-preflight",
                        "order": 0,
                        "title": "Preflight",
                        "goal": "inspect",
                        "script": (
                            "echo preflight; "
                            "echo 'python executable not found' >&2; "
                            "exit 127"
                        ),
                    }
                ]
            }
        )
    )

    assert "python executable not found" not in blocks[0].script
    assert "command -v python3" in blocks[0].script


def test_openai_responses_planner_uses_safe_python_dependency_script() -> None:
    planner = OpenAIResponsesBlockPlanner(OpenAIResponsesPlannerConfig(model="gpt-5.5"))

    blocks = planner._parse_blocks(
        json.dumps(
            {
                "blocks": [
                    {
                        "id": "02-python-deps",
                        "order": 2,
                        "title": "Python Dependencies",
                        "goal": "install",
                        "script": "python3 -m pip install --upgrade pip setuptools wheel",
                    }
                ]
            }
        )
    )

    assert "PIP_BREAK_SYSTEM_PACKAGES" not in blocks[0].script
    assert "python3 -m venv .pheragent-tools" in blocks[0].script
    assert "ln -sf /workspace/repo/.pheragent-tools/bin/uv /usr/local/bin/uv" in blocks[
        0
    ].script
    assert "install_requirements requirements.txt" in blocks[0].script
    assert "pheragent-requirements-sanitized" in blocks[0].script
    assert 'target_file="$sanitized_root/$req_file"' in blocks[0].script
    assert 'cp -R requirements "$sanitized_root/requirements"' in blocks[0].script
    assert 'target.write_text("\\n".join(lines) + "\\n", encoding="utf-8")' in blocks[
        0
    ].script
    assert 'target.write_text("\n".join(lines) + "\n", encoding="utf-8")' not in blocks[
        0
    ].script
    sanitizer_start = blocks[0].script.index("<<'PY'\n") + len("<<'PY'\n")
    sanitizer_end = blocks[0].script.index("\nPY\n", sanitizer_start)
    compile(blocks[0].script[sanitizer_start:sanitizer_end], "<sanitizer>", "exec")
    assert "skipped {name} requirement because CUDA/nvcc is unavailable" in blocks[0].script
    assert '"flash-attn"' in blocks[0].script
    assert '"deepspeed"' in blocks[0].script
    assert "numpy>=1.26,<2" in blocks[0].script
    assert "editable install failed; continuing with dependency-only environment" in blocks[
        0
    ].script
    assert "ensure_pytest /workspace/repo/.venv/bin/python" in blocks[0].script
    assert "ln -sf /workspace/repo/.venv/bin/python /usr/local/bin/python" in blocks[0].script
    assert "ln -sf /workspace/repo/.venv/bin/python /usr/local/bin/python3" in blocks[
        0
    ].script
    assert "ln -sf /workspace/repo/.venv/bin/pytest /usr/local/bin/pytest" in blocks[0].script
    assert blocks[0].validation_command == (
        "cd /workspace/repo && test -x .venv/bin/python && .venv/bin/python -m pip check"
    )


def test_openai_responses_planner_treats_python_language_deps_as_safe_dependency_block() -> None:
    planner = OpenAIResponsesBlockPlanner(OpenAIResponsesPlannerConfig(model="gpt-5.5"))

    blocks = planner._parse_blocks(
        json.dumps(
            {
                "blocks": [
                    {
                        "id": "02-language-deps",
                        "order": 2,
                        "title": "Language Dependencies",
                        "goal": "install python deps",
                        "script": (
                            "python3 -m venv .venv && "
                            ".venv/bin/python -m pip install -r requirements.txt && "
                            ".venv/bin/python -m pip install -e ."
                        ),
                        "validation_command": (
                            "cd /workspace/repo && .venv/bin/python -m pytest --version"
                        ),
                    }
                ]
            }
        )
    )

    assert "python dependencies" in blocks[0].script
    assert "pip install --no-deps -e ." in blocks[0].script
    assert "ensure_pytest /workspace/repo/.venv/bin/python" in blocks[0].script
    assert "if [ ! -x .venv/bin/python ]; then" in blocks[0].script
    assert "rm -rf .venv" in blocks[0].script
    assert blocks[0].validation_command == (
        "cd /workspace/repo && test -x .venv/bin/python && .venv/bin/python -m pip check"
    )


def test_openai_responses_planner_uses_collect_only_for_build_test_validation() -> None:
    planner = OpenAIResponsesBlockPlanner(OpenAIResponsesPlannerConfig(model="gpt-5.5"))

    blocks = planner._parse_blocks(
        json.dumps(
            {
                "blocks": [
                    {
                        "id": "03-build-test-prep",
                        "order": 3,
                        "title": "Build/Test Prep",
                        "goal": "install pytest and validate tests",
                        "script": "python -m pip install pytest",
                        "validation_command": (
                            "cd /workspace/repo && ./.venv/bin/python -m pytest -q"
                        ),
                    }
                ]
            }
        )
    )

    assert blocks[0].validation_command is not None
    assert "--collect-only" in blocks[0].validation_command
    assert ".venv/bin/python" in blocks[0].validation_command
    assert "pytest -q" not in blocks[0].validation_command
    assert "conftest.py" not in blocks[0].script


def test_openai_responses_planner_rewrites_existing_collect_only_to_venv_python() -> None:
    planner = OpenAIResponsesBlockPlanner(OpenAIResponsesPlannerConfig(model="gpt-5.5"))

    blocks = planner._parse_blocks(
        json.dumps(
            {
                "blocks": [
                    {
                        "id": "03-build-test-prep",
                        "order": 3,
                        "title": "Build/Test Prep",
                        "goal": "validate tests",
                        "script": "python3 -m pytest --version",
                        "validation_command": "python3 -m pytest --collect-only -q",
                    }
                ]
            }
        )
    )

    assert blocks[0].validation_command is not None
    assert ".venv/bin/python" in blocks[0].validation_command
    assert "python3 -m pytest --collect-only" not in blocks[0].validation_command
    assert "no pytest tests collected" in blocks[0].validation_command


def test_openai_responses_planner_uses_safe_go_dependency_script() -> None:
    planner = OpenAIResponsesBlockPlanner(OpenAIResponsesPlannerConfig(model="gpt-5.5"))

    blocks = planner._parse_blocks(
        json.dumps(
            {
                "blocks": [
                    {
                        "id": "02-go-deps",
                        "order": 2,
                        "title": "Go Dependencies",
                        "goal": "download modules",
                        "script": "go version && go mod download",
                        "validation_command": "go list -mod=mod ./...",
                    }
                ]
            }
        )
    )

    assert "go dependencies" in blocks[0].script
    assert "install_go_release" in blocks[0].script
    assert "go1.24.0" not in blocks[0].script
    assert blocks[0].validation_command is not None
    assert "-buildvcs=false" in blocks[0].validation_command
    assert "go list -mod=mod ./..." in blocks[0].validation_command


def test_openai_responses_planner_treats_go_system_deps_as_safe_go_block() -> None:
    planner = OpenAIResponsesBlockPlanner(OpenAIResponsesPlannerConfig(model="gpt-5.5"))

    blocks = planner._parse_blocks(
        json.dumps(
            {
                "blocks": [
                    {
                        "id": "01-system-deps",
                        "order": 1,
                        "title": "System Dependencies",
                        "goal": "install go",
                        "script": (
                            "apt-get update && apt-get install -y golang-go git"
                        ),
                        "validation_command": (
                            "go version >/dev/null 2>&1 && git --version >/dev/null 2>&1"
                        ),
                    }
                ]
            }
        )
    )

    assert "go dependencies" in blocks[0].script
    assert "install_go_release" in blocks[0].script
    assert blocks[0].validation_command is not None
    assert "go list -mod=mod ./..." in blocks[0].validation_command


def test_openai_responses_planner_does_not_treat_django_as_go_block() -> None:
    planner = OpenAIResponsesBlockPlanner(OpenAIResponsesPlannerConfig(model="gpt-5.5"))

    blocks = planner._parse_blocks(
        json.dumps(
            {
                "blocks": [
                    {
                        "id": "02-django-deps",
                        "order": 2,
                        "title": "Django Dependencies",
                        "goal": "install django deps",
                        "script": "python -m pip install django",
                        "validation_command": "python -c 'import django'",
                    }
                ]
            }
        )
    )

    assert "go dependencies" not in blocks[0].script


def test_openai_responses_planner_replaces_repo_code_modifying_build_test_script() -> None:
    planner = OpenAIResponsesBlockPlanner(OpenAIResponsesPlannerConfig(model="gpt-5.5"))

    blocks = planner._parse_blocks(
        json.dumps(
            {
                "blocks": [
                    {
                        "id": "03-build-test-prep",
                        "order": 3,
                        "title": "Build/Test Prep",
                        "goal": "prepare tests",
                        "script": "cat > conftest.py <<'PY'\nprint('patch')\nPY",
                        "validation_command": "python -m pytest -q",
                    }
                ]
            }
        )
    )

    assert "build/test prep" in blocks[0].script
    assert "conftest.py" not in blocks[0].script
    assert "--collect-only" in (blocks[0].validation_command or "")


def test_openai_responses_planner_replaces_task_validation_build_test_script() -> None:
    planner = OpenAIResponsesBlockPlanner(OpenAIResponsesPlannerConfig(model="gpt-5.5"))

    blocks = planner._parse_blocks(
        json.dumps(
            {
                "blocks": [
                    {
                        "id": "03-build-test-prep",
                        "order": 3,
                        "title": "Build/Test Prep",
                        "goal": "check dev server",
                        "script": (
                            "cd /workspace/repo\n"
                            "pnpm run build\n"
                            "pnpm run dev >/tmp/dev.log 2>&1 &\n"
                            "sleep 90\n"
                            "curl -fsS http://127.0.0.1:5173 >/dev/null"
                        ),
                        "validation_command": (
                            "curl -fsS http://127.0.0.1:5173 >/dev/null"
                        ),
                    }
                ]
            }
        )
    )

    assert "pnpm run dev" not in blocks[0].script
    assert "node --version" in blocks[0].script
    assert blocks[0].validation_command is not None
    assert "localhost" not in blocks[0].validation_command


def test_openai_responses_planner_replaces_node_runtime_checks_with_safe_script() -> None:
    planner = OpenAIResponsesBlockPlanner(OpenAIResponsesPlannerConfig(model="gpt-5.5"))

    blocks = planner._parse_blocks(
        json.dumps(
            {
                "blocks": [
                    {
                        "id": "21-node-runtime",
                        "order": 21,
                        "title": "Node Runtime",
                        "goal": "verify node runtime",
                        "script": "node --version && npm --version",
                        "validation_command": "node --version && npm --version",
                    }
                ]
            }
        )
    )

    assert "ensuring node runtime" in blocks[0].script
    assert "apt-get install -y --no-install-recommends nodejs npm" in blocks[0].script
    assert "resolve_runtime_bin" in blocks[0].script
    assert 'ln -sf "$NODE_BIN" .pheragent-tools/bin/node' in blocks[0].script
    assert ".pheragent-tools/bin/node --version" in blocks[0].script
    assert blocks[0].validation_command == (
        "test -x .pheragent-tools/bin/node && test -x .pheragent-tools/bin/npm && "
        ".pheragent-tools/bin/node --version && .pheragent-tools/bin/npm --version"
    )


def test_openai_responses_planner_replaces_python_runtime_checks_with_safe_script() -> None:
    planner = OpenAIResponsesBlockPlanner(OpenAIResponsesPlannerConfig(model="gpt-5.5"))

    blocks = planner._parse_blocks(
        json.dumps(
            {
                "blocks": [
                    {
                        "id": "20-python-runtime",
                        "order": 20,
                        "title": "Python Runtime",
                        "goal": "verify python runtime",
                        "script": 'python3 -c "import sys; print(sys.version)"',
                        "validation_command": 'python3 -c "import sys; print(sys.version)"',
                    }
                ]
            }
        )
    )

    assert "ensuring python runtime" in blocks[0].script
    assert "python3 python3-pip python3-venv" in blocks[0].script
    assert "SYSTEM_PYTHON=python3" in blocks[0].script
    assert '"$SYSTEM_PYTHON" -m venv .venv' in blocks[0].script
    assert "./.venv/bin/python -m pip --version" in blocks[0].script
    assert blocks[0].validation_command == (
        "test -x .venv/bin/python && "
        './.venv/bin/python -c "import sys; print(sys.executable); print(sys.version)" '
        "&& ./.venv/bin/python -m pip --version"
    )


def test_openai_responses_planner_sanitizes_wagtail_module_cli() -> None:
    planner = OpenAIResponsesBlockPlanner(OpenAIResponsesPlannerConfig(model="gpt-5.5"))

    blocks = planner._parse_blocks(
        json.dumps(
            {
                "blocks": [
                    {
                        "id": "03-build-test-prep",
                        "order": 3,
                        "title": "Build/Test Prep",
                        "goal": "check wagtail",
                        "script": "python -m wagtail --help && python -m wagtail start mysite",
                        "validation_command": "python -m wagtail --help",
                    }
                ]
            }
        )
    )

    assert "python -m wagtail" not in blocks[0].script
    assert "wagtail start" not in blocks[0].script
    assert blocks[0].validation_command is not None
    assert "python -m wagtail" not in blocks[0].validation_command


def test_openai_responses_planner_rejects_repo_code_modifying_setup_blocks() -> None:
    planner = OpenAIResponsesBlockPlanner(OpenAIResponsesPlannerConfig(model="gpt-5.5"))
    source_write_script = (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "Path('pkg/app.py').write_text('x')\n"
        "PY"
    )

    with pytest.raises(ValueError, match="modifies repository code"):
        planner._parse_blocks(
            json.dumps(
                {
                    "blocks": [
                        {
                            "id": "01-system-deps",
                            "order": 1,
                            "title": "System Dependencies",
                            "goal": "install deps",
                            "script": source_write_script,
                        }
                    ]
                }
            )
        )
