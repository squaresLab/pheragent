from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from .models import CommandBlock, LLMApiMode, RepoContext, to_jsonable
from .planner import (
    BlockPlanner,
    RuleBasedBlockPlanner,
    _node_runtime_script,
    _node_runtime_validation_command,
    _preflight_script,
    _python_runtime_script,
)
from .utils import shell_script, slugify


class IncompleteLLMResponseError(RuntimeError):
    """A streamed Responses API request stopped before producing a complete response."""

    def __init__(self, reason: str, *, content: str, usage: dict[str, int]):
        super().__init__(f"response incomplete: {reason}")
        self.reason = reason
        self.content = content
        self.usage = usage


@dataclass(slots=True)
class OpenAIResponsesPlannerConfig:
    model: str = "gpt-5.5"
    api_mode: LLMApiMode = "responses"
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    base_url: str | None = None
    timeout: float = 120.0
    max_tokens: int = 4096
    max_retries: int = 3
    retry_delay_s: float = 1.0
    fallback_on_error: bool = False


class OpenAIResponsesBlockPlanner:
    def __init__(
        self,
        config: OpenAIResponsesPlannerConfig,
        *,
        fallback: BlockPlanner | None = None,
    ):
        self.config = config
        self.config.api_mode = _normalize_llm_api_mode(self.config.api_mode)
        self.fallback = fallback or RuleBasedBlockPlanner()
        self.token_usage = _empty_token_usage()

    def plan(self, context: RepoContext) -> list[CommandBlock]:
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing API key in env var {self.config.api_key_env}")
        base_url = _resolve_openai_base_url(
            configured_base_url=self.config.base_url,
            base_url_env=self.config.base_url_env,
            api_mode=self.config.api_mode,
        )

        try:
            content = self._create_response(
                _openai_client(base_url=base_url, api_key=api_key, timeout=self.config.timeout),
                self._request_payload(context),
            )
            return self._parse_blocks(content)
        except Exception:
            if self.config.fallback_on_error:
                return self.fallback.plan(context)
            raise

    def _request_payload(self, context: RepoContext) -> dict[str, Any]:
        user_text = json.dumps(
            {
                "output_instructions": "Return JSON only.",
                "repo_context": to_jsonable(context),
                "fallback_blocks": [to_jsonable(block) for block in self.fallback.plan(context)],
            },
            ensure_ascii=False,
            indent=2,
        )
        if self.config.api_mode == "chat-completions":
            return _chat_completion_payload(self.config.model, _SYSTEM_PROMPT, user_text)
        return _responses_payload(self.config.model, _SYSTEM_PROMPT, user_text)

    def _create_response(
        self,
        client: Any,
        payload: dict[str, Any],
    ) -> str:
        content = ""
        max_attempts = max(1, self.config.max_retries)
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                if self.config.api_mode == "chat-completions":
                    content, usage = _create_chat_completion_with_usage(
                        client,
                        payload,
                        error_context="LLM planner",
                    )
                else:
                    stream = client.responses.create(**payload)
                    content, usage = _read_streamed_response_with_usage(
                        stream,
                        error_context="LLM planner",
                    )
                _add_token_usage(self.token_usage, usage)
                break
            except Exception as exc:
                last_error = RuntimeError(_format_llm_error("LLM planner", exc))
                if attempt == max_attempts:
                    raise last_error from exc
                if not _retryable_llm_error(exc):
                    raise last_error from exc
            _sleep_before_retry(attempt, self.config.retry_delay_s)
        else:
            raise RuntimeError(f"LLM planner request failed: {last_error}") from last_error

        return content

    def usage_summary(self) -> dict[str, dict[str, int]]:
        planner_usage = _copy_token_usage(self.token_usage)
        return {
            "planner": planner_usage,
            "total": _copy_token_usage(planner_usage),
        }

    def _parse_blocks(self, content: str) -> list[CommandBlock]:
        data = _parse_json_object(content)
        raw_blocks = data.get("blocks")
        if not isinstance(raw_blocks, list) or not raw_blocks:
            raise ValueError("LLM planner JSON must contain a non-empty blocks list")

        blocks: list[CommandBlock] = []
        used_ids: set[str] = set()
        for index, item in enumerate(raw_blocks):
            if not isinstance(item, dict):
                raise ValueError("block entries must be JSON objects")
            title = str(item.get("title") or "block")
            block_id = str(item.get("id") or f"{index:02d}-{slugify(title)}")
            block_id = slugify(block_id)
            if block_id in used_ids:
                block_id = f"{block_id}-{index}"
            used_ids.add(block_id)
            script = str(item.get("script", "")).strip()
            if not script:
                raise ValueError(f"block {block_id} has empty script")
            script = _sanitize_script_command(script)
            is_preflight = "preflight" in block_id or "preflight" in title.lower()
            if is_preflight:
                script = _preflight_script()
            validation_command = _sanitize_validation_command(
                _optional_str(item.get("validation_command"))
            )
            script_rejection = _repo_code_modification_rejection_reason(script)
            if _is_python_runtime_block(block_id, title, script, validation_command):
                script = _python_runtime_script()
                validation_command = (
                    "test -x .venv/bin/python && "
                    './.venv/bin/python -c "import sys; print(sys.executable); print(sys.version)" '
                    "&& ./.venv/bin/python -m pip --version"
                )
            elif _is_python_dependency_block(block_id, title, script, validation_command):
                script = _safe_python_dependency_script()
                validation_command = _safe_python_dependency_validation_command()
            elif _is_node_runtime_block(block_id, title, script, validation_command):
                script = _node_runtime_script()
                validation_command = _node_runtime_validation_command()
            elif not is_preflight and _is_go_dependency_block(
                block_id,
                title,
                script,
                validation_command,
            ):
                script = _safe_go_dependency_script()
                validation_command = _safe_go_dependency_validation_command()
            elif _is_build_test_prep_block(block_id, title):
                task_like = _uses_task_or_service_validation(script, validation_command)
                validation_command = _safe_build_test_validation_command(
                    validation_command,
                    force_generic=task_like,
                )
                if (
                    script_rejection
                    or task_like
                    or _uses_python_test_tooling(script, validation_command)
                ):
                    script = _safe_build_test_prep_script()
            elif script_rejection:
                raise ValueError(f"block {block_id} modifies repository code: {script_rejection}")
            blocks.append(
                CommandBlock(
                    id=block_id,
                    order=int(item.get("order", index)),
                    title=title,
                    goal=str(item.get("goal") or ""),
                    script=shell_script(script),
                    validation_command=validation_command,
                )
            )
        return sorted(blocks, key=lambda block: (block.order, block.id))


def make_planner(
    *,
    mode: str,
    model: str | None,
    api_key_env: str,
    base_url_env: str,
    base_url: str | None,
    timeout: float,
    max_tokens: int,
    retries: int,
    retry_delay: float,
    api_mode: str = "responses",
) -> BlockPlanner:
    fallback = RuleBasedBlockPlanner()
    if mode == "rules":
        return fallback
    has_key = bool(os.getenv(api_key_env))
    if mode == "auto" and not has_key:
        return fallback
    if mode not in {"auto", "llm"}:
        raise ValueError(f"unsupported planner mode: {mode}")
    normalized_api_mode = _normalize_llm_api_mode(api_mode)
    return OpenAIResponsesBlockPlanner(
        OpenAIResponsesPlannerConfig(
            model=model or os.getenv("PHERAGENT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5.5",
            api_mode=normalized_api_mode,
            api_key_env=api_key_env,
            base_url_env=base_url_env,
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=retries,
            retry_delay_s=retry_delay,
            fallback_on_error=mode == "auto",
        ),
        fallback=fallback,
    )


def _openai_client(*, base_url: str, api_key: str, timeout: float) -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(f"failed to import OpenAI Python SDK: {exc}") from exc

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
    )


def _normalize_llm_api_mode(value: str | None) -> LLMApiMode:
    normalized = (value or "responses").strip().lower().replace("_", "-")
    if normalized in {"response", "responses"}:
        return "responses"
    if normalized in {"chat", "chat-completion", "chat-completions", "chatcompletions"}:
        return "chat-completions"
    raise ValueError(f"unsupported LLM API mode: {value}")


def _resolve_openai_base_url(
    *,
    configured_base_url: str | None,
    base_url_env: str,
    api_mode: str,
) -> str:
    base_url = (configured_base_url or os.getenv(base_url_env) or "").strip().rstrip("/")
    if not base_url:
        base_url = "https://api.openai.com/v1"
    return _normalize_openai_base_url(base_url, api_mode=api_mode)


def _normalize_openai_base_url(base_url: str, *, api_mode: str) -> str:
    normalized_api_mode = _normalize_llm_api_mode(api_mode)
    endpoint_suffix = (
        "/chat/completions" if normalized_api_mode == "chat-completions" else "/responses"
    )
    if base_url.lower().endswith(endpoint_suffix):
        stripped = base_url[: -len(endpoint_suffix)].rstrip("/")
        if stripped:
            return stripped
    return base_url


def _responses_payload(model: str, system_prompt: str, user_text: str) -> dict[str, Any]:
    return {
        "model": model,
        "instructions": system_prompt,
        "input": _response_text_input(user_text),
        "text": {"format": {"type": "json_object"}},
        "stream": True,
    }


def _chat_completion_payload(model: str, system_prompt: str, user_text: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "response_format": {"type": "json_object"},
    }


def _response_text_input(text: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": text,
                }
            ],
        }
    ]


def _read_streamed_response(stream: Any, *, error_context: str) -> str:
    content, _usage = _read_streamed_response_with_usage(stream, error_context=error_context)
    return content


def _create_chat_completion_with_usage(
    client: Any,
    payload: dict[str, Any],
    *,
    error_context: str,
) -> tuple[str, dict[str, int]]:
    response = client.chat.completions.create(**payload)
    content = _extract_chat_completion_text(response)
    if not content:
        raise RuntimeError(f"{error_context} request returned no content")
    usage = _extract_token_usage(response) or _empty_token_usage()
    usage["requests"] = max(usage.get("requests", 0), 1)
    return content, usage


def _extract_chat_completion_text(response: Any) -> str:
    choices = _event_value(response, "choices")
    if not isinstance(choices, list) or not choices:
        return _extract_response_output_text(response)
    text_parts: list[str] = []
    for choice in choices:
        message = _event_value(choice, "message")
        if message is None:
            message = _event_value(choice, "delta")
        content = _event_value(message, "content")
        if isinstance(content, str):
            text_parts.append(content)
            continue
        if isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                    continue
                text = _event_value(part, "text") or _event_value(part, "content")
                if text is not None:
                    text_parts.append(str(text))
    return "".join(text_parts)


def _read_streamed_response_with_usage(
    stream: Any,
    *,
    error_context: str,
) -> tuple[str, dict[str, int]]:
    chunks: list[str] = []
    done_text: str | None = None
    completed_text: str | None = None
    usage = _empty_token_usage()
    try:
        for event in stream:
            event_usage = _extract_token_usage(event)
            if event_usage is not None:
                usage = event_usage
            event_type = _event_value(event, "type")
            if event_type == "response.output_text.delta":
                delta = _event_value(event, "delta")
                if delta is not None:
                    chunks.append(str(delta))
            elif event_type == "response.output_text.done":
                text = _event_value(event, "text")
                if text is not None:
                    done_text = str(text)
            elif event_type == "response.completed":
                response = _event_value(event, "response")
                completed_text = _extract_response_output_text(response)
                response_usage = _extract_token_usage(response)
                if response_usage is not None:
                    usage = response_usage
            elif event_type == "response.incomplete":
                response = _event_value(event, "response")
                response_usage = _extract_token_usage(response)
                if response_usage is not None:
                    usage = response_usage
                details = _event_value(response, "incomplete_details")
                reason = _event_value(details, "reason") or "unknown reason"
                usage["requests"] = max(usage.get("requests", 0), 1)
                raise IncompleteLLMResponseError(
                    str(reason),
                    content="".join(chunks),
                    usage=usage,
                )
            elif event_type in {"error", "response.failed"}:
                raise RuntimeError(f"{error_context} stream failed: {_event_error_text(event)}")
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    content = "".join(chunks)
    if not content:
        content = done_text or completed_text or ""
    if not content:
        raise RuntimeError(f"{error_context} stream returned no content")
    usage["requests"] = max(usage.get("requests", 0), 1)
    return content, usage


def _empty_token_usage() -> dict[str, int]:
    return {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }


def _copy_token_usage(usage: dict[str, int]) -> dict[str, int]:
    return {key: int(usage.get(key, 0) or 0) for key in _empty_token_usage()}


def _add_token_usage(target: dict[str, int], usage: dict[str, int]) -> None:
    for key in _empty_token_usage():
        target[key] = int(target.get(key, 0) or 0) + int(usage.get(key, 0) or 0)


def merge_usage_summaries(*summaries: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = {"total": _empty_token_usage()}
    for summary in summaries:
        for phase, usage in summary.items():
            if phase == "total":
                continue
            phase_usage = merged.setdefault(phase, _empty_token_usage())
            _add_token_usage(phase_usage, usage)
            _add_token_usage(merged["total"], usage)
    if all(value == 0 for value in merged["total"].values()):
        return {}
    return merged


def _extract_token_usage(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict) and hasattr(value, "model_dump"):
        value = value.model_dump()
    usage = _event_value(value, "usage")
    if usage is None:
        usage = value
    if not isinstance(usage, dict) and hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    input_tokens = _usage_int(usage, "input_tokens", "prompt_tokens")
    output_tokens = _usage_int(usage, "output_tokens", "completion_tokens")
    total_tokens = _usage_int(usage, "total_tokens")
    reasoning_tokens = (
        _usage_int(_event_value(usage, "output_tokens_details"), "reasoning_tokens")
        or _usage_int(_event_value(usage, "completion_tokens_details"), "reasoning_tokens")
        or _usage_int(usage, "reasoning_tokens")
    )
    if (
        input_tokens is None
        and output_tokens is None
        and total_tokens is None
        and reasoning_tokens is None
    ):
        return None
    if total_tokens is None:
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)
    return {
        "requests": 0,
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "reasoning_tokens": int(reasoning_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


def _usage_int(value: Any, *names: str) -> int | None:
    for name in names:
        raw = _event_value(value, name)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return None


def _event_value(event: Any, name: str) -> Any:
    if isinstance(event, dict):
        return event.get(name)
    return getattr(event, name, None)


def _event_error_text(event: Any) -> str:
    error = _event_value(event, "error")
    if error is None:
        return str(event)
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(getattr(error, "message", None) or error)


def _extract_response_output_text(response: Any) -> str:
    output_text = _event_value(response, "output_text")
    if isinstance(output_text, str):
        return output_text

    if not isinstance(response, dict) and hasattr(response, "model_dump"):
        response = response.model_dump()
    output = _event_value(response, "output")
    if not isinstance(output, list):
        return ""
    text_parts: list[str] = []
    for item in output:
        content = _event_value(item, "content")
        if not isinstance(content, list):
            continue
        for part in content:
            if _event_value(part, "type") == "output_text":
                text = _event_value(part, "text")
                if text is not None:
                    text_parts.append(str(text))
    return "".join(text_parts)


def _retryable_http_status(status: int) -> bool:
    return status == 429 or 500 <= status <= 599


def _sleep_before_retry(attempt: int, retry_delay_s: float) -> None:
    delay = max(0.0, retry_delay_s) * (2 ** (attempt - 1))
    if delay > 0:
        time.sleep(delay)


def _retryable_llm_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return _retryable_http_status(status_code)
    openai_error_name = type(exc).__name__
    if openai_error_name in {"APIConnectionError", "APITimeoutError"}:
        return True
    return isinstance(
        exc,
        (TimeoutError, ConnectionError, OSError),
    )


def _format_llm_error(error_context: str, exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        detail = _api_status_detail(exc)
        return f"{error_context} request failed: HTTP {status_code}: {detail}"
    if type(exc).__name__ == "APIError":
        return f"{error_context} request failed: {exc}"
    return f"{error_context} request failed: {exc}"


def _api_status_detail(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    detail = getattr(response, "text", None)
    if detail:
        return str(detail)[:1000]
    return str(exc)


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(content):
            if char != "{":
                continue
            try:
                data, _ = decoder.raw_decode(content[index:])
            except json.JSONDecodeError:
                continue
            break
        else:
            raise ValueError("content did not contain a JSON object") from None
    if not isinstance(data, dict):
        raise ValueError("content JSON must be an object")
    return data


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sanitize_validation_command(command: str | None) -> str | None:
    if command is None:
        return command
    command = _sanitize_script_command(command)
    if "__version__" not in command:
        return command
    patched = re.sub(r"print\(([A-Za-z_][A-Za-z0-9_]*)\.__version__\)", r"print(\1)", command)
    return re.sub(r"([A-Za-z_][A-Za-z0-9_]*)\.__version__", r"\1", patched)


def _sanitize_script_command(command: str) -> str:
    return command.replace("python -m wagtail start ", "wagtail start ").replace(
        "python -m wagtail --help",
        "wagtail --help",
    )


def _is_python_dependency_block(
    block_id: str,
    title: str,
    script: str = "",
    validation_command: str | None = None,
) -> bool:
    haystack = f"{block_id} {title}".lower()
    if "python" in haystack and ("dep" in haystack or "environment" in haystack):
        return True
    return "language" in haystack and "dep" in haystack and _uses_python_dependency_tooling(
        script,
        validation_command,
    )


def _is_python_runtime_block(
    block_id: str,
    title: str,
    script: str = "",
    validation_command: str | None = None,
) -> bool:
    haystack = f"{block_id} {title}".lower()
    has_python_label = re.search(r"(?:^|[^a-z0-9])python(?:3)?(?:[^a-z0-9]|$)", haystack)
    if has_python_label and ("runtime" in haystack or "toolchain" in haystack):
        return True

    command_haystack = f"{script}\n{validation_command or ''}".lower()
    runtime_checks = (
        "python3 --version" in command_haystack
        or "python --version" in command_haystack
        or "print(sys.version)" in command_haystack
        or "python3 -m venv -h" in command_haystack
        or "python -m venv -h" in command_haystack
    )
    installs_deps = _uses_python_dependency_tooling(script, validation_command)
    return bool(has_python_label and runtime_checks and not installs_deps)


def _is_build_test_prep_block(block_id: str, title: str) -> bool:
    haystack = f"{block_id} {title}".lower()
    return ("build" in haystack or "test" in haystack) and "prep" in haystack


def _is_node_runtime_block(
    block_id: str,
    title: str,
    script: str = "",
    validation_command: str | None = None,
) -> bool:
    haystack = f"{block_id} {title}".lower()
    has_node_label = re.search(r"(?:^|[^a-z0-9])node(?:js)?(?:[^a-z0-9]|$)", haystack)
    if has_node_label and ("runtime" in haystack or "toolchain" in haystack):
        return True

    command_haystack = f"{script}\n{validation_command or ''}".lower()
    version_checks = (
        "node --version" in command_haystack
        or "node -v" in command_haystack
        or "npm --version" in command_haystack
        or "npm -v" in command_haystack
    )
    installs_deps = re.search(
        r"\b(?:npm|pnpm|yarn)\s+(?:install|ci|add|run)\b",
        command_haystack,
    )
    return bool(has_node_label and version_checks and not installs_deps)


def _is_go_dependency_block(
    block_id: str,
    title: str,
    script: str = "",
    validation_command: str | None = None,
) -> bool:
    haystack = f"{block_id} {title}".lower()
    has_go_label = re.search(r"(?:^|[^a-z0-9])(?:go|golang)(?:[^a-z0-9]|$)", haystack)
    command_haystack = f"{script}\n{validation_command or ''}".lower()
    has_go_command = re.search(
        r"(?:^|[^a-z0-9_-])go\s+(?:mod|list|test|version)\b",
        command_haystack,
    )
    if not has_go_label and not has_go_command:
        return False
    if "dep" in haystack or "language" in haystack or "module" in haystack:
        return True
    return bool(has_go_command)


def _safe_build_test_validation_command(
    command: str | None,
    *,
    force_generic: bool = False,
) -> str | None:
    if force_generic:
        return _generic_build_test_validation_command()
    if command is None:
        return command
    normalized = " ".join(command.split()).lower()
    if "pytest" not in normalized:
        return command
    return (
        "cd /workspace/repo && "
        "test -x .venv/bin/python; "
        "./.venv/bin/python -m pytest --collect-only -q; "
        "status=$?; "
        'if [ "$status" -eq 5 ]; then '
        'echo "[pheragent] no pytest tests collected"; exit 0; '
        "fi; "
        'exit "$status"'
    )


def _generic_build_test_validation_command() -> str:
    return """
cd /workspace/repo
if [ -f package.json ]; then
  test -x .pheragent-tools/bin/node
  test -x .pheragent-tools/bin/npm
  ./.pheragent-tools/bin/node --version >/dev/null 2>&1
  ./.pheragent-tools/bin/npm --version >/dev/null 2>&1
  if command -v pnpm >/dev/null 2>&1; then pnpm --version >/dev/null 2>&1; fi
  if command -v yarn >/dev/null 2>&1; then yarn --version >/dev/null 2>&1; fi
fi
if [ -f pyproject.toml ] || [ -f setup.py ] || [ -f setup.cfg ] || [ -f requirements.txt ]; then
  test -x .venv/bin/python
  ./.venv/bin/python -m pip --version >/dev/null 2>&1 || true
fi
""".strip()


def _uses_python_test_tooling(script: str, validation_command: str | None) -> bool:
    haystack = f"{script}\n{validation_command or ''}".lower()
    return "pytest" in haystack or "python -m pip" in haystack or "pip install" in haystack


def _uses_task_or_service_validation(script: str, validation_command: str | None) -> bool:
    haystack = f"{script}\n{validation_command or ''}".lower()
    return any(
        token in haystack
        for token in (
            "npm run start",
            "pnpm run dev",
            "yarn dev",
            "vite dev",
            "run dev",
            "localhost:",
            "127.0.0.1:",
            "curl -s",
            "sleep 90",
            "sleep 20",
            "wagtail start",
            "django-admin startproject",
        )
    )


def _uses_python_dependency_tooling(script: str, validation_command: str | None) -> bool:
    haystack = f"{script}\n{validation_command or ''}".lower()
    return any(
        token in haystack
        for token in (
            ".venv",
            "pip install",
            "python -m pip",
            "python3 -m pip",
            "python -m venv",
            "python3 -m venv",
            "requirements.txt",
            "pyproject.toml",
            "setup.py",
            "uv sync",
        )
    )


def _repo_code_modification_rejection_reason(script: str) -> str | None:
    normalized = re.sub(r"\s+", " ", script.strip().lower())
    for token in ("conftest.py", "sitecustomize.py"):
        if token in normalized:
            return f"test monkeypatch file {token!r}"
    for token in (".write_text(", ".write_bytes(", "open("):
        if token in normalized:
            return f"python file write token {token!r}"
    if re.search(r"\b(?:sed\s+-i|perl\s+-pi)\b", normalized):
        return "in-place source edit command"
    if re.search(
        r"(?:^|[;&|]\s*)(?:cat|tee)\b[^;&|>]*>\s*(?:/workspace/repo/)?"
        r"[^;&|\s]+\.(?:py|pyi|js|jsx|ts|tsx|java|go|rs|c|cc|cpp|h|hpp|rb|php|sh)",
        normalized,
    ):
        return "redirect writes to source-like file"
    return None


def _safe_build_test_prep_script() -> str:
    return """
cd /workspace/repo

echo "[pheragent] build/test prep"
if [ -f package.json ]; then
  test -x .pheragent-tools/bin/node
  test -x .pheragent-tools/bin/npm
  ./.pheragent-tools/bin/node --version >/dev/null 2>&1
  ./.pheragent-tools/bin/npm --version >/dev/null 2>&1
  if command -v pnpm >/dev/null 2>&1; then pnpm --version >/dev/null 2>&1; fi
  if command -v yarn >/dev/null 2>&1; then yarn --version >/dev/null 2>&1; fi
fi

if [ -f pyproject.toml ] || [ -f setup.py ] || [ -f setup.cfg ] || [ -f requirements.txt ]; then
  test -x .venv/bin/python
  PYTHON_BIN=./.venv/bin/python

  "$PYTHON_BIN" -m pip --version >/dev/null 2>&1 \
    || "$PYTHON_BIN" -m ensurepip --upgrade \
    || true
  "$PYTHON_BIN" -m pytest --version >/dev/null 2>&1 || "$PYTHON_BIN" -m pip install pytest
  if [ -d tests ] && grep -R "@pytest.mark.asyncio\\|pytest_asyncio" tests pyproject.toml \
    >/dev/null 2>&1; then
    "$PYTHON_BIN" -m pip install pytest-asyncio
  fi
  "$PYTHON_BIN" -m pytest --version
fi
""".strip()


def _safe_python_dependency_script() -> str:
    return """
cd /workspace/repo

echo "[pheragent] python dependencies"
ensure_python3() {
  if command -v python3 >/dev/null 2>&1 && python3 -m venv -h >/dev/null 2>&1; then
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache python3 py3-pip py3-virtualenv
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y python3 python3-pip python3-virtualenv
  elif command -v yum >/dev/null 2>&1; then
    yum install -y python3 python3-pip python3-virtualenv
  else
    echo "python3 with venv support not found and no supported package manager is available" >&2
    exit 127
  fi
}

ensure_python3

mkdir -p .cache/uv .cache/pip
export UV_CACHE_DIR=/workspace/repo/.cache/uv
export PIP_CACHE_DIR=/workspace/repo/.cache/pip

ensure_pytest() {
  target_python="$1"
  if [ ! -x "$target_python" ]; then
    return 0
  fi
  "$target_python" -m pip --version >/dev/null 2>&1 \
    || "$target_python" -m ensurepip --upgrade \
    || true
  "$target_python" -m pytest --version >/dev/null 2>&1 || "$target_python" -m pip install pytest
}

expose_venv_tools() {
  if [ ! -x /workspace/repo/.venv/bin/python ]; then
    return 0
  fi
  if [ -w /usr/local/bin ] || [ "$(id -u)" = "0" ]; then
    ln -sf /workspace/repo/.venv/bin/python /usr/local/bin/python
    ln -sf /workspace/repo/.venv/bin/python /usr/local/bin/python3
    if [ -x /workspace/repo/.venv/bin/pip ]; then
      ln -sf /workspace/repo/.venv/bin/pip /usr/local/bin/pip
      ln -sf /workspace/repo/.venv/bin/pip /usr/local/bin/pip3
    fi
    if [ -x /workspace/repo/.venv/bin/pytest ]; then
      ln -sf /workspace/repo/.venv/bin/pytest /usr/local/bin/pytest
    fi
  fi
}

install_requirements() {
  req_file="$1"
  target_file="$req_file"
  if [ -f "$req_file" ]; then
    sanitized_root=/tmp/pheragent-requirements-sanitized
    rm -rf "$sanitized_root"
    mkdir -p "$sanitized_root/$(dirname "$req_file")"
    target_file="$sanitized_root/$req_file"
    PHERAGENT_SKIP_CUDA_DEPS=0
    if [ -z "${CUDA_HOME:-}" ] && ! command -v nvcc >/dev/null 2>&1; then
      PHERAGENT_SKIP_CUDA_DEPS=1
    fi
    export PHERAGENT_SKIP_CUDA_DEPS
    if [ -d requirements ]; then
      cp -R requirements "$sanitized_root/requirements"
    fi
    ./.venv/bin/python - "$req_file" "$target_file" <<'PY'
import os
import re
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
skip_cuda_deps = os.environ.get("PHERAGENT_SKIP_CUDA_DEPS") == "1"
cuda_source_packages = {
    "apex",
    "bitsandbytes",
    "causal-conv1d",
    "deepspeed",
    "flash-attn",
    "flash-attention",
    "flashattention",
    "mamba-ssm",
    "vllm",
    "xformers",
}
changed = False
lines: list[str] = []
for raw in source.read_text(encoding="utf-8").splitlines():
    stripped = raw.strip()
    if not stripped or stripped.startswith("#"):
        lines.append(raw)
        continue
    passthrough = ("-r ", "--requirement", "-c ", "--constraint", "-e ", "--editable")
    if stripped.startswith(passthrough):
        lines.append(raw)
        continue
    name = re.split(r"\\s*(?:===|==|~=|!=|<=|>=|<|>|;)", stripped, 1)[0]
    name = name.split("[", 1)[0].strip().lower().replace("_", "-")
    if skip_cuda_deps and name in cuda_source_packages:
        changed = True
        print(
            f"[pheragent] skipped {name} requirement because CUDA/nvcc is unavailable",
            file=sys.stderr,
        )
        continue
    if (
        sys.version_info >= (3, 12)
        and name == "numpy"
        and re.search(r"(?:<|<=)\\s*1\\.2[0-3](?:\\.|$)", stripped)
    ):
        lines.append("numpy>=1.26,<2")
        changed = True
        print("[pheragent] relaxed numpy<1.24 requirement for Python 3.12", file=sys.stderr)
        continue
    lines.append(raw)
target.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
if not changed:
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
PY
  fi
  ./.venv/bin/python -m pip install -r "$target_file"
}

if [ -f uv.lock ]; then
  if ! command -v uv >/dev/null 2>&1; then
    python3 -m venv .pheragent-tools
    ./.pheragent-tools/bin/python -m pip install --upgrade pip
    ./.pheragent-tools/bin/python -m pip install uv
    if [ -w /usr/local/bin ] || [ "$(id -u)" = "0" ]; then
      ln -sf /workspace/repo/.pheragent-tools/bin/uv /usr/local/bin/uv
    else
      export PATH="/workspace/repo/.pheragent-tools/bin:$PATH"
    fi
  fi
  uv sync --locked --all-extras --dev \
    || uv sync --frozen --all-extras --dev \
    || uv sync --all-extras --dev
  ensure_pytest /workspace/repo/.venv/bin/python
  expose_venv_tools
else
  if [ ! -x .venv/bin/python ]; then
    rm -rf .venv
    python3 -m venv .venv
  fi
  ./.venv/bin/python -m pip install --upgrade pip setuptools wheel
  if [ -f requirements.txt ]; then
    install_requirements requirements.txt
  fi
  if [ -f pyproject.toml ] || [ -f setup.py ] || [ -f setup.cfg ]; then
    ./.venv/bin/python -m pip install -e '.[dev]' \
      || ./.venv/bin/python -m pip install -e . \
      || ./.venv/bin/python -m pip install --no-deps -e . \
      || echo "[pheragent] editable install failed; continuing with dependency-only environment" >&2
  fi
  ensure_pytest /workspace/repo/.venv/bin/python
  expose_venv_tools
fi
""".strip()


def _safe_go_dependency_script() -> str:
    return """
cd /workspace/repo

echo "[pheragent] go dependencies"
export PATH="/usr/local/go/bin:$PATH"

required_go_version() {
  awk '/^go[[:space:]]+[0-9]+[.][0-9]+/ { print $2; exit }' go.mod 2>/dev/null || true
}

go_minor_version() {
  "$1" version 2>/dev/null | sed -n 's/.*go1[.]\\([0-9][0-9]*\\).*/\\1/p' | head -1
}

install_go_release() {
  version="$1"
  case "$version" in
    *.*.*) ;;
    *) version="${version}.0" ;;
  esac
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) go_arch=amd64 ;;
    aarch64|arm64) go_arch=arm64 ;;
    *) echo "unsupported go architecture: $arch" >&2; exit 1 ;;
  esac
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends ca-certificates curl tar
  fi
  rm -rf /usr/local/go
  curl -fsSL "https://go.dev/dl/go${version}.linux-${go_arch}.tar.gz" -o /tmp/pheragent-go.tgz
  tar -C /usr/local -xzf /tmp/pheragent-go.tgz
  rm -f /tmp/pheragent-go.tgz
  export PATH="/usr/local/go/bin:$PATH"
}

required_go="$(required_go_version)"
required_minor=""
if [ -n "$required_go" ]; then
  required_minor="$(printf '%s\n' "$required_go" | awk -F. '{ print $2 + 0 }')"
fi
current_minor=""
if command -v go >/dev/null 2>&1; then
  current_minor="$(go_minor_version "$(command -v go)")"
fi
if [ -n "$required_minor" ] && [ "$required_minor" -ge 24 ]; then
  if [ -z "$current_minor" ] || [ "$current_minor" -lt "$required_minor" ]; then
    install_go_release "$required_go"
  fi
fi

go version
go mod download
""".strip()


def _safe_go_dependency_validation_command() -> str:
    return (
        'cd /workspace/repo && PATH="/usr/local/go/bin:$PATH" '
        'GOFLAGS="${GOFLAGS:-} -buildvcs=false" go list -mod=mod ./... >/dev/null'
    )


def _safe_python_dependency_validation_command() -> str:
    return (
        "cd /workspace/repo && test -x .venv/bin/python && "
        ".venv/bin/python -m pip check"
    )


_SYSTEM_PROMPT = """
You are pheragent's environment setup block planner.

Return JSON only. The JSON object must have this shape:
{
  "blocks": [
    {
      "id": "00-preflight",
      "order": 0,
      "title": "Preflight",
      "goal": "short purpose",
      "script": "POSIX sh commands",
      "validation_command": "optional POSIX sh command"
    }
  ]
}

Rules:
- Generate command blocks for a Docker container whose repo working directory is /workspace/repo.
- If repo_context.task_description is present, use it as the target setup goal:
  install the tools/dependencies needed for that task, while keeping validation
  lightweight and environment-focused.
- Prefer 4-7 functional blocks per repository. Use fewer blocks for
  single-ecosystem repositories with simple setup. Use more blocks when the
  repository requires multiple runtimes, native build toolchains, or long
  dependency installation logic.
- Split a block when it mixes different repair owners:
  system packages, language runtime installation, virtualenv/package-manager
  bootstrap, project dependency installation, native build configuration,
  test-tooling or validation preparation.
- Prefer this block template and naming scheme:
  00-preflight, 10-system-packages, 20-runtime-toolchain,
  30-project-dependencies, optional 40-native-build-config,
  50-test-tooling, optional 60-service-or-final-validation-prep.
- For multi-language repositories, split runtime and dependency blocks by
  ecosystem when useful, such as 20-python-runtime, 21-node-runtime,
  30-python-deps, 31-node-deps. Keep the total block count capped at 7 when
  practical.
- Use POSIX sh, not bash-specific syntax.
- Keep commands deterministic and idempotent where practical.
- Do not start long-running services in setup blocks.
- Do not use host paths outside /workspace/repo.
- Build/test prep validation must verify tools can run, not require the full test
  suite to pass. Prefer import checks, version checks, or pytest --collect-only.
- Do not create conftest.py, sitecustomize.py, or monkeypatch application routes
  to make project tests pass; this agent builds environments, not application fixes.
- Prefer the provided fallback_blocks when they are suitable, and improve them only
  when repo context indicates a better block sequence.
- The word JSON appears here intentionally because JSON mode requires an explicit JSON instruction.
""".strip()
