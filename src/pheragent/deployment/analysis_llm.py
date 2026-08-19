from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pheragent.llm_planner import (
    IncompleteLLMResponseError,
    _format_llm_error,
    _openai_client,
    _read_streamed_response_with_usage,
    _resolve_openai_base_url,
)

from .serialization import write_json


@dataclass(slots=True)
class AnalysisLLMConfig:
    enabled: bool = True
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    base_url: str | None = None
    timeout: float = 120.0
    max_output_tokens: int = 5000
    max_requests: int = 2
    cache_dir: Path | None = None
    retry_failed: bool = False
    refresh_cache: bool = False
    reasoning_effort: str | None = None


@dataclass(slots=True)
class LLMRequestBudget:
    limit: int
    attempted: int = 0

    def consume(self) -> bool:
        if self.attempted >= self.limit:
            return False
        self.attempted += 1
        return True


@dataclass(frozen=True, slots=True)
class ClassificationOutcome[T: BaseModel]:
    value: T | None
    stage: str
    status: str
    usage: dict[str, int]
    input_tokens_estimate: int
    warning: str | None = None
    failure_history_path: Path | None = None


class CachedStructuredClassifier:
    """Template for one validated, cached, structured classification request."""

    def __init__(self, config: AnalysisLLMConfig, budget: LLMRequestBudget):
        self._config = config
        self._budget = budget

    def classify[T: BaseModel](
        self,
        *,
        stage: str,
        prompt_version: str,
        instructions: str,
        payload: dict[str, Any],
        response_format: dict[str, Any],
        response_model: type[T],
        validate: Callable[[T], None],
        output_token_limit: int | None = None,
    ) -> ClassificationOutcome[T]:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        estimate = max(1, len(serialized) // 4)
        effective_output_tokens = min(
            self._config.max_output_tokens,
            output_token_limit or self._config.max_output_tokens,
        )
        if not self._config.enabled:
            return ClassificationOutcome(
                None,
                stage,
                "disabled",
                {},
                estimate,
            )
        api_key = os.getenv(self._config.api_key_env)
        if not api_key:
            return ClassificationOutcome(
                None,
                stage,
                "not_requested_no_key",
                {},
                estimate,
                warning=f"{stage} skipped because {self._config.api_key_env} is not set",
            )

        cache_key = _cache_key(
            prompt_version=prompt_version,
            model=self._config.model,
            payload=payload,
            response_format=response_format,
            max_output_tokens=effective_output_tokens,
            reasoning_effort=self._config.reasoning_effort,
        )
        cache_path = (
            self._config.cache_dir / stage / f"{cache_key}.json" if self._config.cache_dir else None
        )
        failure_path = (
            self._config.cache_dir / "failures" / stage / f"{cache_key}.json"
            if self._config.cache_dir
            else None
        )

        cache_warning: str | None = None
        if not self._config.refresh_cache and cache_path and cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                value = response_model.model_validate(cached["result"])
                validate(value)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                cache_warning = f"ignored invalid {stage} success cache: {exc}"
            else:
                return ClassificationOutcome(
                    value,
                    stage,
                    "cache",
                    {},
                    estimate,
                )

        if (
            not self._config.refresh_cache
            and failure_path
            and failure_path.is_file()
            and not self._config.retry_failed
        ):
            failure = _read_json_object(failure_path)
            attempts = failure.get("attempts", [])
            last_attempt = attempts[-1] if isinstance(attempts, list) and attempts else {}
            error = str(last_attempt.get("error", "previous matching request failed"))
            preserved_response = last_attempt.get("response")
            if isinstance(preserved_response, str) and preserved_response.strip():
                try:
                    value = response_model.model_validate(
                        _parse_structured_json_object(preserved_response)
                    )
                    validate(value)
                except KeyError, TypeError, ValueError:
                    pass
                else:
                    if cache_path:
                        write_json(
                            cache_path,
                            {
                                "status": "recovered_from_failure_history",
                                "prompt_version": prompt_version,
                                "model": self._config.model,
                                "result": value.model_dump(mode="json"),
                                "usage": {},
                            },
                        )
                    return ClassificationOutcome(
                        value,
                        stage,
                        "recovered_failure_cache",
                        {},
                        estimate,
                        warning=(
                            f"recovered {stage} from its preserved failed response after "
                            "current contract validation succeeded"
                        ),
                        failure_history_path=failure_path,
                    )
            return ClassificationOutcome(
                None,
                stage,
                "failure_cache",
                {},
                estimate,
                warning=(
                    f"skipped {stage} because the identical request previously failed: "
                    f"{error}; use --retry-failed-llm to retry"
                ),
                failure_history_path=failure_path,
            )

        if not self._budget.consume():
            return ClassificationOutcome(
                None,
                stage,
                "request_budget_exhausted",
                {},
                estimate,
                warning=(
                    f"{stage} skipped because the run reached its "
                    f"{self._budget.limit}-request LLM budget"
                ),
            )

        content = ""
        usage: dict[str, int] = {}
        try:
            base_url = _resolve_openai_base_url(
                configured_base_url=self._config.base_url,
                base_url_env=self._config.base_url_env,
                api_mode="responses",
            )
            client = _openai_client(
                base_url=base_url,
                api_key=api_key,
                timeout=self._config.timeout,
            )
            request: dict[str, Any] = {
                "model": self._config.model,
                "instructions": instructions,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": json.dumps(payload, ensure_ascii=False),
                            }
                        ],
                    }
                ],
                "text": {"format": response_format},
                "max_output_tokens": effective_output_tokens,
                "stream": True,
            }
            if self._config.reasoning_effort:
                request["reasoning"] = {"effort": self._config.reasoning_effort}
            stream = client.responses.create(**request)
            content, usage = _read_streamed_response_with_usage(
                stream,
                error_context=stage.replace("_", " "),
            )
            value = response_model.model_validate(_parse_structured_json_object(content))
            validate(value)
        except Exception as exc:
            failure_status = "failed"
            if isinstance(exc, IncompleteLLMResponseError):
                content = exc.content
                usage = exc.usage
                reason = "".join(
                    character if character.isalnum() else "_" for character in exc.reason.casefold()
                ).strip("_")
                failure_status = f"incomplete_{reason or 'response'}"
            elif isinstance(exc, ValueError):
                failure_status = "invalid_response"
            usage["requests"] = max(1, int(usage.get("requests", 0)))
            formatted_error = _format_llm_error(stage.replace("_", " "), exc)
            if failure_path:
                _record_failure(
                    failure_path,
                    cache_key=cache_key,
                    prompt_version=prompt_version,
                    model=self._config.model,
                    failure_kind=failure_status,
                    error=formatted_error,
                    usage=usage,
                    response=content,
                )
            return ClassificationOutcome(
                None,
                stage,
                failure_status,
                usage,
                estimate,
                warning="; ".join(
                    item
                    for item in (
                        cache_warning,
                        f"{stage} unavailable: {formatted_error}",
                    )
                    if item
                ),
                failure_history_path=failure_path,
            )

        usage["requests"] = max(1, int(usage.get("requests", 0)))
        if cache_path:
            write_json(
                cache_path,
                {
                    "status": "succeeded",
                    "prompt_version": prompt_version,
                    "model": self._config.model,
                    "result": value.model_dump(mode="json"),
                    "usage": usage,
                },
            )
        return ClassificationOutcome(
            value,
            stage,
            "llm",
            usage,
            estimate,
            warning=cache_warning,
        )


def strict_response_format(
    response_model: type[BaseModel],
    *,
    name: str,
) -> dict[str, Any]:
    """Build the strict Responses API format from the runtime validation contract."""
    schema = response_model.model_json_schema()
    _make_schema_strict(schema)
    return {
        "type": "json_schema",
        "name": name,
        "strict": True,
        "schema": schema,
    }


def _make_schema_strict(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _make_schema_strict(item)
        return
    if not isinstance(node, dict):
        return

    node.pop("default", None)
    node.pop("title", None)
    properties = node.get("properties")
    if isinstance(properties, dict):
        node["additionalProperties"] = False
        node["required"] = list(properties)
    for value in node.values():
        _make_schema_strict(value)


def aggregate_usage(*outcomes: ClassificationOutcome[Any]) -> dict[str, int]:
    usage: dict[str, int] = {}
    for outcome in outcomes:
        for key, value in outcome.usage.items():
            usage[key] = usage.get(key, 0) + int(value)
    return usage


def _cache_key(
    *,
    prompt_version: str,
    model: str,
    payload: dict[str, Any],
    response_format: dict[str, Any],
    max_output_tokens: int,
    reasoning_effort: str | None,
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "prompt_version": prompt_version,
                "model": model,
                "input": payload,
                "response_format": response_format,
                "max_output_tokens": max_output_tokens,
                "reasoning_effort": reasoning_effort,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _parse_structured_json_object(content: str) -> dict[str, Any]:
    """Parse strict structured output without salvaging nested objects from truncation."""
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"structured response was not complete JSON: {exc.msg} at character {exc.pos}"
        ) from None
    if not isinstance(value, dict):
        raise ValueError("structured response JSON must be an object")
    return value


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _record_failure(
    path: Path,
    *,
    cache_key: str,
    prompt_version: str,
    model: str,
    failure_kind: str,
    error: str,
    usage: dict[str, int],
    response: str,
) -> None:
    existing = _read_json_object(path) if path.is_file() else {}
    attempts = existing.get("attempts", [])
    if not isinstance(attempts, list):
        attempts = []
    attempts.append(
        {
            "attempted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "kind": failure_kind,
            "error": error,
            "usage": _normalize_usage(usage),
            "response": response or None,
        }
    )
    write_json(
        path,
        {
            "status": "failed",
            "cache_key": cache_key,
            "prompt_version": prompt_version,
            "model": model,
            "attempts": attempts,
        },
    )


def _normalize_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(key): int(item) for key, item in value.items() if isinstance(item, int | float)}
