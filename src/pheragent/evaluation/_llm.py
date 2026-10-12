from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pheragent.deployment.llm import LLMClient, LLMError
from pheragent.deployment.serialization import write_json


@dataclass(frozen=True, slots=True)
class EvaluationResult[T: BaseModel]:
    value: T | None
    stage: str
    status: str
    usage: dict[str, int]
    input_tokens_estimate: int
    warning: str | None = None


class EvaluationLLMClient:
    """Add reproducible evaluation caching and failure records to an LLM client."""

    def __init__(
        self,
        client: LLMClient,
        *,
        max_requests: int,
        cache_dir: Path | None,
        retry_failed: bool,
        refresh_cache: bool,
    ) -> None:
        self.client = client
        self.max_requests = max_requests
        self.cache_dir = cache_dir
        self.retry_failed = retry_failed
        self.refresh_cache = refresh_cache
        self.requests = 0

    def complete[T: BaseModel](
        self,
        response_model: type[T],
        *,
        stage: str,
        prompt_version: str,
        instructions: str,
        payload: dict[str, Any],
        validate: Callable[[T], None],
    ) -> EvaluationResult[T]:
        estimate = max(1, len(json.dumps(payload, sort_keys=True, separators=(",", ":"))) // 4)
        if not os.getenv(self.client.api_key_env):
            return EvaluationResult(
                None,
                stage,
                "not_requested_no_key",
                {},
                estimate,
                f"{stage} skipped because {self.client.api_key_env} is not set",
            )

        key = _cache_key(prompt_version, self.client, payload, response_model)
        success_path = self.cache_dir / stage / f"{key}.json" if self.cache_dir else None
        failure_path = (
            self.cache_dir / "failures" / stage / f"{key}.json" if self.cache_dir else None
        )
        warning = None
        if not self.refresh_cache and success_path and success_path.is_file():
            try:
                value = response_model.model_validate(_read_json(success_path)["result"])
                validate(value)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                warning = f"ignored invalid {stage} success cache: {exc}"
            else:
                return EvaluationResult(value, stage, "cache", {}, estimate)

        if (
            not self.refresh_cache
            and failure_path
            and failure_path.is_file()
            and not self.retry_failed
        ):
            failure = _read_json(failure_path)
            attempts = failure.get("attempts", [])
            last = attempts[-1] if isinstance(attempts, list) and attempts else {}
            try:
                value = response_model.model_validate_json(str(last.get("response", "")))
                validate(value)
            except (TypeError, ValueError):
                error = str(last.get("error", "previous matching request failed"))
                return EvaluationResult(
                    None,
                    stage,
                    "failure_cache",
                    {},
                    estimate,
                    f"skipped {stage} because the identical request previously failed: {error}; "
                    "use --retry-failed-llm to retry",
                )
            if success_path:
                _write_success(success_path, prompt_version, self.client, value, {})
            return EvaluationResult(
                value,
                stage,
                "recovered_failure_cache",
                {},
                estimate,
                f"recovered {stage} from its preserved failed response",
            )

        if self.requests >= self.max_requests:
            return EvaluationResult(
                None,
                stage,
                "request_budget_exhausted",
                {},
                estimate,
                f"{stage} skipped because the run reached its "
                f"{self.max_requests}-request LLM budget",
            )
        self.requests += 1

        started = time.monotonic()
        response = ""
        usage: dict[str, int] = {}
        try:
            value, usage = self.client.complete(
                response_model,
                instructions=instructions,
                payload=payload,
            )
            response = value.model_dump_json()
            validate(value)
        except (LLMError, ValueError) as exc:
            if isinstance(exc, LLMError):
                status, response, usage = exc.status, exc.response, exc.usage
            else:
                status = "invalid_response"
            message = str(exc)
            if failure_path:
                _record_failure(
                    failure_path,
                    key=key,
                    prompt_version=prompt_version,
                    client=self.client,
                    status=status,
                    error=message,
                    usage=usage,
                    response=response,
                    duration=time.monotonic() - started,
                )
            return EvaluationResult(
                None,
                stage,
                status,
                usage,
                estimate,
                "; ".join(item for item in (warning, f"{stage} unavailable: {message}") if item),
            )

        if success_path:
            _write_success(success_path, prompt_version, self.client, value, usage)
        return EvaluationResult(value, stage, "llm", usage, estimate, warning)


def aggregate_usage(results: list[EvaluationResult[Any]]) -> dict[str, int]:
    usage: dict[str, int] = {}
    for result in results:
        for key, value in result.usage.items():
            usage[key] = usage.get(key, 0) + int(value)
    return usage


def _cache_key(
    prompt_version: str,
    client: LLMClient,
    payload: dict[str, Any],
    response_model: type[BaseModel],
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "prompt_version": prompt_version,
                "model": client.model,
                "input": payload,
                "schema": response_model.model_json_schema(),
                "max_output_tokens": client.max_output_tokens,
                "reasoning_effort": client.reasoning_effort,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_success(
    path: Path,
    prompt_version: str,
    client: LLMClient,
    value: BaseModel,
    usage: dict[str, int],
) -> None:
    write_json(
        path,
        {
            "status": "succeeded",
            "prompt_version": prompt_version,
            "model": client.model,
            "result": value.model_dump(mode="json"),
            "usage": usage,
        },
    )


def _record_failure(
    path: Path,
    *,
    key: str,
    prompt_version: str,
    client: LLMClient,
    status: str,
    error: str,
    usage: dict[str, int],
    response: str,
    duration: float,
) -> None:
    attempts = _read_json(path).get("attempts", []) if path.is_file() else []
    if not isinstance(attempts, list):
        attempts = []
    attempts.append(
        {
            "attempted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "kind": status,
            "error": error,
            "usage": usage,
            "response": response or None,
            "duration_seconds": round(duration, 6),
        }
    )
    write_json(
        path,
        {
            "status": "failed",
            "cache_key": key,
            "prompt_version": prompt_version,
            "model": client.model,
            "attempts": attempts,
        },
    )
