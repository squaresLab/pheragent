from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel

from pheragent.llm_planner import (
    _extract_token_usage,
    _format_llm_error,
    _openai_client,
    _resolve_openai_base_url,
)

from .telemetry import set_output, span

DEFAULT_MODEL = "gpt-5.6-terra"


class LLMError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: str = "failed",
        response: str = "",
        usage: dict[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.response = response
        self.usage = usage or {}

class LLMClient:
    """Return structurally validated model output."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        base_url: str | None = None,
        timeout: float = 120.0,
        max_output_tokens: int | None = 5000,
        reasoning_effort: str | None = None,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.base_url = base_url
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort

    def complete[T: BaseModel](
        self,
        response_model: type[T],
        *,
        instructions: str,
        payload: dict[str, Any],
        max_output_tokens: int | None = None,
        max_attempts: int = 2,
    ) -> tuple[T, dict[str, int]]:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise LLMError(
                f"{self.api_key_env} is not set",
                status="not_requested_no_key",
            )

        output_limit = _minimum(self.max_output_tokens, max_output_tokens)
        name = _schema_name(response_model)
        content = ""
        usage: dict[str, int] = {}
        attempts = 0
        with span(
            f"llm.{name}",
            span_type="LLM",
            input={"instructions": instructions, "payload": payload},
            **{"gen_ai.request.model": self.model},
        ) as active_span:
            try:
                client = _openai_client(
                    base_url=_resolve_openai_base_url(
                        configured_base_url=self.base_url,
                        base_url_env=self.base_url_env,
                        api_mode="responses",
                    ),
                    api_key=api_key,
                    timeout=self.timeout,
                )
                request: dict[str, Any] = {
                    "model": self.model,
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
                }
                if output_limit is not None:
                    request["max_output_tokens"] = output_limit
                if self.reasoning_effort:
                    request["reasoning"] = {"effort": self.reasoning_effort}
                for attempts in range(1, max_attempts + 1):
                    try:
                        response = client.responses.parse(**request, text_format=response_model)
                        content = response.output_text
                        usage = _extract_token_usage(response.usage) or {}
                        value = response.output_parsed
                        if value is None:
                            raise ValueError("structured response did not contain a parsed object")
                        break
                    except ValueError:
                        if attempts == max_attempts:
                            raise
            except Exception as exc:
                status = "invalid_response" if isinstance(exc, ValueError) else "failed"
                usage["requests"] = max(attempts, 1, int(usage.get("requests", 0)))
                message = _format_llm_error(name.replace("_", " "), exc)
                set_output(active_span, {"status": status, "error": message}, usage)
                raise LLMError(
                    message,
                    status=status,
                    response=content,
                    usage=usage,
                ) from exc

            usage["requests"] = max(attempts, 1, int(usage.get("requests", 0)))
            set_output(active_span, {"status": "succeeded", "response": content}, usage)
            return value, usage

def _minimum(*values: int | None) -> int | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None

def _schema_name(response_model: type[BaseModel]) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", response_model.__name__).lower()
