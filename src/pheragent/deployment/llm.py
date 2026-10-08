from __future__ import annotations

import json
import os
import re
from typing import Any

from pydantic import BaseModel

from pheragent.llm_planner import (
    IncompleteLLMResponseError,
    _format_llm_error,
    _openai_client,
    _read_streamed_response_with_usage,
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
    ) -> tuple[T, dict[str, int]]:
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
                    "text": {"format": _response_format(response_model, name)},
                    "stream": True,
                }
                if output_limit is not None:
                    request["max_output_tokens"] = output_limit
                if self.reasoning_effort:
                    request["reasoning"] = {"effort": self.reasoning_effort}
                content, usage = _read_streamed_response_with_usage(
                    client.responses.create(**request),
                    error_context=name.replace("_", " "),
                )
                value = response_model.model_validate(_parse_object(content))
            except Exception as exc:
                status = "invalid_response" if isinstance(exc, ValueError) else "failed"
                if isinstance(exc, IncompleteLLMResponseError):
                    content = exc.content
                    usage = exc.usage
                    reason = re.sub(r"\W+", "_", exc.reason.casefold()).strip("_")
                    status = f"incomplete_{reason or 'response'}"
                usage["requests"] = max(1, int(usage.get("requests", 0)))
                message = _format_llm_error(name.replace("_", " "), exc)
                set_output(active_span, {"status": status, "error": message}, usage)
                raise LLMError(
                    message,
                    status=status,
                    response=content,
                    usage=usage,
                ) from exc

            usage["requests"] = max(1, int(usage.get("requests", 0)))
            set_output(active_span, {"status": "succeeded", "response": content}, usage)
            return value, usage

def _minimum(*values: int | None) -> int | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None

def _schema_name(response_model: type[BaseModel]) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", response_model.__name__).lower()

def _response_format(response_model: type[BaseModel], name: str) -> dict[str, Any]:
    schema = response_model.model_json_schema()
    _make_schema_strict(schema)
    return {"type": "json_schema", "name": name, "strict": True, "schema": schema}

def _make_schema_strict(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _make_schema_strict(item)
        return
    if not isinstance(node, dict):
        return

    properties = node.get("properties")
    node.pop("default", None)
    node.pop("title", None)
    if isinstance(properties, dict):
        node["additionalProperties"] = False
        node["required"] = list(properties)
        for value in properties.values():
            _make_schema_strict(value)
    for key, value in node.items():
        if key != "properties":
            _make_schema_strict(value)

def _parse_object(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"structured response was not complete JSON: {exc.msg} at character {exc.pos}"
        ) from None
    if not isinstance(value, dict):
        raise ValueError("structured response JSON must be an object")
    return value
