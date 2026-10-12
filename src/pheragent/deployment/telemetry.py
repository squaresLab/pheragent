"""Optional OpenTelemetry export for deployment runs."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager

from .redaction import redact_secrets

_provider = None
_tracer = None


def _configure() -> None:
    global _provider, _tracer
    if _tracer is not None:
        return
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    project_key = os.getenv("LMNR_PROJECT_API_KEY")
    if not endpoint and not project_key:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise RuntimeError(
            "install observability support with: uv sync --extra observability"
        ) from exc

    endpoint = endpoint or "https://api.lmnr.ai/v1/traces"
    headers = {"Authorization": f"Bearer {project_key}"} if project_key else None
    _provider = TracerProvider(
        resource=Resource.create(
            {"service.name": os.getenv("OTEL_SERVICE_NAME", "pheragent-deployment")}
        )
    )
    _provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
    )
    trace.set_tracer_provider(_provider)
    _tracer = trace.get_tracer("pheragent.deployment")


def _content(value) -> str | None:
    if os.getenv("PHERAGENT_TRACE_CONTENT") != "1":
        return None
    return redact_secrets(json.dumps(value, ensure_ascii=False, default=str))


@contextmanager
def span(name: str, *, span_type: str = "DEFAULT", input=None, **attributes):
    _configure()
    if _tracer is None:
        yield None
        return
    clean = {key: value for key, value in attributes.items() if value is not None}
    clean["lmnr.span.type"] = span_type
    with _tracer.start_as_current_span(name, attributes=clean) as active:
        if content := _content(input):
            active.set_attribute("lmnr.span.input", content)
        yield active


def record_event(event: str, data: dict) -> None:
    if _tracer is None:
        return
    value = data.get("value", {})
    result = data.get("result", {})
    attributes = {
        "pheragent.event": event,
        "pheragent.iteration": data.get("iteration"),
        "pheragent.tool": data.get("tool") or value.get("tool"),
        "pheragent.focus": value.get("focus"),
        "pheragent.reason": value.get("reason") or result.get("reason"),
        "pheragent.status": result.get("status"),
    }
    with span(
        f"agent.{event}",
        span_type="TOOL" if event == "tool" else "DEFAULT",
        input=data,
        **attributes,
    ):
        pass


def set_output(active, value, usage: dict[str, int] | None = None) -> None:
    if active is None:
        return
    if content := _content(value):
        active.set_attribute("lmnr.span.output", content)
    for name, amount in (usage or {}).items():
        if isinstance(amount, int):
            active.set_attribute(f"gen_ai.usage.{name}", amount)


def flush() -> None:
    if _provider is not None:
        _provider.force_flush(timeout_millis=5000)
