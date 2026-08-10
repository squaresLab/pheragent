from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pheragent.llm_planner import (
    _format_llm_error,
    _openai_client,
    _parse_json_object,
    _read_streamed_response_with_usage,
    _resolve_openai_base_url,
)

from .analysis_models import (
    AnalysisBlockType,
    AnalysisQuestion,
    CandidateComponent,
    ComponentClassificationAssignment,
    ComponentClassificationResponse,
    DeploymentSignalBundle,
)
from .serialization import write_json

_PROMPT_VERSION = "phase1-component-classification-v1"
_SYSTEM_PROMPT = """Classify each discovered deployment component by functional purpose.
Return exactly one assignment for every C-prefixed component ID supplied by the schema. Never
return or classify B-prefixed provided block IDs. A locked classification is authoritative and
must be returned unchanged. For an unlocked component, use the deterministic hint and the
deployment relations to choose a concise snake_case type, subtype, and optional domain. Do not
create deployment commands, paths, technologies, components, IDs, or grouping IDs. Execution
order alone is not a hard dependency. Return structured JSON only."""


@dataclass(slots=True)
class AnalysisLLMConfig:
    mode: str = "auto"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    base_url: str | None = None
    timeout: float = 120.0
    max_output_tokens: int = 3000
    cache_dir: Path | None = None
    retry_failed: bool = False


@dataclass(frozen=True, slots=True)
class ComponentClassificationOutcome:
    classification: ComponentClassificationResponse | None
    used: str
    usage: dict[str, int]
    input_tokens_estimate: int
    warning: str | None = None
    failure_history_path: Path | None = None


def classify_components_with_llm(
    signals: DeploymentSignalBundle,
    *,
    config: AnalysisLLMConfig,
) -> ComponentClassificationOutcome:
    mode = config.mode.strip().casefold()
    if mode not in {"auto", "deterministic", "llm"}:
        raise ValueError(f"unsupported analysis synthesizer: {config.mode}")
    compact_input = build_compact_classification_input(signals)
    serialized = json.dumps(compact_input, sort_keys=True, separators=(",", ":"))
    estimate = max(1, len(serialized) // 4)
    if mode == "deterministic":
        return ComponentClassificationOutcome(None, "deterministic", {}, estimate)
    if not signals.candidate_components:
        return ComponentClassificationOutcome(
            None,
            "deterministic-no-components",
            {},
            estimate,
            warning="LLM classification skipped because no component candidates were discovered",
        )
    api_key = os.getenv(config.api_key_env)
    if not api_key:
        if mode == "llm":
            raise RuntimeError(f"missing API key in env var {config.api_key_env}")
        return ComponentClassificationOutcome(None, "deterministic-no-key", {}, estimate)

    cache_key = hashlib.sha256(
        json.dumps(
            {
                "prompt_version": _PROMPT_VERSION,
                "model": config.model,
                "input": compact_input,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    cache_path = config.cache_dir / f"{cache_key}.json" if config.cache_dir else None
    failure_path = (
        config.cache_dir / "failures" / f"{cache_key}.json" if config.cache_dir else None
    )
    if cache_path and cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        classification, issues = _normalize_response(cached["classification"], signals)
        cached_issues = cached.get("repair_issues", [])
        if isinstance(cached_issues, list):
            issues = [*map(str, cached_issues), *issues]
        issues = list(dict.fromkeys(issues))
        return ComponentClassificationOutcome(
            classification,
            "llm-cache-with-fallback" if issues else "llm-cache",
            {key: int(value) for key, value in cached.get("usage", {}).items()},
            estimate,
            warning=_fallback_warning(issues),
        )

    if failure_path and failure_path.is_file() and not config.retry_failed:
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        attempts = failure.get("attempts", [])
        last_attempt = attempts[-1] if attempts else {}
        error = str(last_attempt.get("error", "previous matching LLM request failed"))
        if mode == "llm":
            raise RuntimeError(
                "deployment component classification skipped because an identical request "
                "previously "
                f"failed: {error}; use --retry-failed-llm to retry explicitly"
            )
        return ComponentClassificationOutcome(
            None,
            "llm-failure-cache",
            {},
            estimate,
            warning=(
                "Skipped an LLM request because the identical model/input/prompt contract "
                f"previously failed; used deterministic classification: {error}"
            ),
            failure_history_path=failure_path,
        )

    base_url = _resolve_openai_base_url(
        configured_base_url=config.base_url,
        base_url_env=config.base_url_env,
        api_mode="responses",
    )
    content = ""
    usage: dict[str, int] = {}
    attempted = False
    try:
        client = _openai_client(base_url=base_url, api_key=api_key, timeout=config.timeout)
        attempted = True
        stream = client.responses.create(
            model=config.model,
            instructions=_SYSTEM_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(compact_input, ensure_ascii=False),
                        }
                    ],
                }
            ],
            text={"format": _response_format(signals)},
            max_output_tokens=config.max_output_tokens,
            stream=True,
        )
        content, usage = _read_streamed_response_with_usage(
            stream, error_context="deployment component classification"
        )
        classification, issues = _normalize_response(
            _parse_json_object(content), signals
        )
    except Exception as exc:
        if attempted:
            usage["requests"] = max(1, int(usage.get("requests", 0)))
        formatted_error = _format_llm_error("deployment component classification", exc)
        if failure_path:
            _record_failure(
                failure_path,
                cache_key=cache_key,
                model=config.model,
                error=formatted_error,
                usage=usage,
                response=content,
            )
        if mode == "llm":
            raise RuntimeError(formatted_error) from exc
        return ComponentClassificationOutcome(
            None,
            "llm-failed-deterministic",
            usage,
            estimate,
            warning=(
                "LLM component classification unavailable; used deterministic classification: "
                f"{formatted_error}"
            ),
            failure_history_path=failure_path,
        )
    usage["requests"] = max(1, int(usage.get("requests", 0)))
    if cache_path:
        write_json(
            cache_path,
            {
                "status": "succeeded",
                "prompt_version": _PROMPT_VERSION,
                "model": config.model,
                "classification": classification.model_dump(mode="json"),
                "repair_issues": issues,
                "usage": usage,
            },
        )
    return ComponentClassificationOutcome(
        classification,
        "llm-with-fallback" if issues else "llm",
        usage,
        estimate,
        warning=_fallback_warning(issues),
    )


def build_compact_classification_input(signals: DeploymentSignalBundle) -> dict[str, Any]:
    """Return the compact, source-free payload supplied for classification."""
    return {
        "contract": {
            "reserved_block_ids": [block.id for block in signals.context_blocks],
            "required_component_ids": [
                component.id for component in signals.candidate_components
            ],
            "rules": [
                "Return one assignment for every required C-prefixed component ID.",
                "Never return a B-prefixed provided block ID.",
                "Keep locked classifications unchanged.",
                "Do not invent IDs or grouping identifiers.",
            ],
        },
        "provided_blocks": [
            {
                "id": block.id,
                "type": block.type,
                "subtype": block.subtype,
                "implementation": block.implementation,
                "provides": block.provides,
            }
            for block in signals.context_blocks
        ],
        "components": [
            {
                "id": component.id,
                "name": component.name,
                "implementation": component.implementation,
                "entrypoint": (
                    component.deployment.entrypoint if component.deployment else None
                ),
                "classification_hint": {
                    "block_type": component.classification.block_type,
                    "subtype": component.classification.subtype,
                    "domain": component.classification.domain,
                    "confidence": component.classification.confidence,
                },
                "classification_locked": _classification_locked(component),
            }
            for component in signals.candidate_components
        ],
        "stages": [
            {
                "id": stage.id,
                "entrypoint": stage.entrypoint.path,
                "component_ids": stage.component_ids,
            }
            for stage in signals.deployment_stages
        ],
        "relations": [
            {
                "source": relation.source,
                "target": relation.target,
                "relation": relation.relation,
                "strength": relation.strength,
            }
            for relation in signals.relations
        ],
    }


def _response_format(signals: DeploymentSignalBundle) -> dict[str, Any]:
    component_properties = {
        component.id: _assignment_schema(component)
        for component in signals.candidate_components
    }
    component_ids = list(component_properties)
    return {
        "type": "json_schema",
        "name": "component_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "assignments": {
                    "type": "object",
                    "properties": component_properties,
                    "required": component_ids,
                    "additionalProperties": False,
                },
                "unresolved": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "minLength": 1},
                            "reason": {"type": "string", "minLength": 1},
                        },
                        "required": ["question", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["assignments", "unresolved"],
            "additionalProperties": False,
        },
    }


def _assignment_schema(component: CandidateComponent) -> dict[str, Any]:
    classification = component.classification
    locked = _classification_locked(component)
    block_types = (
        [classification.block_type.value]
        if locked
        else [item.value for item in AnalysisBlockType]
    )
    subtype: dict[str, Any] = {"type": "string"}
    domain: dict[str, Any]
    if locked:
        subtype["enum"] = [classification.subtype]
        domain = (
            {"type": "null", "enum": [None]}
            if classification.domain is None
            else {"type": "string", "enum": [classification.domain]}
        )
    else:
        subtype["pattern"] = r"^[a-z0-9]+(?:_[a-z0-9]+)*$"
        domain = {
            "type": ["string", "null"],
            "pattern": r"^[a-z0-9]+(?:_[a-z0-9]+)*$",
        }
    return {
        "type": "object",
        "properties": {
            "block_type": {"type": "string", "enum": block_types},
            "subtype": subtype,
            "domain": domain,
        },
        "required": ["block_type", "subtype", "domain"],
        "additionalProperties": False,
    }


def _classification_locked(component: CandidateComponent) -> bool:
    return component.classification.confidence >= 1.0


def _normalize_response(
    payload: dict[str, Any],
    signals: DeploymentSignalBundle,
) -> tuple[ComponentClassificationResponse, list[str]]:
    raw_assignments = payload.get("assignments")
    if not isinstance(raw_assignments, dict):
        raw_assignments = {}
    issues: list[str] = []
    expected = {component.id for component in signals.candidate_components}
    unknown = sorted(set(raw_assignments) - expected)
    if unknown:
        issues.append(f"ignored unknown component IDs: {', '.join(unknown)}")

    assignments: dict[str, ComponentClassificationAssignment] = {}
    deterministic_fallbacks: list[str] = []
    preserved_locks: list[str] = []
    for component in signals.candidate_components:
        fallback = ComponentClassificationAssignment(
            block_type=component.classification.block_type,
            subtype=component.classification.subtype,
            domain=component.classification.domain,
        )
        raw_assignment = raw_assignments.get(component.id)
        try:
            assignment = ComponentClassificationAssignment.model_validate(raw_assignment)
        except (TypeError, ValueError):
            assignments[component.id] = fallback
            deterministic_fallbacks.append(component.id)
            continue
        if _classification_locked(component) and assignment != fallback:
            assignments[component.id] = fallback
            preserved_locks.append(component.id)
            continue
        assignments[component.id] = assignment
    if deterministic_fallbacks:
        issues.append(
            "used deterministic classification for "
            + _summarize_ids(deterministic_fallbacks)
        )
    if preserved_locks:
        issues.append("preserved locked classification for " + _summarize_ids(preserved_locks))

    unresolved: list[AnalysisQuestion] = []
    raw_unresolved = payload.get("unresolved", [])
    if not isinstance(raw_unresolved, list):
        raw_unresolved = []
        issues.append("ignored invalid unresolved questions")
    for item in raw_unresolved:
        try:
            unresolved.append(AnalysisQuestion.model_validate(item))
        except (TypeError, ValueError):
            issues.append("ignored an invalid unresolved question")
    return ComponentClassificationResponse(
        assignments=assignments,
        unresolved=unresolved,
    ), issues


def _fallback_warning(issues: list[str]) -> str | None:
    if not issues:
        return None
    return "LLM classification was repaired with deterministic fallbacks: " + "; ".join(issues)


def _summarize_ids(component_ids: list[str], *, limit: int = 5) -> str:
    visible = ", ".join(component_ids[:limit])
    remaining = len(component_ids) - limit
    return f"{visible} (+{remaining} more)" if remaining > 0 else visible


def _record_failure(
    path: Path,
    *,
    cache_key: str,
    model: str,
    error: str,
    usage: dict[str, int],
    response: str,
) -> None:
    existing: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, ValueError):
            existing = {}
    attempts = existing.get("attempts", [])
    if not isinstance(attempts, list):
        attempts = []
    attempts.append(
        {
            "attempted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
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
            "prompt_version": _PROMPT_VERSION,
            "model": model,
            "attempts": attempts,
        },
    )


def _normalize_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): int(item)
        for key, item in value.items()
        if isinstance(item, int | float)
    }
