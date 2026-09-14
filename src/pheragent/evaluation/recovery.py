from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import Field

from pheragent.deployment.models import ContractModel


class RecoveryUsage(ContractModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    cost_usd: float | None = Field(default=None, ge=0.0)


class RecoveryEvaluationReport(ContractModel):
    run_id: str
    failure_capture_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    sandbox_validation_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    live_repair_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_attempts_before_success: float | None = Field(default=None, ge=0.0)
    unresolved_failures: int = Field(ge=0)
    usage: RecoveryUsage


def evaluate_recovery(run_directory: Path) -> RecoveryEvaluationReport:
    """Measure recorded recovery behavior without judging its proposed explanation."""
    run = run_directory.expanduser().resolve(strict=True)
    execution = _read_object(run / "execution.json")
    bundle = _read_object(run / "failure-bundle.json")
    manifest = _read_object(run / "run-manifest.json")
    attempts = _objects(execution.get("attempts"))
    recoveries = _objects(execution.get("recoveries"))
    failures = _objects(bundle.get("failures"))
    failed_attempts = [attempt for attempt in attempts if attempt.get("status") == "failed"]
    validations = [
        recovery["validation"]
        for recovery in recoveries
        if isinstance(recovery.get("validation"), dict)
    ]
    resolved_steps = {
        str(recovery.get("step_id"))
        for recovery in recoveries
        if recovery.get("status") == "resolved"
    }
    attempted_steps = {str(recovery.get("step_id")) for recovery in recoveries}
    completed = {str(item) for item in execution.get("completed", [])}
    successful_repairs = resolved_steps & completed
    attempts_by_step = {
        step_id: sum(attempt.get("step_id") == step_id for attempt in attempts)
        for step_id in successful_repairs
    }
    usage = _usage(recoveries, manifest)
    return RecoveryEvaluationReport(
        run_id=str(manifest.get("run_id", run.name)),
        failure_capture_rate=_rate(len(failures), len(failed_attempts)),
        sandbox_validation_success_rate=_rate(
            sum(validation.get("succeeded") is True for validation in validations),
            len(validations),
        ),
        live_repair_success_rate=_rate(len(successful_repairs), len(attempted_steps)),
        mean_attempts_before_success=(
            mean(attempts_by_step.values()) if attempts_by_step else None
        ),
        unresolved_failures=len(execution.get("failed", [])),
        usage=usage,
    )


def _usage(recoveries: list[dict[str, Any]], manifest: dict[str, Any]) -> RecoveryUsage:
    totals: dict[str, int] = {}
    for recovery in recoveries:
        raw = recovery.get("usage")
        if not isinstance(raw, dict):
            continue
        for key, value in raw.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    input_tokens = totals.get("input_tokens", 0)
    output_tokens = totals.get("output_tokens", 0)
    reasoning_tokens = totals.get("reasoning_tokens", 0)
    return RecoveryUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=totals.get(
            "total_tokens",
            input_tokens + output_tokens,
        ),
        duration_seconds=float(manifest.get("duration_seconds", 0.0)),
    )


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _objects(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
