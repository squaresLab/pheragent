from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from .models import SourcesConfig


def load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        raise ValueError(f"YAML document is empty: {path}")
    return payload


def load_sources_config(path: Path) -> SourcesConfig:
    return SourcesConfig.model_validate(load_yaml(path))


def write_yaml(path: Path, model: BaseModel | Any) -> None:
    payload = _jsonable(model)
    rendered = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    _atomic_write_text(path, rendered)


def write_json(path: Path, model: BaseModel | Any) -> None:
    payload = _jsonable(model)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    _atomic_write_text(path, rendered)


def write_jsonl(path: Path, models: Iterable[BaseModel]) -> None:
    lines = [
        json.dumps(model.model_dump(mode="json", exclude_none=True), ensure_ascii=False)
        for model in models
    ]
    rendered = "\n".join(lines)
    if lines:
        rendered += "\n"
    _atomic_write_text(path, rendered)


def write_text(path: Path, content: str) -> None:
    _atomic_write_text(path, content)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
