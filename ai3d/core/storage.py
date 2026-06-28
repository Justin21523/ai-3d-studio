"""Storage helpers for YAML-backed Pydantic model persistence."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    ensure_directory(path.parent)
    path.write_text(yaml.safe_dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")


def read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_model(path: Path, model: BaseModel) -> None:
    write_yaml(path, model.model_dump(mode="json"))


def read_model(path: Path, model_cls: Type[T]) -> T:
    return model_cls.model_validate(read_yaml(path))
