"""Mesh2Splat dependency resolver for the CPU conversion path."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ai3d.core.paths import ai_models_root

_DEFAULT_LOCAL = "vision/mesh2splat"


class Mesh2SplatLoader:
    def __init__(self, model_path: Optional[Path] = None) -> None:
        self._explicit_path = model_path

    def resolve_weight_path(self) -> Optional[Path]:
        candidates: list[Path] = []
        if self._explicit_path:
            candidates.append(self._explicit_path)

        try:
            from ai3d.registry.model_registry import ModelRegistry
            entry = ModelRegistry().get("mesh2splat")
            candidates.append(Path(entry.local_path))
        except Exception:
            pass

        candidates.append(ai_models_root() / _DEFAULT_LOCAL)
        return next((path for path in candidates if path.exists()), None)

    def is_available(self) -> bool:
        try:
            import open3d  # type: ignore[import]  # noqa: F401
            import trimesh  # type: ignore[import]  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> "Mesh2SplatLoader":
        try:
            import trimesh  # type: ignore[import]  # noqa: F401
        except ImportError as exc:
            raise ImportError("trimesh package required for Mesh2Splat CPU conversion") from exc
        return self

    def get_model_paths(self) -> tuple[list[str], list[str]]:
        path = self.resolve_weight_path()
        if path:
            return [str(path)], []
        return [], [str(ai_models_root() / _DEFAULT_LOCAL)]
