"""Hunyuan3D-2 model weight resolver and lazy loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ai3d.core.logging import get_logger
from ai3d.core.paths import ai_models_root

_log = get_logger(__name__)

_DEFAULT_LOCAL_SHAPE = "vision/hunyuan3d-2"
_DEFAULT_LOCAL_PAINT = "vision/hunyuan3d-2-paint"
_HF_REPO_SHAPE = "tencent/Hunyuan3D-2"
_HF_REPO_PAINT = "tencent/Hunyuan3D-2"


class Hunyuan3DLoader:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        paint_path: Optional[Path] = None,
        device: str = "cuda",
        enable_texture: bool = True,
    ) -> None:
        self._device = device
        self._explicit_path = model_path
        self._explicit_paint_path = paint_path
        self._enable_texture = enable_texture
        self._shape_pipeline: Any = None
        self._paint_pipeline: Any = None

    def resolve_shape_path(self) -> Optional[Path]:
        candidates: list[Path] = []

        if self._explicit_path:
            candidates.append(self._explicit_path)

        try:
            from ai3d.registry.model_registry import ModelRegistry
            entry = ModelRegistry().get("hunyuan3d-2")
            candidates.append(Path(entry.local_path))
        except Exception:
            pass

        candidates.append(ai_models_root() / _DEFAULT_LOCAL_SHAPE)

        for p in candidates:
            if p.exists():
                _log.debug("Hunyuan3D shape weights found at: %s", p)
                return p

        return None

    def resolve_paint_path(self) -> Optional[Path]:
        candidates: list[Path] = []

        if self._explicit_paint_path:
            candidates.append(self._explicit_paint_path)

        candidates.append(ai_models_root() / _DEFAULT_LOCAL_PAINT)

        for p in candidates:
            if p.exists():
                _log.debug("Hunyuan3D paint weights found at: %s", p)
                return p

        return None

    def is_available(self) -> bool:
        if self.resolve_shape_path() is None:
            return False
        try:
            import hy3dgen  # type: ignore[import]  # noqa: F401
            return True
        except ImportError:
            return False

    def load_shape(self) -> Any:
        if self._shape_pipeline is not None:
            return self._shape_pipeline

        path = self.resolve_shape_path()
        if path is None:
            raise RuntimeError(
                f"Hunyuan3D shape weights not found. Download:\n"
                f"  huggingface-cli download {_HF_REPO_SHAPE} --local-dir "
                f"{ai_models_root() / _DEFAULT_LOCAL_SHAPE}"
            )

        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "hy3dgen package not installed. "
                "Install from https://github.com/tencent/Hunyuan3D-2"
            ) from exc

        _log.info("Loading Hunyuan3D shape pipeline from %s", path)
        self._shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(path), device=self._device
        )
        return self._shape_pipeline

    def load_paint(self) -> Any:
        if self._paint_pipeline is not None:
            return self._paint_pipeline

        paint_path = self.resolve_paint_path()
        if paint_path is None:
            _log.warning("Hunyuan3D paint weights not found; texture baking disabled")
            return None

        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline  # type: ignore[import]
        except ImportError:
            _log.warning("hy3dgen.texgen not available; texture baking disabled")
            return None

        _log.info("Loading Hunyuan3D paint pipeline from %s", paint_path)
        self._paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(str(paint_path))
        self._paint_pipeline.to(self._device)
        return self._paint_pipeline

    def get_model_paths(self) -> tuple[list[str], list[str]]:
        found, missing = [], []
        shape = self.resolve_shape_path()
        if shape:
            found.append(str(shape))
        else:
            missing.append(str(ai_models_root() / _DEFAULT_LOCAL_SHAPE))
        paint = self.resolve_paint_path()
        if paint:
            found.append(str(paint))
        else:
            missing.append(str(ai_models_root() / _DEFAULT_LOCAL_PAINT))
        return found, missing
