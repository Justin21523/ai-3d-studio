"""InstantMesh model weight resolver and lazy loader.

InstantMesh has no pip package — expects source installation at the path configured
in paths.yaml (instantmesh_root) or the default /mnt/c/ai_tools/InstantMesh.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from ai3d.core.logging import get_logger
from ai3d.core.paths import ai_models_root, ai_tools_root

_log = get_logger(__name__)

_DEFAULT_LOCAL = "vision/instantmesh"
_DEFAULT_SOURCE = "InstantMesh"
_HF_REPO = "TencentARC/InstantMesh"


class InstantMeshLoader:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda",
    ) -> None:
        self._device = device
        self._explicit_path = model_path
        self._zero123_pipeline: Any = None
        self._lrm_model: Any = None

    def _source_root(self) -> Path:
        return ai_tools_root() / _DEFAULT_SOURCE

    def resolve_weight_path(self) -> Optional[Path]:
        candidates: list[Path] = []

        if self._explicit_path:
            candidates.append(self._explicit_path)

        try:
            from ai3d.registry.model_registry import ModelRegistry
            entry = ModelRegistry().get("instantmesh")
            candidates.append(Path(entry.local_path))
        except Exception:
            pass

        candidates.append(ai_models_root() / _DEFAULT_LOCAL)

        for p in candidates:
            if p.exists():
                _log.debug("InstantMesh weights found at: %s", p)
                return p

        return None

    def _ensure_source_on_path(self) -> bool:
        src = self._source_root()
        if not src.exists():
            return False
        src_str = str(src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
        return True

    def is_available(self) -> bool:
        if self.resolve_weight_path() is None:
            return False
        if not self._ensure_source_on_path():
            return False
        try:
            from diffusers import DiffusionPipeline  # type: ignore[import]  # noqa: F401
            return True
        except ImportError:
            return False

    def load_zero123plus(self) -> Any:
        if self._zero123_pipeline is not None:
            return self._zero123_pipeline

        try:
            from diffusers import DiffusionPipeline  # type: ignore[import]
            import torch
        except ImportError as exc:
            raise ImportError("diffusers package required for InstantMesh") from exc

        _log.info("Loading Zero123++ for InstantMesh multi-view generation")
        self._zero123_pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16,
        )
        self._zero123_pipeline.to(self._device)
        return self._zero123_pipeline

    def load_lrm(self) -> Any:
        if self._lrm_model is not None:
            return self._lrm_model

        weight_path = self.resolve_weight_path()
        if weight_path is None:
            raise RuntimeError(
                f"InstantMesh weights not found. Download from:\n"
                f"  huggingface-cli download {_HF_REPO} --local-dir "
                f"{ai_models_root() / _DEFAULT_LOCAL}"
            )

        if not self._ensure_source_on_path():
            raise RuntimeError(
                f"InstantMesh source not found at {self._source_root()}. "
                "Clone from https://github.com/TencentARC/InstantMesh"
            )

        try:
            import torch
            from src.models.lrm_mesh import InstantMesh  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "InstantMesh source not on path — check instantmesh_root in configs/paths.yaml"
            ) from exc

        _log.info("Loading InstantMesh LRM from %s", weight_path)
        ckpt = torch.load(str(weight_path / "model.ckpt"), map_location="cpu")
        cfg = ckpt.get("hyper_parameters", {})
        self._lrm_model = InstantMesh(**cfg.get("model_cfg", {}))
        self._lrm_model.load_state_dict(ckpt["state_dict"])
        self._lrm_model.to(self._device).eval()
        return self._lrm_model

    def get_model_paths(self) -> tuple[list[str], list[str]]:
        path = self.resolve_weight_path()
        if path:
            return [str(path)], []
        return [], [str(ai_models_root() / _DEFAULT_LOCAL)]
