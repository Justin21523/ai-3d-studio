"""TripoSR model weight resolver and lazy loader."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from ai3d.core.logging import get_logger
from ai3d.core.paths import ai_models_root, ai_tools_root

_log = get_logger(__name__)

_DEFAULT_LOCAL = "vision/triposr"
_DEFAULT_SOURCE = "TripoSR"
_HF_REPO = "stabilityai/TripoSR"


class TripoSRLoader:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda",
    ) -> None:
        self._device = device
        self._explicit_path = model_path
        self._model: Any = None

    # ── Path resolution ───────────────────────────────────────────────────────

    def resolve_path(self) -> Optional[Path]:
        """Return the first weight directory that actually exists on disk."""
        candidates: list[Path] = []

        if self._explicit_path:
            candidates.append(self._explicit_path)

        # Registry entry path
        try:
            from ai3d.registry.model_registry import ModelRegistry
            entry = ModelRegistry().get("triposr")
            candidates.append(Path(entry.local_path))
        except Exception:
            pass

        # Default convention path
        candidates.append(ai_models_root() / _DEFAULT_LOCAL)

        for p in candidates:
            if p.exists():
                _log.debug("TripoSR weights found at: %s", p)
                return p

        return None

    def is_available(self) -> bool:
        if self.resolve_path() is None:
            return False
        if not self._ensure_source_on_path():
            return False
        try:
            from tsr.system import TSR  # type: ignore[import]  # noqa: F401
            return True
        except ImportError:
            return False

    # ── Model loading ─────────────────────────────────────────────────────────

    def load(self) -> Any:
        if self._model is not None:
            return self._model

        path = self.resolve_path()
        if path is None:
            raise RuntimeError(
                f"TripoSR weights not found. Download with:\n"
                f"  huggingface-cli download {_HF_REPO} --local-dir "
                f"{ai_models_root() / _DEFAULT_LOCAL}"
            )

        if not self._ensure_source_on_path():
            raise RuntimeError(
                f"TripoSR source not found at {self._source_root()}. "
                "Clone from https://github.com/VAST-AI-Research/TripoSR"
            )

        self._patch_marching_cubes_for_cuda13()

        try:
            from tsr.system import TSR  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "TripoSR source dependencies are not installed. "
                "Run: pip install -r /mnt/c/ai_tools/TripoSR/requirements.txt"
            ) from exc

        _log.info("Loading TripoSR from %s on %s", path, self._device)
        self._model = TSR.from_pretrained(
            str(path),
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self._model.to(self._device)
        self._model.eval()
        self._model.renderer.set_chunk_size(8192)
        return self._model

    def get_model_paths(self) -> tuple[list[str], list[str]]:
        """Return (found_paths, missing_paths) for availability reporting."""
        path = self.resolve_path()
        if path:
            return [str(path)], []
        return [], [str(ai_models_root() / _DEFAULT_LOCAL)]

    def _source_root(self) -> Path:
        return ai_tools_root() / _DEFAULT_SOURCE

    def _ensure_source_on_path(self) -> bool:
        src = self._source_root()
        if not src.exists():
            return False
        src_str = str(src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
        return True

    def _patch_marching_cubes_for_cuda13(self) -> None:
        try:
            from tsr.models.isosurface import MarchingCubeHelper  # type: ignore[import]
        except ImportError:
            return

        if getattr(MarchingCubeHelper, "_ai3d_cpu_mcubes_patch", False):
            return

        def forward_cpu(self, level):
            level = -level.view(self.resolution, self.resolution, self.resolution)
            v_pos, t_pos_idx = self.mc_func(level.detach().cpu(), 0.0)
            v_pos = v_pos[..., [2, 1, 0]]
            v_pos = v_pos / (self.resolution - 1.0)
            return v_pos.to(level.device), t_pos_idx.to(level.device)

        MarchingCubeHelper.forward = forward_cpu
        MarchingCubeHelper._ai3d_cpu_mcubes_patch = True
