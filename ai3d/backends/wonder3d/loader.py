"""Wonder3D source resolver and lazy loader."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from ai3d.core.logging import get_logger
from ai3d.core.paths import ai_models_root, ai_tools_root

_log = get_logger(__name__)

_DEFAULT_LOCAL = "vision/wonder3d"
_DEFAULT_SOURCE = "Wonder3D"
_HF_REPO = "xxlong0/Wonder3D"


class Wonder3DLoader:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda",
    ) -> None:
        self._device = device
        self._explicit_path = model_path
        self._pipeline: Any = None

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

    def resolve_weight_path(self) -> Optional[Path]:
        candidates: list[Path] = []

        if self._explicit_path:
            candidates.append(self._explicit_path)

        try:
            from ai3d.registry.model_registry import ModelRegistry
            entry = ModelRegistry().get("wonder3d")
            candidates.append(Path(entry.local_path))
        except Exception:
            pass

        candidates.append(ai_models_root() / _DEFAULT_LOCAL)

        for path in candidates:
            if path.exists():
                _log.debug("Wonder3D weights found at: %s", path)
                return path

        return None

    def is_available(self) -> bool:
        if self.resolve_weight_path() is None:
            return False
        if not self._ensure_source_on_path():
            return False
        try:
            from wonder3d.ldm.models.diffusion import ddpm  # type: ignore[import]  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        weight_path = self.resolve_weight_path()
        if weight_path is None:
            raise RuntimeError(
                f"Wonder3D weights not found. Download from:\n"
                f"  huggingface-cli download {_HF_REPO} --local-dir "
                f"{ai_models_root() / _DEFAULT_LOCAL}"
            )

        if not self._ensure_source_on_path():
            raise RuntimeError(
                f"Wonder3D source not found at {self._source_root()}. "
                "Clone from https://github.com/xxlong0/Wonder3D"
            )

        try:
            import torch
            from wonder3d.ldm.models.diffusion.ddpm import LatentDiffusion  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Wonder3D source not on path — check that Wonder3D is cloned and its deps installed"
            ) from exc

        ckpt_candidates = [
            weight_path / "model.ckpt",
            weight_path / "last.ckpt",
            weight_path / "wonder3d.ckpt",
        ]
        ckpt = next((path for path in ckpt_candidates if path.exists()), None)
        if ckpt is None:
            matches = sorted(weight_path.glob("*.ckpt")) + sorted(weight_path.glob("*.pth"))
            ckpt = matches[0] if matches else None
        if ckpt is None:
            raise RuntimeError(f"Wonder3D checkpoint not found under {weight_path}")

        _log.info("Loading Wonder3D from %s on %s", ckpt, self._device)
        model = LatentDiffusion()
        state = torch.load(str(ckpt), map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict, strict=False)
        model.to(self._device)
        model.eval()
        self._pipeline = model
        return self._pipeline

    def get_model_paths(self) -> tuple[list[str], list[str]]:
        path = self.resolve_weight_path()
        if path:
            return [str(path)], []
        return [], [str(ai_models_root() / _DEFAULT_LOCAL)]
