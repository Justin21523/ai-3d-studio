"""CRM (Convolutional Reconstruction Model) weight resolver and lazy loader.

CRM has no pip package — expects source installation at /mnt/c/ai_tools/CRM.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from ai3d.core.logging import get_logger
from ai3d.core.paths import ai_models_root, ai_tools_root

_log = get_logger(__name__)

_DEFAULT_LOCAL = "vision/crm"
_DEFAULT_SOURCE = "CRM"
_HF_REPO = "Zhengyi-Wang/CRM"


class CRMLoader:
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

    def resolve_weight_path(self) -> Optional[Path]:
        candidates: list[Path] = []

        if self._explicit_path:
            candidates.append(self._explicit_path)

        try:
            from ai3d.registry.model_registry import ModelRegistry
            entry = ModelRegistry().get("crm")
            candidates.append(Path(entry.local_path))
        except Exception:
            pass

        candidates.append(ai_models_root() / _DEFAULT_LOCAL)

        for p in candidates:
            if p.exists():
                _log.debug("CRM weights found at: %s", p)
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

    def load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        weight_path = self.resolve_weight_path()
        if weight_path is None:
            raise RuntimeError(
                f"CRM weights not found. Download from:\n"
                f"  huggingface-cli download {_HF_REPO} --local-dir "
                f"{ai_models_root() / _DEFAULT_LOCAL}"
            )

        if not self._ensure_source_on_path():
            raise RuntimeError(
                f"CRM source not found at {self._source_root()}. "
                "Clone from https://github.com/thu-ml/CRM"
            )

        try:
            import torch
            from model import CRM  # type: ignore[import]
            from imagedream.ldm.util import set_seed  # type: ignore[import]  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "CRM source not on path — check that CRM is cloned and its deps installed"
            ) from exc

        _log.info("Loading CRM from %s on %s", weight_path, self._device)

        crm_ckpt = weight_path / "CRM.pth"
        if not crm_ckpt.exists():
            raise RuntimeError(f"CRM checkpoint not found: {crm_ckpt}")

        self._pipeline = CRM.from_pretrained(str(weight_path)).to(self._device)
        self._pipeline.eval()
        return self._pipeline

    def get_model_paths(self) -> tuple[list[str], list[str]]:
        path = self.resolve_weight_path()
        if path:
            return [str(path)], []
        return [], [str(ai_models_root() / _DEFAULT_LOCAL)]
