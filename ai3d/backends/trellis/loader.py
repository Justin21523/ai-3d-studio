"""TRELLIS model weight resolver and lazy loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ai3d.core.logging import get_logger
from ai3d.core.paths import ai_models_root

_log = get_logger(__name__)

_DEFAULT_LOCAL = "vision/trellis"
_HF_REPO = "microsoft/TRELLIS-image-to-3D"


class TRELLISLoader:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda",
    ) -> None:
        self._device = device
        self._explicit_path = model_path
        self._pipeline: Any = None

    def resolve_path(self) -> Optional[Path]:
        candidates: list[Path] = []

        if self._explicit_path:
            candidates.append(self._explicit_path)

        try:
            from ai3d.registry.model_registry import ModelRegistry
            entry = ModelRegistry().get("trellis")
            candidates.append(Path(entry.local_path))
        except Exception:
            pass

        candidates.append(ai_models_root() / _DEFAULT_LOCAL)

        for p in candidates:
            if p.exists():
                _log.debug("TRELLIS weights found at: %s", p)
                return p

        return None

    def is_available(self) -> bool:
        if self.resolve_path() is None:
            return False
        try:
            import trellis  # type: ignore[import]  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline

        path = self.resolve_path()
        if path is None:
            raise RuntimeError(
                f"TRELLIS weights not found. Download with:\n"
                f"  huggingface-cli download {_HF_REPO} --local-dir "
                f"{ai_models_root() / _DEFAULT_LOCAL}"
            )

        try:
            from trellis.pipelines import TrellisImageTo3DPipeline  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "TRELLIS package not installed. "
                "Follow setup at https://github.com/microsoft/TRELLIS"
            ) from exc

        _log.info("Loading TRELLIS from %s on %s", path, self._device)
        self._pipeline = TrellisImageTo3DPipeline.from_pretrained(str(path))
        self._pipeline.cuda()
        return self._pipeline

    def get_model_paths(self) -> tuple[list[str], list[str]]:
        path = self.resolve_path()
        if path:
            return [str(path)], []
        return [], [str(ai_models_root() / _DEFAULT_LOCAL)]
