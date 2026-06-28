"""SF3D (Stable Fast 3D) model weight resolver and lazy loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ai3d.core.logging import get_logger
from ai3d.core.paths import ai_models_root

_log = get_logger(__name__)

_DEFAULT_LOCAL = "vision/sf3d"
_HF_REPO = "stabilityai/stable-fast-3d"


class SF3DLoader:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda",
    ) -> None:
        self._device = device
        self._explicit_path = model_path
        self._model: Any = None

    def resolve_path(self) -> Optional[Path]:
        candidates: list[Path] = []

        if self._explicit_path:
            candidates.append(self._explicit_path)

        try:
            from ai3d.registry.model_registry import ModelRegistry
            entry = ModelRegistry().get("sf3d")
            candidates.append(Path(entry.local_path))
        except Exception:
            pass

        candidates.append(ai_models_root() / _DEFAULT_LOCAL)

        for p in candidates:
            if p.exists():
                _log.debug("SF3D weights found at: %s", p)
                return p

        return None

    def is_available(self) -> bool:
        return self.availability_reason() is None

    def availability_reason(self) -> Optional[str]:
        """Return None when load prerequisites are present, else a precise reason."""
        if self.resolve_path() is None:
            return "Model weights not found on disk."
        try:
            from sf3d.system import SF3D  # type: ignore[import]  # noqa: F401
        except ImportError as exc:
            return f"Stable Fast 3D Python package is not importable: {exc}."
        return None

    def load(self) -> Any:
        if self._model is not None:
            return self._model

        path = self.resolve_path()
        if path is None:
            raise RuntimeError(
                f"SF3D weights not found. Download with:\n"
                f"  huggingface-cli download {_HF_REPO} --local-dir "
                f"{ai_models_root() / _DEFAULT_LOCAL}"
            )

        try:
            from sf3d.system import SF3D  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "SF3D package not installed. "
                "Run: pip install -e /mnt/c/ai_tools/stable-fast-3d"
            ) from exc

        _log.info("Loading SF3D from %s on %s", path, self._device)
        self._model = SF3D.from_pretrained(
            str(path),
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        self._model.to(self._device)
        self._model.eval()
        return self._model

    def get_model_paths(self) -> tuple[list[str], list[str]]:
        path = self.resolve_path()
        if path:
            return [str(path)], []
        return [], [str(ai_models_root() / _DEFAULT_LOCAL)]
