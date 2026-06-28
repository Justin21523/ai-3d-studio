"""Background removal preprocessor using rembg."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ai3d.core.logging import get_logger
from ai3d.core.models import BasePreprocessor, PreprocessRequest, PreprocessResult
from ai3d.core.storage import ensure_directory

_log = get_logger(__name__)


class BackgroundRemovalPreprocessor(BasePreprocessor):
    """Removes background from an image using rembg (u2net by default)."""

    name = "background_removal"

    def __init__(self, model_name: str = "u2net") -> None:
        self._model_name = model_name
        self._session: Any = None

    def process(self, request: PreprocessRequest) -> PreprocessResult:
        input_path = Path(request.input_path)
        if not input_path.exists():
            return PreprocessResult(
                success=False,
                error=f"Input file not found: {input_path}",
            )

        try:
            from PIL import Image  # type: ignore[import]
            import rembg  # type: ignore[import]
        except ImportError as exc:
            return PreprocessResult(
                success=False,
                error=f"Missing dependency: {exc}. Install: pip install rembg Pillow",
            )

        output_dir = ensure_directory(Path(request.output_dir))
        out_path = output_dir / f"{input_path.stem}_nobg.png"

        try:
            with open(input_path, "rb") as f:
                raw = f.read()

            session = self._get_session(rembg)
            result_bytes = rembg.remove(raw, session=session)

            from io import BytesIO
            result_image = Image.open(BytesIO(result_bytes)).convert("RGBA")

            if request.target_size:
                result_image = result_image.resize(request.target_size)

            result_image.save(str(out_path), format="PNG")
            _log.info("Background removed: %s -> %s", input_path.name, out_path.name)

            return PreprocessResult(
                success=True,
                output_path=str(out_path),
            )
        except Exception as exc:
            _log.error("Background removal failed for %s: %s", input_path, exc)
            return PreprocessResult(
                success=False,
                error=str(exc),
            )

    def _get_session(self, rembg_module: Any) -> Any:
        if self._session is None:
            _log.debug("Initializing rembg session: %s", self._model_name)
            self._session = rembg_module.new_session(self._model_name)
        return self._session
