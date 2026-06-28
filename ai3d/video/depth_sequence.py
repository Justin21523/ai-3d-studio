"""Depth sequence processor — normalize and stack depth maps into PNG sequences."""
from __future__ import annotations

from pathlib import Path
from typing import List

from ai3d.core.logging import get_logger
from ai3d.core.storage import ensure_directory

_log = get_logger(__name__)


class DepthSequenceExporter:
    """Normalizes depth frame sequences (EXR or PNG) to 8-bit PNG output."""

    def export_depth_stack(
        self,
        depth_dir: Path,
        output_dir: Path,
        normalize: bool = True,
        glob_pattern: str = "*.exr",
    ) -> List[Path]:
        """
        Normalize and convert depth frames to PNG.

        Tries EXR first; falls back to PNG if no EXR files are found.
        Returns list of output frame paths sorted by name.
        """
        try:
            import numpy as np  # type: ignore[import]
            from PIL import Image  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(f"Missing dependency: {exc}") from exc

        ensure_directory(output_dir)

        frames = sorted(depth_dir.glob(glob_pattern))
        if not frames:
            frames = sorted(depth_dir.glob("*.png"))

        if not frames:
            _log.warning("No depth frames found in: %s", depth_dir)
            return []

        output_paths: List[Path] = []

        # Find global min/max for consistent normalization across sequence
        if normalize:
            depth_arrays = []
            for frame_path in frames:
                arr = _load_depth_frame(frame_path)
                if arr is not None:
                    depth_arrays.append(arr)

            if depth_arrays:
                global_min = float(min(a.min() for a in depth_arrays))
                global_max = float(max(a.max() for a in depth_arrays))
            else:
                global_min, global_max = 0.0, 1.0
        else:
            global_min, global_max = 0.0, 1.0

        depth_range = max(global_max - global_min, 1e-6)

        for i, frame_path in enumerate(frames):
            arr = _load_depth_frame(frame_path)
            if arr is None:
                _log.warning("Could not load depth frame: %s", frame_path)
                continue

            if normalize:
                arr = (arr - global_min) / depth_range
                arr = (arr * 255).clip(0, 255).astype("uint8")
            else:
                arr = arr.clip(0, 255).astype("uint8")

            out_path = output_dir / f"depth_{i:04d}.png"
            Image.fromarray(arr).save(str(out_path))
            output_paths.append(out_path)

        _log.info("Depth sequence: %d frames written to %s", len(output_paths), output_dir)
        return output_paths


def _load_depth_frame(frame_path: Path):
    """Load a depth frame as a 2D float32 numpy array (grayscale)."""
    try:
        import numpy as np  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        ext = frame_path.suffix.lower()
        if ext == ".exr":
            try:
                import OpenEXR  # type: ignore[import]
                import Imath

                exr = OpenEXR.InputFile(str(frame_path))
                dw = exr.header()["dataWindow"]
                size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
                depth_str = exr.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT))
                arr = np.frombuffer(depth_str, dtype=np.float32).reshape(size[1], size[0])
                return arr
            except ImportError:
                # Fall back to PIL for EXR (may not support depth channels)
                img = Image.open(frame_path).convert("F")
                return np.array(img, dtype=np.float32)
        else:
            img = Image.open(frame_path).convert("L")
            return np.array(img, dtype=np.float32)
    except Exception as exc:
        return None
