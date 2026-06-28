"""Alpha mask utilities for extracting and applying subject masks."""
from __future__ import annotations

from pathlib import Path

from ai3d.core.logging import get_logger
from ai3d.core.storage import ensure_directory

_log = get_logger(__name__)


def extract_alpha_mask(image_path: Path, output_dir: Path) -> Path:
    """Extract the alpha channel from an RGBA image as a grayscale mask PNG."""
    from PIL import Image  # type: ignore[import]

    ensure_directory(output_dir)
    img = Image.open(image_path).convert("RGBA")
    r, g, b, a = img.split()
    mask_path = output_dir / f"{image_path.stem}_mask.png"
    a.save(str(mask_path))
    _log.debug("Alpha mask saved: %s", mask_path)
    return mask_path


def apply_white_background(image_path: Path, output_path: Path) -> Path:
    """Composite an RGBA image onto a white background and save as RGB PNG."""
    from PIL import Image  # type: ignore[import]

    img = Image.open(image_path).convert("RGBA")
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(background, img).convert("RGB")
    ensure_directory(output_path.parent)
    composite.save(str(output_path), format="PNG")
    _log.debug("White background applied: %s", output_path)
    return output_path


def pad_to_square(image_path: Path, output_path: Path, fill: tuple[int, int, int, int] = (0, 0, 0, 0)) -> Path:
    """Pad image to square by adding transparent or solid-color padding."""
    from PIL import Image  # type: ignore[import]

    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    side = max(w, h)
    padded = Image.new("RGBA", (side, side), fill)
    offset_x = (side - w) // 2
    offset_y = (side - h) // 2
    padded.paste(img, (offset_x, offset_y), img)
    ensure_directory(output_path.parent)
    padded.save(str(output_path), format="PNG")
    _log.debug("Padded to square: %s", output_path)
    return output_path
