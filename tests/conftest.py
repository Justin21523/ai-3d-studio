"""Shared pytest fixtures for ai-3d-studio tests."""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture()
def tmp_image(tmp_path: Path) -> Path:
    """Create a 512x512 RGBA test image."""
    img = Image.new("RGBA", (512, 512), (128, 200, 100, 255))
    # Draw a simple centered square so coverage check passes
    for x in range(100, 412):
        for y in range(100, 412):
            img.putpixel((x, y), (200, 100, 50, 255))
    p = tmp_path / "test_input.png"
    img.save(str(p))
    return p


@pytest.fixture()
def tmp_rgb_image(tmp_path: Path) -> Path:
    """Create a 512x512 RGB test image."""
    img = Image.new("RGB", (512, 512), (100, 150, 200))
    p = tmp_path / "test_input_rgb.png"
    img.save(str(p))
    return p
