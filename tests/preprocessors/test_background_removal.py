"""Tests for BackgroundRemovalPreprocessor."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ai3d.preprocessors.background_removal import BackgroundRemovalPreprocessor
from ai3d.core.models import PreprocessRequest


def test_missing_input(tmp_path):
    pre = BackgroundRemovalPreprocessor()
    result = pre.process(PreprocessRequest(
        input_path=str(tmp_path / "nonexistent.png"),
        output_dir=str(tmp_path),
    ))
    assert not result.success
    assert "not found" in result.error.lower()


def test_process_success(tmp_path, tmp_image):
    """Test with a mocked rembg so we don't need the actual model."""
    from io import BytesIO
    from PIL import Image

    fake_output = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    buf = BytesIO()
    fake_output.save(buf, format="PNG")
    fake_bytes = buf.getvalue()

    mock_session = MagicMock()
    mock_rembg = MagicMock()
    mock_rembg.remove.return_value = fake_bytes
    mock_rembg.new_session.return_value = mock_session

    with patch.dict("sys.modules", {"rembg": mock_rembg}):
        pre = BackgroundRemovalPreprocessor()
        pre._session = mock_session
        result = pre.process(PreprocessRequest(
            input_path=str(tmp_image),
            output_dir=str(tmp_path / "out"),
        ))

    assert result.success
    assert result.output_path is not None
    assert Path(result.output_path).exists()
