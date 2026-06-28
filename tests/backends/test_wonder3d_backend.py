"""Unit tests for Wonder3DBackend (mocked — no model weights required)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from ai3d.core.models import GenerationRequest, StandardOutput


@pytest.fixture()
def backend():
    from ai3d.backends.wonder3d.backend import Wonder3DBackend
    return Wonder3DBackend(device="cpu")


def test_name(backend):
    assert backend.name == "wonder3d"


def test_export_metadata(backend):
    meta = backend.export_metadata()
    assert StandardOutput.MULTIVIEW_IMAGES in meta.supported_output_types
    assert StandardOutput.NORMAL_MAPS in meta.supported_output_types


def test_check_unavailable(backend):
    with patch.object(backend._loader, "is_available", return_value=False), \
         patch.object(backend._loader, "get_model_paths", return_value=([], ["/missing"])):
        result = backend.check_availability()
    assert result.available is False
    assert result.backend == "wonder3d"


def test_generate_load_failure(backend, tmp_path):
    with patch.object(backend._loader, "load", side_effect=RuntimeError("no Wonder3D")):
        request = GenerationRequest(
            request_id="test-001",
            input_image_path=str(tmp_path / "img.jpg"),
            backend="wonder3d",
            output_dir=str(tmp_path / "out"),
        )
        result = backend.generate(request)
    assert result.success is False
    assert "no Wonder3D" in result.error
