"""Tests for SF3DBackend — model-mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from ai3d.backends.sf3d.backend import SF3DBackend
from ai3d.core.models import GenerationRequest, StandardOutput


@pytest.fixture()
def backend():
    return SF3DBackend(device="cpu")


def test_check_availability_no_weights(backend):
    with patch.object(backend._loader, "is_available", return_value=False):
        result = backend.check_availability()
    assert not result.available
    assert result.backend == "sf3d"


def test_estimate_requirements(backend):
    est = backend.estimate_requirements()
    assert est.vram_gb == 6.0


def test_export_metadata(backend):
    meta = backend.export_metadata()
    assert meta.name == "Stable Fast 3D"
    assert StandardOutput.TEXTURED_MESH in meta.supported_output_types


def test_generate_missing_input(backend, tmp_path):
    backend._model = MagicMock()
    request = GenerationRequest(
        request_id="s1",
        input_image_path=str(tmp_path / "ghost.png"),
        backend="sf3d",
        output_dir=str(tmp_path / "out"),
    )
    result = backend.generate(request)
    assert not result.success
    assert "not found" in result.error.lower()
