"""Unit tests for Mesh2SplatBackend (mocked — no model weights required)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from ai3d.core.models import GenerationRequest, StandardOutput


@pytest.fixture()
def backend():
    from ai3d.backends.mesh2splat.backend import Mesh2SplatBackend
    return Mesh2SplatBackend()


def test_name(backend):
    assert backend.name == "mesh2splat"


def test_export_metadata(backend):
    meta = backend.export_metadata()
    assert StandardOutput.SPLAT_PLY in meta.supported_output_types


def test_check_availability(backend):
    with patch.object(backend._loader, "is_available", return_value=True), \
         patch.object(backend._loader, "get_model_paths", return_value=([], [])):
        available = backend.check_availability()
    assert available.available is True

    with patch.object(backend._loader, "is_available", return_value=False), \
         patch.object(backend._loader, "get_model_paths", return_value=([], [])):
        unavailable = backend.check_availability()
    assert unavailable.available is False


def test_generate_load_failure(backend, tmp_path):
    with patch.object(backend._loader, "load", side_effect=RuntimeError("no trimesh")):
        request = GenerationRequest(
            request_id="test-001",
            input_image_path=str(tmp_path / "img.jpg"),
            backend="mesh2splat",
            output_dir=str(tmp_path / "out"),
        )
        result = backend.generate(request)
    assert result.success is False
    assert "no trimesh" in result.error
