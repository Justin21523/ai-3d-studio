"""Unit tests for Hunyuan3DBackend (mocked — no model weights required)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai3d.core.models import AvailabilityResult, BackendStatus, GenerationRequest, StandardOutput


@pytest.fixture()
def backend():
    from ai3d.backends.hunyuan3d.backend import Hunyuan3DBackend
    return Hunyuan3DBackend(device="cpu")


def test_name(backend):
    assert backend.name == "hunyuan3d"


def test_estimate_requirements(backend):
    est = backend.estimate_requirements()
    assert est.vram_gb >= 24.0


def test_export_metadata(backend):
    meta = backend.export_metadata()
    assert meta.source_repo == "tencent/Hunyuan3D-2"
    assert StandardOutput.MULTIVIEW_IMAGES in meta.supported_output_types


def test_check_availability_no_weights(backend):
    with patch.object(backend._loader, "is_available", return_value=False), \
         patch.object(backend._loader, "get_model_paths", return_value=([], ["/missing"])):
        result = backend.check_availability()
    assert result.available is False
    assert result.backend == "hunyuan3d"


def test_generate_load_failure(backend, tmp_path):
    with patch.object(backend._loader, "load_shape", side_effect=RuntimeError("no weights")):
        request = GenerationRequest(
            request_id="test-001",
            input_image_path=str(tmp_path / "img.jpg"),
            backend="hunyuan3d",
            output_dir=str(tmp_path / "out"),
        )
        result = backend.generate(request)
    assert result.success is False
    assert "no weights" in result.error


def test_generate_success_glb(backend, tmp_path):
    input_img = tmp_path / "img.png"
    input_img.write_bytes(b"fake")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    fake_mesh = MagicMock()
    fake_mesh.export.side_effect = lambda p: Path(p).write_bytes(b"glb")

    fake_shape_pipeline = MagicMock()
    fake_shape_pipeline.return_value = {"mesh": [fake_mesh]}

    with patch.object(backend._loader, "load_shape", return_value=fake_shape_pipeline), \
         patch.object(backend._loader, "load_paint", return_value=None), \
         patch("PIL.Image.open") as mock_open:
        mock_open.return_value.convert.return_value = MagicMock()
        request = GenerationRequest(
            request_id="test-002",
            input_image_path=str(input_img),
            backend="hunyuan3d",
            output_dir=str(output_dir),
            output_types=[StandardOutput.GLB],
        )
        result = backend.generate(request)

    assert result.success is True
    assert len(result.artifacts) >= 1


def test_get_capabilities_scaffold(backend):
    with patch.object(backend._loader, "is_available", return_value=False), \
         patch.object(backend._loader, "get_model_paths", return_value=([], [])):
        cap = backend.get_capabilities()
    assert cap.status == BackendStatus.SCAFFOLD
    assert cap.supports_multiview is True
