"""Unit tests for TRELLISBackend (mocked — no model weights required)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai3d.core.models import AvailabilityResult, BackendStatus, GenerationRequest, StandardOutput


@pytest.fixture()
def backend():
    from ai3d.backends.trellis.backend import TRELLISBackend
    return TRELLISBackend(device="cpu")


def test_name(backend):
    assert backend.name == "trellis"


def test_estimate_requirements(backend):
    est = backend.estimate_requirements()
    assert est.vram_gb >= 16.0
    assert est.estimated_seconds > 0


def test_export_metadata(backend):
    meta = backend.export_metadata()
    assert meta.source_repo == "microsoft/TRELLIS"
    assert StandardOutput.SPLAT_PLY in meta.supported_output_types
    assert StandardOutput.GLB in meta.supported_output_types


def test_check_availability_no_weights(backend, tmp_path):
    with patch.object(backend._loader, "is_available", return_value=False), \
         patch.object(backend._loader, "get_model_paths", return_value=([], ["/fake/path"])):
        result = backend.check_availability()
    assert isinstance(result, AvailabilityResult)
    assert result.available is False
    assert result.backend == "trellis"


def test_check_availability_ready(backend):
    with patch.object(backend._loader, "is_available", return_value=True), \
         patch.object(backend._loader, "get_model_paths", return_value=(["/found"], [])):
        result = backend.check_availability()
    assert result.available is True


def test_generate_load_failure(backend, tmp_path):
    with patch.object(backend._loader, "load", side_effect=RuntimeError("weights missing")):
        request = GenerationRequest(
            request_id="test-001",
            input_image_path=str(tmp_path / "img.jpg"),
            backend="trellis",
            output_dir=str(tmp_path / "out"),
        )
        result = backend.generate(request)
    assert result.success is False
    assert "weights missing" in result.error


def test_generate_success_glb(backend, tmp_path):
    input_img = tmp_path / "img.png"
    input_img.write_bytes(b"fake")
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    fake_mesh = MagicMock()
    fake_mesh.export.side_effect = lambda p: Path(p).write_bytes(b"glb")

    fake_pipeline = MagicMock()
    fake_pipeline.return_value = {"mesh": [fake_mesh], "gaussian": [MagicMock()]}

    with patch.object(backend._loader, "load", return_value=fake_pipeline), \
         patch("PIL.Image.open") as mock_open:
        mock_open.return_value.convert.return_value = MagicMock()
        request = GenerationRequest(
            request_id="test-002",
            input_image_path=str(input_img),
            backend="trellis",
            output_dir=str(output_dir),
            output_types=[StandardOutput.GLB],
        )
        result = backend.generate(request)

    assert result.success is True
    assert result.provider == "trellis"


def test_get_capabilities_scaffold(backend):
    with patch.object(backend._loader, "is_available", return_value=False), \
         patch.object(backend._loader, "get_model_paths", return_value=([], [])):
        cap = backend.get_capabilities()
    assert cap.status == BackendStatus.SCAFFOLD
    assert cap.supports_splat is True
