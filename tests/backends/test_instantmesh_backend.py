"""Unit tests for InstantMeshBackend (mocked — no model weights required)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai3d.core.models import AvailabilityResult, BackendStatus, GenerationRequest, StandardOutput


@pytest.fixture()
def backend():
    from ai3d.backends.instantmesh.backend import InstantMeshBackend
    return InstantMeshBackend(device="cpu")


def test_name(backend):
    assert backend.name == "instantmesh"


def test_estimate_requirements(backend):
    est = backend.estimate_requirements()
    assert est.vram_gb >= 20.0


def test_export_metadata(backend):
    meta = backend.export_metadata()
    assert "InstantMesh" in meta.name
    assert StandardOutput.MULTIVIEW_IMAGES in meta.supported_output_types


def test_check_unavailable(backend):
    with patch.object(backend._loader, "is_available", return_value=False), \
         patch.object(backend._loader, "get_model_paths", return_value=([], ["/missing"])):
        result = backend.check_availability()
    assert result.available is False
    assert result.backend == "instantmesh"


def test_generate_load_failure(backend, tmp_path):
    with patch.object(backend._loader, "load_zero123plus", side_effect=RuntimeError("no diffusers")):
        request = GenerationRequest(
            request_id="test-001",
            input_image_path=str(tmp_path / "img.jpg"),
            backend="instantmesh",
            output_dir=str(tmp_path / "out"),
        )
        result = backend.generate(request)
    assert result.success is False
    assert "no diffusers" in result.error


def test_get_capabilities_scaffold(backend):
    with patch.object(backend._loader, "is_available", return_value=False), \
         patch.object(backend._loader, "get_model_paths", return_value=([], [])):
        cap = backend.get_capabilities()
    assert cap.status == BackendStatus.SCAFFOLD
    assert cap.supports_multiview is True
