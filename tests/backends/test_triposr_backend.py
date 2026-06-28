"""Tests for TripoSRBackend — model-mocked."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ai3d.backends.triposr.backend import TripoSRBackend
from ai3d.core.models import GenerationRequest, StandardOutput


@pytest.fixture()
def backend():
    return TripoSRBackend(device="cpu")


def test_check_availability_no_weights(backend, tmp_path):
    # Weights don't exist in tmp_path, so should report unavailable
    with patch.object(backend._loader, "is_available", return_value=False):
        result = backend.check_availability()
    assert result.available is False
    assert result.backend == "triposr"


def test_check_availability_no_cuda(backend):
    with patch.object(backend._loader, "is_available", return_value=True):
        result = backend.check_availability()
    # device is "cpu" so CUDA check is skipped
    assert result.available is True


def test_generate_missing_input(backend, tmp_path):
    with patch.object(backend._loader, "load", return_value=MagicMock()):
        backend._model = MagicMock()
        request = GenerationRequest(
            request_id="t1",
            input_image_path=str(tmp_path / "nonexistent.png"),
            backend="triposr",
            output_dir=str(tmp_path / "out"),
        )
        result = backend.generate(request)
    assert not result.success
    assert "not found" in result.error.lower()


def test_estimate_requirements(backend):
    est = backend.estimate_requirements()
    assert est.vram_gb == 8.0
    assert est.estimated_seconds > 0


def test_export_metadata(backend):
    meta = backend.export_metadata()
    assert meta.name == "TripoSR"
    assert StandardOutput.GLB in meta.supported_output_types


def test_get_capabilities_unavailable(backend):
    with patch.object(backend._loader, "is_available", return_value=False):
        cap = backend.get_capabilities()
    assert not cap.available
