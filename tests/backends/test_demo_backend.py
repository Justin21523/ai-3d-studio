"""Tests for the mock-safe demo backend."""
from __future__ import annotations

from ai3d.backends.demo.backend import DemoBackend
from ai3d.core.models import GenerationRequest, StandardOutput


def test_demo_backend_available():
    backend = DemoBackend()
    assert backend.check_availability().available
    assert backend.get_capabilities().metadata["mock_safe"] is True


def test_demo_backend_generates_artifacts(tmp_path, tmp_image):
    backend = DemoBackend()
    result = backend.generate(
        GenerationRequest(
            request_id="demo-test",
            input_image_path=str(tmp_image),
            backend="demo",
            output_types=[StandardOutput.GLB, StandardOutput.OBJ],
            output_dir=str(tmp_path / "out"),
        )
    )

    assert result.success
    assert result.metadata["mock_safe"] is True
    assert len(result.artifacts) == 3
    assert all((tmp_path / "out" / name).exists() for name in [
        "demo_asset.glb",
        "demo_asset.obj",
        "demo_manifest.yaml",
    ])
