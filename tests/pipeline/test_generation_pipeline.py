"""Tests for GenerationPipeline stages."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ai3d.pipeline.generation_pipeline import GenerationPipeline
from ai3d.core.models import (
    ArtifactRef,
    GenerationRequest,
    GenerationResult,
    PipelineStage,
    StandardOutput,
)


@pytest.fixture()
def mock_registry(tmp_path: Path):
    """BackendRegistry mock that returns a successful GLB artifact."""
    glb_path = tmp_path / "generated" / "output.glb"
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    glb_path.write_bytes(b"fake glb")

    mock_backend = MagicMock()
    mock_backend.generate.return_value = GenerationResult(
        success=True,
        provider="mock",
        task_type="image-to-3d",
        artifacts=[ArtifactRef(path=str(glb_path), output_type=StandardOutput.GLB)],
    )
    mock_reg = MagicMock()
    mock_reg.get.return_value = mock_backend
    return mock_reg


def test_pipeline_ingest_missing_image(tmp_path):
    pipeline = GenerationPipeline(
        skip_blender=True, skip_video_pack=True, skip_registration=True
    )
    request = GenerationRequest(
        request_id="p1",
        input_image_path=str(tmp_path / "ghost.png"),
        backend="mock",
        output_dir=str(tmp_path / "out"),
    )
    manifest = pipeline.run(request)
    assert PipelineStage.INPUT_INGEST in manifest.stages_failed


def test_pipeline_stages_recorded(tmp_path, tmp_image, mock_registry):
    pipeline = GenerationPipeline(
        backend_registry=mock_registry,
        skip_blender=True,
        skip_video_pack=True,
        skip_registration=True,
    )
    request = GenerationRequest(
        request_id="p2",
        input_image_path=str(tmp_image),
        backend="mock",
        output_dir=str(tmp_path / "out"),
        remove_background=False,
        run_quality_check=False,
    )
    manifest = pipeline.run(request)

    assert PipelineStage.INPUT_INGEST in manifest.stages_completed
    assert PipelineStage.GENERATION_3D in manifest.stages_completed
    assert manifest.run_id == "p2"


def test_manifest_saved_to_disk(tmp_path, tmp_image, mock_registry):
    pipeline = GenerationPipeline(
        backend_registry=mock_registry,
        skip_blender=True,
        skip_video_pack=True,
        skip_registration=True,
    )
    request = GenerationRequest(
        request_id="p3",
        input_image_path=str(tmp_image),
        backend="mock",
        output_dir=str(tmp_path / "out"),
        remove_background=False,
        run_quality_check=False,
    )
    pipeline.run(request)
    assert (tmp_path / "out" / "pipeline_manifest.yaml").exists()


def test_pipeline_runs_mesh_to_splat_when_requested(tmp_path, tmp_image, mock_registry):
    splat_path = tmp_path / "out" / "splats" / "p4.ply"
    splat_result = GenerationResult(
        success=True,
        provider="mesh2splat",
        task_type="mesh-to-splat",
        artifacts=[ArtifactRef(path=str(splat_path), output_type=StandardOutput.SPLAT_PLY)],
    )
    pipeline = GenerationPipeline(
        backend_registry=mock_registry,
        skip_blender=True,
        skip_video_pack=True,
        skip_registration=True,
    )
    request = GenerationRequest(
        request_id="p4",
        input_image_path=str(tmp_image),
        backend="mock",
        output_types=[StandardOutput.GLB, StandardOutput.SPLAT_PLY],
        output_dir=str(tmp_path / "out"),
        remove_background=False,
        run_quality_check=False,
    )

    with patch("ai3d.backends.mesh2splat.backend.Mesh2SplatBackend.generate", return_value=splat_result) as generate:
        manifest = pipeline.run(request)

    assert PipelineStage.MESH_TO_SPLAT in manifest.stages_completed
    assert any(a.output_type == StandardOutput.SPLAT_PLY for a in manifest.artifacts)
    generate.assert_called_once()
    backend_request = mock_registry.get.return_value.generate.call_args.args[0]
    assert backend_request.output_types == [StandardOutput.GLB]
