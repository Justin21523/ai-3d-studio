"""Tests for core Pydantic models and enumerations."""
from __future__ import annotations

import pytest
from ai3d.core.models import (
    ArtifactRef,
    AvailabilityResult,
    BackendMetadata,
    GenerationRequest,
    GenerationResult,
    PathConfig,
    PipelineManifest,
    ProviderCapability,
    ResourceEstimate,
    StandardOutput,
    TurntableSpec,
    VideoConditioningPack,
)


def test_standard_output_enum_values():
    assert StandardOutput.GLB.value == "glb"
    assert StandardOutput.SPLAT_PLY.value == "splat_ply"


def test_generation_request_defaults():
    req = GenerationRequest(
        request_id="abc",
        input_image_path="/tmp/img.png",
        backend="sf3d",
        output_dir="/tmp/out",
    )
    assert req.output_types == [StandardOutput.GLB]
    assert req.remove_background is True
    assert req.device == "cuda"


def test_generation_result_roundtrip():
    result = GenerationResult(
        success=True,
        provider="sf3d",
        task_type="image-to-3d",
        artifacts=[ArtifactRef(path="/tmp/out.glb", output_type=StandardOutput.GLB)],
    )
    dumped = result.model_dump(mode="json")
    restored = GenerationResult.model_validate(dumped)
    assert restored.success is True
    assert restored.artifacts[0].output_type == StandardOutput.GLB


def test_path_config_defaults():
    cfg = PathConfig()
    assert cfg.ai_models_root == "/mnt/c/ai_models"
    assert cfg.comfyui_base_url == "http://127.0.0.1:8188"


def test_turntable_spec_defaults():
    spec = TurntableSpec(asset_path="/tmp/mesh.glb", output_dir="/tmp/render")
    assert spec.frame_count == 72
    assert spec.fps == 24
    assert spec.resolution_x == 1024


def test_availability_result_roundtrip():
    avail = AvailabilityResult(available=False, backend="trellis", reason="scaffold")
    data = avail.model_dump(mode="json")
    restored = AvailabilityResult.model_validate(data)
    assert restored.backend == "trellis"
    assert not restored.available


def test_pipeline_manifest_stages():
    from ai3d.core.models import PipelineStage
    req = GenerationRequest(
        request_id="x1", input_image_path="/tmp/x.png",
        backend="triposr", output_dir="/tmp/x",
    )
    manifest = PipelineManifest(run_id="x1", request=req)
    manifest.stages_completed.append(PipelineStage.INPUT_INGEST)
    assert PipelineStage.INPUT_INGEST in manifest.stages_completed
