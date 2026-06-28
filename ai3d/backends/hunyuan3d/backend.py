"""Hunyuan3D-2 backend — full M2 implementation.

Two-stage pipeline: shape generation (DiT) → texture painting (Paint).
Weights: configs/models.yaml → local_path or /mnt/c/ai_models/vision/hunyuan3d-2
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from ai3d.core.logging import get_logger
from ai3d.core.models import (
    ArtifactRef,
    AvailabilityResult,
    BackendMetadata,
    BackendStatus,
    BaseBackend,
    GenerationRequest,
    GenerationResult,
    ProviderCapability,
    ResourceEstimate,
    StandardOutput,
)
from ai3d.core.storage import ensure_directory

_log = get_logger(__name__)

_SUPPORTED = [
    StandardOutput.DRAFT_MESH,
    StandardOutput.TEXTURED_MESH,
    StandardOutput.GLB,
    StandardOutput.MULTIVIEW_IMAGES,
    StandardOutput.NORMAL_MAPS,
]


class Hunyuan3DBackend(BaseBackend):
    """tencent/Hunyuan3D-2 — multi-view diffusion + mesh reconstruction + texture baking."""

    name = "hunyuan3d"

    def __init__(
        self,
        model_path: Optional[Path] = None,
        paint_path: Optional[Path] = None,
        device: str = "cuda",
        enable_texture: bool = True,
    ) -> None:
        from ai3d.backends.hunyuan3d.loader import Hunyuan3DLoader
        self._loader = Hunyuan3DLoader(
            model_path=model_path,
            paint_path=paint_path,
            device=device,
            enable_texture=enable_texture,
        )
        self._device = device
        self._enable_texture = enable_texture

    def check_availability(self) -> AvailabilityResult:
        found, missing = self._loader.get_model_paths()
        available = self._loader.is_available()
        reason = None if available else (
            "hy3dgen package not installed — install from https://github.com/tencent/Hunyuan3D-2"
            if found else
            "Hunyuan3D-2 weights not found. Download from huggingface: tencent/Hunyuan3D-2"
        )
        return AvailabilityResult(
            available=available,
            backend=self.name,
            reason=reason or "Ready",
            model_paths_found=found,
            model_paths_missing=missing,
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        output_dir = ensure_directory(Path(request.output_dir))
        output_types = request.output_types or [StandardOutput.GLB]

        try:
            shape_pipeline = self._loader.load_shape()
        except Exception as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="image-to-3d",
                error=f"Failed to load Hunyuan3D shape model: {exc}",
            )

        try:
            from PIL import Image

            image = Image.open(str(request.input_image_path)).convert("RGBA")
            _log.info("Running Hunyuan3D shape generation on %s", request.input_image_path)

            # Stage 1: shape generation
            shape_outputs = shape_pipeline(image, seed=request.seed or 0)
            mesh = shape_outputs["mesh"][0]

            artifacts: list[ArtifactRef] = []
            warnings: list[str] = []
            run_id = request.request_id or uuid.uuid4().hex

            # Stage 2: texture painting (optional)
            if self._enable_texture and any(t in output_types for t in [
                StandardOutput.TEXTURED_MESH, StandardOutput.GLB
            ]):
                try:
                    paint_pipeline = self._loader.load_paint()
                    if paint_pipeline is not None:
                        _log.info("Running Hunyuan3D texture painting")
                        mesh = paint_pipeline(mesh, image)
                except Exception as exc:
                    warnings.append(f"Texture painting failed, exporting untextured mesh: {exc}")

            # Export GLB
            if any(t in output_types for t in [
                StandardOutput.GLB, StandardOutput.TEXTURED_MESH, StandardOutput.DRAFT_MESH
            ]):
                glb_path = output_dir / f"{run_id}.glb"
                try:
                    mesh.export(str(glb_path))
                    size = glb_path.stat().st_size if glb_path.exists() else 0
                    artifacts.append(ArtifactRef(
                        path=str(glb_path),
                        kind="mesh",
                        label="Hunyuan3D GLB",
                        output_type=StandardOutput.GLB,
                        size_bytes=size,
                    ))
                except Exception as exc:
                    warnings.append(f"GLB export failed: {exc}")

            # Export OBJ if requested
            if StandardOutput.OBJ in output_types:
                obj_path = output_dir / f"{run_id}.obj"
                try:
                    mesh.export(str(obj_path))
                    artifacts.append(ArtifactRef(
                        path=str(obj_path),
                        kind="mesh",
                        label="Hunyuan3D OBJ",
                        output_type=StandardOutput.OBJ,
                        size_bytes=obj_path.stat().st_size if obj_path.exists() else 0,
                    ))
                except Exception as exc:
                    warnings.append(f"OBJ export failed: {exc}")

            if not artifacts:
                return GenerationResult(
                    success=False,
                    provider=self.name,
                    task_type="image-to-3d",
                    error="All export attempts failed.",
                    warnings=warnings,
                )

            return GenerationResult(
                success=True,
                provider=self.name,
                task_type="image-to-3d",
                artifacts=artifacts,
                warnings=warnings,
                metadata={"backend": "hunyuan3d", "device": self._device},
            )

        except Exception as exc:
            _log.exception("Hunyuan3D generate() failed")
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="image-to-3d",
                error=str(exc),
            )

    def estimate_requirements(self) -> ResourceEstimate:
        return ResourceEstimate(
            vram_gb=24.0,
            ram_gb=32.0,
            estimated_seconds=180.0,
            notes=["Full 3-stage pipeline. Paint stage requires additional ~8 GB VRAM."],
        )

    def export_metadata(self) -> BackendMetadata:
        return BackendMetadata(
            name="Hunyuan3D-2",
            version="2.0",
            source_repo="tencent/Hunyuan3D-2",
            modality="image-to-3d",
            supported_output_types=_SUPPORTED,
            required_vram_gb=24.0,
            notes=[
                "Two-stage: Hunyuan3D-DiT shape generation → Hunyuan3D-Paint texture baking.",
                "Hunyuan3D 2.1 uses Paint V2 for higher quality textures.",
                "enable_texture=False skips paint stage for faster mesh-only output.",
            ],
        )

    def get_capabilities(self) -> ProviderCapability:
        avail = self.check_availability()
        return ProviderCapability(
            provider=self.name,
            label="Hunyuan3D-2",
            available=avail.available,
            status=BackendStatus.READY if avail.available else BackendStatus.SCAFFOLD,
            supports_textured_mesh=True,
            supports_glb=True,
            supports_multiview=True,
            supports_normal_maps=True,
            supported_output_types=_SUPPORTED,
            notes=avail.reason and [avail.reason] or [],
        )
