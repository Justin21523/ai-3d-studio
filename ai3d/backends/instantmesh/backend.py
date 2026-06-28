"""InstantMesh backend — full M2 implementation.

Two-stage pipeline: Zero123++ multi-view generation → InstantMesh LRM reconstruction.
Source: clone TencentARC/InstantMesh to /mnt/c/ai_tools/InstantMesh.
Weights: configs/models.yaml → local_path or /mnt/c/ai_models/vision/instantmesh
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
    StandardOutput.OBJ,
    StandardOutput.MULTIVIEW_IMAGES,
]

_NUM_VIEWS = 6


class InstantMeshBackend(BaseBackend):
    """TencentARC/InstantMesh — Zero123++ multi-view + LRM sparse-view 3D reconstruction."""

    name = "instantmesh"

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda",
    ) -> None:
        from ai3d.backends.instantmesh.loader import InstantMeshLoader
        self._loader = InstantMeshLoader(model_path=model_path, device=device)
        self._device = device

    def check_availability(self) -> AvailabilityResult:
        found, missing = self._loader.get_model_paths()
        available = self._loader.is_available()
        reason = None if available else (
            "InstantMesh source or diffusers not found — clone TencentARC/InstantMesh"
            if found else
            "InstantMesh weights not found. Download from TencentARC/InstantMesh on HuggingFace."
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
            zero123_pipeline = self._loader.load_zero123plus()
            lrm_model = self._loader.load_lrm()
        except Exception as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="image-to-3d",
                error=f"Failed to load InstantMesh models: {exc}",
            )

        try:
            from PIL import Image
            import torch

            image = Image.open(str(request.input_image_path)).convert("RGBA")
            run_id = request.request_id or uuid.uuid4().hex

            _log.info("InstantMesh: generating %d views via Zero123++", _NUM_VIEWS)

            # Stage 1: multi-view generation
            with torch.no_grad():
                mv_result = zero123_pipeline(
                    image,
                    num_inference_steps=75,
                    generator=torch.Generator(device="cpu").manual_seed(request.seed or 0),
                ).images[0]

            artifacts: list[ArtifactRef] = []
            warnings: list[str] = []

            # Save multi-view strip if requested
            if StandardOutput.MULTIVIEW_IMAGES in output_types:
                mv_path = output_dir / f"{run_id}_multiview.png"
                mv_result.save(str(mv_path))
                artifacts.append(ArtifactRef(
                    path=str(mv_path),
                    kind="multiview",
                    label="Zero123++ 6-view strip",
                    output_type=StandardOutput.MULTIVIEW_IMAGES,
                    size_bytes=mv_path.stat().st_size,
                ))

            # Stage 2: LRM reconstruction
            _log.info("InstantMesh: LRM sparse-view reconstruction")
            with torch.no_grad():
                mesh = lrm_model.extract_mesh(mv_result)

            # GLB export
            if any(t in output_types for t in [
                StandardOutput.GLB, StandardOutput.TEXTURED_MESH, StandardOutput.DRAFT_MESH
            ]):
                glb_path = output_dir / f"{run_id}.glb"
                try:
                    mesh.export(str(glb_path))
                    artifacts.append(ArtifactRef(
                        path=str(glb_path),
                        kind="mesh",
                        label="InstantMesh GLB",
                        output_type=StandardOutput.GLB,
                        size_bytes=glb_path.stat().st_size if glb_path.exists() else 0,
                    ))
                except Exception as exc:
                    warnings.append(f"GLB export failed: {exc}")

            # OBJ export
            if StandardOutput.OBJ in output_types:
                obj_path = output_dir / f"{run_id}.obj"
                try:
                    mesh.export(str(obj_path))
                    artifacts.append(ArtifactRef(
                        path=str(obj_path),
                        kind="mesh",
                        label="InstantMesh OBJ",
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
                metadata={"backend": "instantmesh", "num_views": _NUM_VIEWS, "device": self._device},
            )

        except Exception as exc:
            _log.exception("InstantMesh generate() failed")
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="image-to-3d",
                error=str(exc),
            )

    def estimate_requirements(self) -> ResourceEstimate:
        return ResourceEstimate(
            vram_gb=23.0,
            ram_gb=32.0,
            estimated_seconds=90.0,
            notes=["Zero123++ + LRM combined. Can split models to reduce peak VRAM."],
        )

    def export_metadata(self) -> BackendMetadata:
        return BackendMetadata(
            name="InstantMesh",
            version="1.0",
            source_repo="TencentARC/InstantMesh",
            modality="image-to-3d",
            supported_output_types=_SUPPORTED,
            required_vram_gb=23.0,
            notes=[
                "Zero123++ multi-view synthesis (6 views) → InstantMesh LRM reconstruction.",
                "No pip package — requires source clone at /mnt/c/ai_tools/InstantMesh.",
            ],
        )

    def get_capabilities(self) -> ProviderCapability:
        avail = self.check_availability()
        return ProviderCapability(
            provider=self.name,
            label="InstantMesh",
            available=avail.available,
            status=BackendStatus.READY if avail.available else BackendStatus.SCAFFOLD,
            supports_textured_mesh=True,
            supports_glb=True,
            supports_multiview=True,
            supported_output_types=_SUPPORTED,
            notes=avail.reason and [avail.reason] or [],
        )
