"""CRM (Convolutional Reconstruction Model) backend — full M2 implementation.

Two-stage pipeline: Zero123++ multi-view (RGBN) → CRM reconstruction.
Source: clone Zhengyi-Wang/CRM to /mnt/c/ai_tools/CRM.
Weights: configs/models.yaml → local_path or /mnt/c/ai_models/vision/crm
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
    StandardOutput.NORMAL_MAPS,
]


class CRMBackend(BaseBackend):
    """Zhengyi-Wang/CRM — Zero123++ RGBN multi-view → convolutional mesh reconstruction."""

    name = "crm"

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda",
    ) -> None:
        from ai3d.backends.crm.loader import CRMLoader
        self._loader = CRMLoader(model_path=model_path, device=device)
        self._device = device

    def check_availability(self) -> AvailabilityResult:
        found, missing = self._loader.get_model_paths()
        available = self._loader.is_available()
        reason = None if available else (
            "CRM source or diffusers not found — clone Zhengyi-Wang/CRM"
            if found else
            "CRM weights not found. Download from Zhengyi-Wang/CRM on HuggingFace."
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
            crm_model = self._loader.load()
        except Exception as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="image-to-3d",
                error=f"Failed to load CRM model: {exc}",
            )

        try:
            from PIL import Image
            import torch

            image = Image.open(str(request.input_image_path)).convert("RGBA")
            run_id = request.request_id or uuid.uuid4().hex

            _log.info("CRM: generating multi-view RGBN images")

            # Load Zero123++ for multi-view generation
            from diffusers import DiffusionPipeline  # type: ignore[import]
            zero123_pipeline = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.2",
                custom_pipeline="sudo-ai/zero123plus-pipeline",
                torch_dtype=torch.float16,
            ).to(self._device)

            with torch.no_grad():
                mv_result = zero123_pipeline(
                    image,
                    num_inference_steps=75,
                    generator=torch.Generator(device="cpu").manual_seed(request.seed or 0),
                ).images[0]

            artifacts: list[ArtifactRef] = []
            warnings: list[str] = []

            # Save multi-view if requested
            if StandardOutput.MULTIVIEW_IMAGES in output_types:
                mv_path = output_dir / f"{run_id}_multiview.png"
                mv_result.save(str(mv_path))
                artifacts.append(ArtifactRef(
                    path=str(mv_path),
                    kind="multiview",
                    label="CRM RGBN multi-view strip",
                    output_type=StandardOutput.MULTIVIEW_IMAGES,
                    size_bytes=mv_path.stat().st_size,
                ))

            # CRM reconstruction
            _log.info("CRM: mesh reconstruction")
            with torch.no_grad():
                mesh = crm_model(mv_result)

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
                        label="CRM GLB",
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
                        label="CRM OBJ",
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
                metadata={"backend": "crm", "device": self._device},
            )

        except Exception as exc:
            _log.exception("CRM generate() failed")
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="image-to-3d",
                error=str(exc),
            )

    def estimate_requirements(self) -> ResourceEstimate:
        return ResourceEstimate(
            vram_gb=8.0,
            ram_gb=16.0,
            estimated_seconds=45.0,
            notes=["Low-VRAM alternative. Zero123++ + CRM together fit in ~8 GB."],
        )

    def export_metadata(self) -> BackendMetadata:
        return BackendMetadata(
            name="CRM",
            version="1.0",
            source_repo="Zhengyi-Wang/CRM",
            modality="image-to-3d",
            supported_output_types=_SUPPORTED,
            required_vram_gb=8.0,
            notes=[
                "Convolutional Reconstruction Model using RGBN multi-view input.",
                "Low-VRAM alternative (~8 GB) when InstantMesh/TRELLIS are too large.",
                "No pip package — requires source clone at /mnt/c/ai_tools/CRM.",
            ],
        )

    def get_capabilities(self) -> ProviderCapability:
        avail = self.check_availability()
        return ProviderCapability(
            provider=self.name,
            label="CRM",
            available=avail.available,
            status=BackendStatus.READY if avail.available else BackendStatus.SCAFFOLD,
            supports_textured_mesh=True,
            supports_glb=True,
            supports_multiview=True,
            supports_normal_maps=True,
            supported_output_types=_SUPPORTED,
            notes=avail.reason and [avail.reason] or [],
        )
