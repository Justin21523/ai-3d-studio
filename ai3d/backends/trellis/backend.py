"""TRELLIS backend — full M2 implementation.

Requires the TRELLIS conda environment from microsoft/TRELLIS.
Model weights: configs/models.yaml → local_path or /mnt/c/ai_models/vision/trellis
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
    StandardOutput.SPLAT_PLY,
    StandardOutput.NORMAL_MAPS,
]


class TRELLISBackend(BaseBackend):
    """microsoft/TRELLIS image-to-3D — structured latent diffusion → 3DGS + textured mesh."""

    name = "trellis"

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda",
    ) -> None:
        from ai3d.backends.trellis.loader import TRELLISLoader
        self._loader = TRELLISLoader(model_path=model_path, device=device)
        self._device = device

    def check_availability(self) -> AvailabilityResult:
        found, missing = self._loader.get_model_paths()
        available = self._loader.is_available()
        reason = None if available else (
            "TRELLIS package not installed — follow setup at https://github.com/microsoft/TRELLIS"
            if found else
            f"TRELLIS weights not found. Run: huggingface-cli download microsoft/TRELLIS-image-to-3D"
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
            pipeline = self._loader.load()
        except Exception as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="image-to-3d",
                error=f"Failed to load TRELLIS model: {exc}",
            )

        try:
            from PIL import Image

            image = Image.open(str(request.input_image_path)).convert("RGBA")
            _log.info("Running TRELLIS inference on %s", request.input_image_path)

            outputs = pipeline.run(image, seed=request.seed or 0)

            artifacts: list[ArtifactRef] = []
            warnings: list[str] = []

            # 3DGS splat output
            if StandardOutput.SPLAT_PLY in output_types:
                splat_path = output_dir / f"{request.request_id or uuid.uuid4().hex}_splat.ply"
                try:
                    outputs["gaussian"][0].save_ply(str(splat_path))
                    artifacts.append(ArtifactRef(
                        path=str(splat_path),
                        kind="splat",
                        label="3DGS splat",
                        output_type=StandardOutput.SPLAT_PLY,
                        size_bytes=splat_path.stat().st_size,
                    ))
                except Exception as exc:
                    warnings.append(f"3DGS export failed: {exc}")

            # Textured mesh / GLB output
            if any(t in output_types for t in [
                StandardOutput.GLB,
                StandardOutput.TEXTURED_MESH,
                StandardOutput.DRAFT_MESH,
            ]):
                glb_path = output_dir / f"{request.request_id or uuid.uuid4().hex}.glb"
                try:
                    outputs["mesh"][0].export(str(glb_path))
                    size = glb_path.stat().st_size if glb_path.exists() else 0
                    artifacts.append(ArtifactRef(
                        path=str(glb_path),
                        kind="mesh",
                        label="Textured GLB",
                        output_type=StandardOutput.GLB,
                        size_bytes=size,
                    ))
                except Exception as exc:
                    warnings.append(f"GLB export failed: {exc}")

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
                metadata={"backend": "trellis", "device": self._device},
            )

        except Exception as exc:
            _log.exception("TRELLIS generate() failed")
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="image-to-3d",
                error=str(exc),
            )

    def estimate_requirements(self) -> ResourceEstimate:
        return ResourceEstimate(
            vram_gb=16.0,
            ram_gb=24.0,
            estimated_seconds=120.0,
            notes=["TRELLIS.2 variant may need ~20 GB VRAM. Enable attention slicing for lower VRAM."],
        )

    def export_metadata(self) -> BackendMetadata:
        return BackendMetadata(
            name="TRELLIS",
            version="1.0",
            source_repo="microsoft/TRELLIS",
            modality="image-to-3d",
            supported_output_types=_SUPPORTED,
            required_vram_gb=16.0,
            notes=[
                "Structured latent diffusion → 3DGS + textured mesh simultaneously.",
                "Highest quality open-source image-to-3D as of early 2025.",
                "Supports both TRELLIS and TRELLIS.2 checkpoints via model_path.",
            ],
        )

    def get_capabilities(self) -> ProviderCapability:
        avail = self.check_availability()
        return ProviderCapability(
            provider=self.name,
            label="TRELLIS",
            available=avail.available,
            status=BackendStatus.READY if avail.available else BackendStatus.SCAFFOLD,
            supports_textured_mesh=True,
            supports_glb=True,
            supports_splat=True,
            supports_normal_maps=True,
            supported_output_types=_SUPPORTED,
            notes=avail.reason and [avail.reason] or [],
        )
