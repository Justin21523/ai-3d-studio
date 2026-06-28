"""SF3D (Stable Fast 3D) backend — fully implemented for Milestone 1."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

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
from ai3d.backends.sf3d.loader import SF3DLoader

_log = get_logger(__name__)

_SUPPORTED = [
    StandardOutput.DRAFT_MESH,
    StandardOutput.CLEANED_MESH,
    StandardOutput.TEXTURED_MESH,
    StandardOutput.GLB,
    StandardOutput.OBJ,
]

_TARGET_SIZE = (512, 512)


class SF3DBackend(BaseBackend):
    """Adapter for stabilityai/stable-fast-3d — PBR-textured GLB generation."""

    name = "sf3d"

    def __init__(
        self,
        device: str = "cuda",
        model_path: Optional[Path] = None,
        bake_resolution: int = 1024,
    ) -> None:
        self._device = device
        self._bake_resolution = bake_resolution
        self._loader = SF3DLoader(model_path=model_path, device=device)
        self._model: Any = None

    # ── Availability ──────────────────────────────────────────────────────────

    def check_availability(self) -> AvailabilityResult:
        found, missing = self._loader.get_model_paths()
        available = self._loader.is_available()

        reason: Optional[str] = None
        if not available:
            reason = "Model weights not found on disk."
        else:
            try:
                import torch  # type: ignore[import]
                if self._device == "cuda" and not torch.cuda.is_available():
                    available = False
                    reason = "CUDA requested but torch.cuda.is_available() is False."
            except ImportError:
                available = False
                reason = "PyTorch not installed."

        return AvailabilityResult(
            available=available,
            backend=self.name,
            reason=reason,
            model_paths_found=found,
            model_paths_missing=missing,
        )

    # ── Generation ────────────────────────────────────────────────────────────

    def generate(self, request: GenerationRequest) -> GenerationResult:
        output_dir = ensure_directory(Path(request.output_dir))
        warnings: list[str] = []

        if self._model is None:
            try:
                self._model = self._loader.load()
            except Exception as exc:
                return GenerationResult(
                    success=False,
                    provider=self.name,
                    task_type="image-to-3d",
                    error=str(exc),
                )

        try:
            from PIL import Image  # type: ignore[import]
            import torch  # type: ignore[import]
        except ImportError as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="image-to-3d",
                error=f"Missing dependency: {exc}",
            )

        image_path = Path(request.input_image_path)
        if not image_path.exists():
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="image-to-3d",
                error=f"Input image not found: {image_path}",
            )

        try:
            image = Image.open(image_path).convert("RGBA").resize(_TARGET_SIZE)
        except Exception as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="image-to-3d",
                error=f"Failed to open/resize image: {exc}",
            )

        try:
            _log.info("SF3D: running inference on %s", image_path.name)
            with torch.no_grad():
                mesh, _glob_dict = self._model.run_image(
                    image,
                    bake_resolution=self._bake_resolution,
                )
        except Exception as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="image-to-3d",
                error=f"Inference failed: {exc}",
            )

        artifacts: list[ArtifactRef] = []

        for output_type in request.output_types:
            if output_type not in _SUPPORTED:
                warnings.append(f"SF3D does not support output type: {output_type.value}")
                continue

            try:
                ext = _ext_for(output_type)
                out_path = output_dir / f"output{ext}"
                mesh.export(str(out_path))
                artifacts.append(
                    ArtifactRef(
                        path=str(out_path),
                        kind="mesh",
                        label=output_type.value,
                        output_type=output_type,
                        size_bytes=out_path.stat().st_size,
                    )
                )
                _log.info("SF3D: exported %s -> %s", output_type.value, out_path)
            except Exception as exc:
                warnings.append(f"Export failed for {output_type.value}: {exc}")

        return GenerationResult(
            success=bool(artifacts),
            provider=self.name,
            task_type="image-to-3d",
            artifacts=artifacts,
            warnings=warnings,
        )

    # ── Metadata ──────────────────────────────────────────────────────────────

    def estimate_requirements(self) -> ResourceEstimate:
        return ResourceEstimate(
            vram_gb=6.0,
            ram_gb=16.0,
            estimated_seconds=20.0,
            supports_cpu_fallback=False,
            notes=["PBR texture baking at 1024px adds ~8s. ~20s total on RTX 3090."],
        )

    def export_metadata(self) -> BackendMetadata:
        return BackendMetadata(
            name="Stable Fast 3D",
            version="1.0",
            source_repo="stabilityai/stable-fast-3d",
            modality="image-to-3d",
            supported_output_types=_SUPPORTED,
            required_vram_gb=6.0,
            notes=[
                "Produces PBR-textured GLB natively via UV baking.",
                "Best choice for Blender editing workflows that need real materials.",
            ],
        )

    def get_capabilities(self) -> ProviderCapability:
        meta = self.export_metadata()
        avail = self.check_availability()
        return ProviderCapability(
            provider=self.name,
            label=meta.name,
            available=avail.available,
            status=BackendStatus.READY if avail.available else BackendStatus.UNAVAILABLE,
            supports_textured_mesh=True,
            supports_glb=True,
            supported_output_types=meta.supported_output_types,
            notes=meta.notes,
            metadata={"source_repo": meta.source_repo},
        )


def _ext_for(output_type: StandardOutput) -> str:
    mapping = {
        StandardOutput.GLB: ".glb",
        StandardOutput.OBJ: ".obj",
        StandardOutput.DRAFT_MESH: ".glb",
        StandardOutput.CLEANED_MESH: ".glb",
        StandardOutput.TEXTURED_MESH: ".glb",
    }
    return mapping.get(output_type, ".glb")
