"""Wonder3D backend for single-image multi-view RGB and normal generation."""
from __future__ import annotations

import uuid
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

_log = get_logger(__name__)

_SUPPORTED = [
    StandardOutput.MULTIVIEW_IMAGES,
    StandardOutput.NORMAL_MAPS,
]


class Wonder3DBackend(BaseBackend):
    """xxlong0/Wonder3D multi-view diffusion."""

    name = "wonder3d"

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda",
    ) -> None:
        from ai3d.backends.wonder3d.loader import Wonder3DLoader
        self._loader = Wonder3DLoader(model_path=model_path, device=device)
        self._device = device

    def check_availability(self) -> AvailabilityResult:
        found, missing = self._loader.get_model_paths()
        available = self._loader.is_available()
        reason = None if available else (
            "Wonder3D source or dependencies not found — clone xxlong0/Wonder3D"
            if found else
            "Wonder3D weights not found. Download from xxlong0/Wonder3D."
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
        output_types = request.output_types or _SUPPORTED

        try:
            model = self._loader.load()
        except Exception as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="multiview-diffusion",
                error=f"Failed to load Wonder3D model: {exc}",
            )

        try:
            from PIL import Image
            import torch

            image = Image.open(str(request.input_image_path)).convert("RGBA")
            run_id = request.request_id or uuid.uuid4().hex
            _log.info("Running Wonder3D inference on %s", request.input_image_path)

            views, normals = self._run_inference(model, image, request.seed or 0, request.backend_params)
            artifacts: list[ArtifactRef] = []
            warnings: list[str] = []

            if StandardOutput.MULTIVIEW_IMAGES in output_types:
                mv_dir = ensure_directory(output_dir / "multiview")
                for idx, view in enumerate(views):
                    path = mv_dir / f"{run_id}_view_{idx:02d}.png"
                    view.save(str(path))
                    artifacts.append(ArtifactRef(
                        path=str(path),
                        kind="multiview",
                        label=f"Wonder3D RGB view {idx}",
                        output_type=StandardOutput.MULTIVIEW_IMAGES,
                        size_bytes=path.stat().st_size if path.exists() else 0,
                    ))

            if StandardOutput.NORMAL_MAPS in output_types:
                normal_dir = ensure_directory(output_dir / "normals")
                for idx, normal in enumerate(normals):
                    path = normal_dir / f"{run_id}_normal_{idx:02d}.png"
                    normal.save(str(path))
                    artifacts.append(ArtifactRef(
                        path=str(path),
                        kind="normal",
                        label=f"Wonder3D normal map {idx}",
                        output_type=StandardOutput.NORMAL_MAPS,
                        size_bytes=path.stat().st_size if path.exists() else 0,
                    ))

            if not artifacts:
                return GenerationResult(
                    success=False,
                    provider=self.name,
                    task_type="multiview-diffusion",
                    error="No requested Wonder3D artifacts were produced.",
                    warnings=warnings,
                )

            return GenerationResult(
                success=True,
                provider=self.name,
                task_type="multiview-diffusion",
                artifacts=artifacts,
                warnings=warnings,
                metadata={"backend": "wonder3d", "num_views": len(views), "device": self._device},
            )
        except Exception as exc:
            _log.exception("Wonder3D generate() failed")
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="multiview-diffusion",
                error=str(exc),
            )

    def _run_inference(self, model: Any, image: Any, seed: int, params: dict) -> tuple[list[Any], list[Any]]:
        import torch

        generator = torch.Generator(device="cpu").manual_seed(seed)
        call_kwargs = {
            "num_inference_steps": params.get("num_inference_steps", 50),
            "generator": generator,
        }

        with torch.no_grad():
            if hasattr(model, "infer"):
                result = model.infer(image, **call_kwargs)
            elif hasattr(model, "generate"):
                result = model.generate(image, **call_kwargs)
            elif callable(model):
                result = model(image, **call_kwargs)
            else:
                raise RuntimeError("Wonder3D model does not expose infer(), generate(), or __call__().")

        views = self._extract_images(result, ("views", "images", "rgb", "multiview"))
        normals = self._extract_images(result, ("normals", "normal_maps", "normal"))
        if len(views) < 6 or len(normals) < 6:
            raise RuntimeError(
                f"Wonder3D inference returned {len(views)} RGB views and {len(normals)} normals; expected 6 each."
            )
        return views[:6], normals[:6]

    def _extract_images(self, result: Any, keys: tuple[str, ...]) -> list[Any]:
        for key in keys:
            if isinstance(result, dict) and key in result:
                return list(result[key])
            if hasattr(result, key):
                return list(getattr(result, key))
        return []

    def estimate_requirements(self) -> ResourceEstimate:
        return ResourceEstimate(
            vram_gb=16.0,
            ram_gb=24.0,
            estimated_seconds=60.0,
            notes=["Diffusion-only step. Downstream mesh reconstruction is separate."],
        )

    def export_metadata(self) -> BackendMetadata:
        return BackendMetadata(
            name="Wonder3D",
            version="1.0",
            source_repo="xxlong0/Wonder3D",
            modality="multiview-diffusion",
            supported_output_types=_SUPPORTED,
            required_vram_gb=16.0,
            notes=[
                "Produces 6-view RGB images and 6 normal maps.",
                "Does not produce a 3D mesh directly.",
            ],
        )

    def get_capabilities(self) -> ProviderCapability:
        avail = self.check_availability()
        return ProviderCapability(
            provider=self.name,
            label="Wonder3D",
            available=avail.available,
            status=BackendStatus.READY if avail.available else BackendStatus.SCAFFOLD,
            supports_multiview=True,
            supports_normal_maps=True,
            supported_output_types=_SUPPORTED,
            notes=avail.reason and [avail.reason] or [],
        )
