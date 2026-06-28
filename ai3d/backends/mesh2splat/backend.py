"""Mesh2Splat backend for practical CPU mesh-to-3DGS PLY conversion."""
from __future__ import annotations

import math
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
    StandardOutput.SPLAT_PLY,
]


class Mesh2SplatBackend(BaseBackend):
    """CPU mesh-to-3DGS conversion using trimesh surface sampling."""

    name = "mesh2splat"

    def __init__(self, model_path: Optional[Path] = None) -> None:
        from ai3d.backends.mesh2splat.loader import Mesh2SplatLoader
        self._loader = Mesh2SplatLoader(model_path=model_path)

    def check_availability(self) -> AvailabilityResult:
        found, missing = self._loader.get_model_paths()
        available = self._loader.is_available()
        return AvailabilityResult(
            available=available,
            backend=self.name,
            reason="Ready" if available else "Mesh2Splat requires trimesh and open3d for the CPU path.",
            model_paths_found=found,
            model_paths_missing=missing,
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        output_dir = ensure_directory(Path(request.output_dir))

        try:
            self._loader.load()
        except Exception as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="mesh-to-splat",
                error=f"Failed to load Mesh2Splat dependencies: {exc}",
            )

        try:
            import numpy as np
            import trimesh

            mesh_path = Path(request.input_image_path)
            mesh = trimesh.load(str(mesh_path), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(mesh.dump()))
            if mesh.is_empty or len(mesh.faces) == 0:
                raise RuntimeError(f"Mesh has no faces: {mesh_path}")

            count = int(request.backend_params.get("num_points", 100_000))
            count = max(1, count)
            points, face_indices = trimesh.sample.sample_surface(mesh, count)
            normals = np.asarray(mesh.face_normals)[face_indices]
            colors = self._sample_colors(mesh, face_indices)
            scale = float(request.backend_params.get("scale", self._estimate_scale(mesh, count)))
            opacity = float(request.backend_params.get("opacity", 0.75))

            run_id = request.request_id or uuid.uuid4().hex
            ply_path = output_dir / f"{run_id}.ply"
            self._write_3dgs_ply(ply_path, points, normals, colors, opacity, scale)

            return GenerationResult(
                success=True,
                provider=self.name,
                task_type="mesh-to-splat",
                artifacts=[
                    ArtifactRef(
                        path=str(ply_path),
                        kind="splat",
                        label="Mesh2Splat 3DGS PLY",
                        output_type=StandardOutput.SPLAT_PLY,
                        size_bytes=ply_path.stat().st_size if ply_path.exists() else 0,
                    )
                ],
                metadata={"backend": "mesh2splat", "num_points": count},
            )
        except Exception as exc:
            _log.exception("Mesh2Splat generate() failed")
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="mesh-to-splat",
                error=str(exc),
            )

    def _sample_colors(self, mesh, face_indices):
        import numpy as np

        default = np.full((len(face_indices), 3), 0.7, dtype=float)
        visual = getattr(mesh, "visual", None)
        vertex_colors = getattr(visual, "vertex_colors", None)
        if vertex_colors is None or len(vertex_colors) == 0:
            return default

        faces = np.asarray(mesh.faces)[face_indices]
        rgb = np.asarray(vertex_colors)[:, :3].astype(float) / 255.0
        return rgb[faces].mean(axis=1)

    def _estimate_scale(self, mesh, count: int) -> float:
        area = max(float(mesh.area), 1e-9)
        return math.sqrt(area / max(count, 1)) * 0.5

    def _write_3dgs_ply(self, path: Path, points, normals, colors, opacity: float, scale: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = len(points)
        with path.open("w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {rows}\n")
            for name in (
                "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
                "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
            ):
                f.write(f"property float {name}\n")
            f.write("end_header\n")

            log_scale = math.log(max(scale, 1e-8))
            opacity_logit = math.log(max(opacity, 1e-6) / max(1.0 - opacity, 1e-6))
            for point, normal, color in zip(points, normals, colors):
                f.write(
                    "{:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} "
                    "{:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} "
                    "{:.8f} {:.8f} {:.8f} {:.8f}\n".format(
                        point[0], point[1], point[2],
                        normal[0], normal[1], normal[2],
                        color[0], color[1], color[2],
                        opacity_logit,
                        log_scale, log_scale, log_scale,
                        1.0, 0.0, 0.0, 0.0,
                    )
                )

    def estimate_requirements(self) -> ResourceEstimate:
        return ResourceEstimate(
            vram_gb=0.0,
            ram_gb=8.0,
            estimated_seconds=30.0,
            supports_cpu_fallback=True,
            notes=["CPU mesh sampling conversion; no diffusion model required."],
        )

    def export_metadata(self) -> BackendMetadata:
        return BackendMetadata(
            name="Mesh2Splat",
            version="1.0",
            source_repo="mesh2splat/cpu-fallback",
            modality="mesh-to-splat",
            supported_output_types=_SUPPORTED,
            required_vram_gb=0.0,
            notes=[
                "Converts GLB/OBJ meshes to 3D Gaussian Splat PLY via surface sampling.",
                "Uses trimesh for CPU conversion; learned model weights are optional.",
            ],
        )

    def get_capabilities(self) -> ProviderCapability:
        avail = self.check_availability()
        return ProviderCapability(
            provider=self.name,
            label="Mesh2Splat",
            available=avail.available,
            status=BackendStatus.READY if avail.available else BackendStatus.SCAFFOLD,
            supports_splat=True,
            supported_output_types=_SUPPORTED,
            notes=avail.reason and [avail.reason] or [],
        )
