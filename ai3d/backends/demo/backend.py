"""Demo backend — deterministic CPU-safe image-to-3D showcase path.

This backend is intentionally not a model. It creates a small representative
mesh and manifest so the CLI, pipeline, API, screenshots, and portfolio demo
can exercise a successful flow without GPU weights, Blender, or ComfyUI.
"""
from __future__ import annotations

import math
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
from ai3d.core.storage import ensure_directory, write_yaml

_log = get_logger(__name__)

_SUPPORTED = [
    StandardOutput.DRAFT_MESH,
    StandardOutput.CLEANED_MESH,
    StandardOutput.TEXTURED_MESH,
    StandardOutput.GLB,
    StandardOutput.OBJ,
    StandardOutput.SPLAT_PLY,
]


class DemoBackend(BaseBackend):
    """Deterministic CPU-only backend for demos and smoke tests."""

    name = "demo"

    def check_availability(self) -> AvailabilityResult:
        try:
            import trimesh  # type: ignore[import]  # noqa: F401
        except ImportError as exc:
            return AvailabilityResult(
                available=False,
                backend=self.name,
                reason=f"trimesh is required for demo mode: {exc}",
            )

        return AvailabilityResult(
            available=True,
            backend=self.name,
            reason="CPU-safe demo mode is available.",
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        output_dir = ensure_directory(Path(request.output_dir))
        image_path = Path(request.input_image_path)
        if not image_path.exists():
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="image-to-3d-demo",
                error=f"Input image not found: {image_path}",
            )

        try:
            import trimesh  # type: ignore[import]
        except ImportError as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="image-to-3d-demo",
                error=f"Missing dependency: {exc}. Run: pip install trimesh",
            )

        mesh = _build_showcase_mesh(trimesh)
        artifacts: list[ArtifactRef] = []
        warnings: list[str] = []

        requested = request.output_types or [StandardOutput.GLB]
        for output_type in requested:
            if output_type not in _SUPPORTED:
                warnings.append(f"Demo backend does not emit {output_type.value}; skipped.")
                continue

            out_path = output_dir / f"demo_asset{_ext_for(output_type)}"
            try:
                mesh.export(str(out_path))
            except Exception as exc:
                warnings.append(f"Export failed for {output_type.value}: {exc}")
                continue

            artifacts.append(
                ArtifactRef(
                    path=str(out_path),
                    kind="mesh",
                    label=f"demo_{output_type.value}",
                    output_type=output_type,
                    size_bytes=out_path.stat().st_size,
                    metadata={
                        "mock_safe": True,
                        "source_image": str(image_path),
                        "vertex_count": int(len(mesh.vertices)),
                        "face_count": int(len(mesh.faces)),
                    },
                )
            )

        manifest_path = output_dir / "demo_manifest.yaml"
        write_yaml(
            manifest_path,
            {
                "mode": "mock-safe-demo",
                "source_image": str(image_path),
                "backend": self.name,
                "requested_outputs": [t.value for t in requested],
                "artifacts": [artifact.path for artifact in artifacts],
                "notes": [
                    "This deterministic mesh proves the orchestration path without model weights.",
                    "Use SF3D or TripoSR backends for real image-to-3D inference.",
                ],
            },
        )
        artifacts.append(
            ArtifactRef(
                path=str(manifest_path),
                kind="manifest",
                label="demo_manifest",
                size_bytes=manifest_path.stat().st_size,
                metadata={"mock_safe": True},
            )
        )

        _log.info("Demo backend exported %d artifacts to %s", len(artifacts), output_dir)
        return GenerationResult(
            success=bool(artifacts),
            provider=self.name,
            task_type="image-to-3d-demo",
            artifacts=artifacts,
            warnings=warnings,
            metadata={
                "mock_safe": True,
                "scenario": request.metadata.get("scenario", "product_turntable"),
                "input_image": str(image_path),
            },
        )

    def estimate_requirements(self) -> ResourceEstimate:
        return ResourceEstimate(
            vram_gb=0.0,
            ram_gb=0.5,
            estimated_seconds=1.0,
            supports_cpu_fallback=True,
            notes=["Deterministic CPU-safe demo path; no model weights required."],
        )

    def export_metadata(self) -> BackendMetadata:
        return BackendMetadata(
            name="AI 3D Studio Demo Backend",
            version="1.0-demo",
            source_repo="local/mock-safe",
            modality="image-to-3d-demo",
            supported_output_types=_SUPPORTED,
            required_vram_gb=0.0,
            notes=[
                "Creates a deterministic representative mesh for public demos.",
                "Exercises CLI/API/pipeline artifact flow without GPU or external services.",
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
            supports_splat=True,
            supported_output_types=meta.supported_output_types,
            notes=meta.notes,
            metadata={
                "source_repo": meta.source_repo,
                "mock_safe": True,
                "requires_gpu": False,
            },
        )


def _build_showcase_mesh(trimesh):
    base = trimesh.creation.icosphere(subdivisions=2, radius=0.72)
    base.apply_scale((1.0, 1.0, 1.18))

    ring = trimesh.creation.torus(
        major_radius=0.88,
        minor_radius=0.035,
        major_segments=72,
        minor_segments=10,
    )
    ring.apply_transform(trimesh.transformations.rotation_matrix(math.radians(18), [1, 0, 0]))

    pedestal = trimesh.creation.cylinder(radius=0.58, height=0.16, sections=72)
    pedestal.apply_translation([0, 0, -0.78])

    mesh = trimesh.util.concatenate([base, ring, pedestal])
    mesh.visual.vertex_colors = _vertex_colors(len(mesh.vertices))
    return mesh


def _vertex_colors(count: int):
    try:
        import numpy as np  # type: ignore[import]
    except ImportError:
        return None

    colors = np.zeros((count, 4), dtype=np.uint8)
    for idx in range(count):
        t = idx / max(count - 1, 1)
        colors[idx] = [
            int(36 + 130 * t),
            int(210 - 80 * t),
            int(180 + 50 * math.sin(t * math.pi)),
            255,
        ]
    return colors


def _ext_for(output_type: StandardOutput) -> str:
    mapping: dict[StandardOutput, str] = {
        StandardOutput.GLB: ".glb",
        StandardOutput.OBJ: ".obj",
        StandardOutput.FBX: ".fbx",
        StandardOutput.SPLAT_PLY: ".ply",
        StandardOutput.DRAFT_MESH: ".glb",
        StandardOutput.CLEANED_MESH: ".glb",
        StandardOutput.TEXTURED_MESH: ".glb",
    }
    return mapping.get(output_type, ".glb")
