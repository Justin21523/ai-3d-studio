"""Mesh decimator — polygon reduction via trimesh quadric decimation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ai3d.core.logging import get_logger
from ai3d.core.models import (
    ArtifactRef,
    BasePostprocessor,
    DecimateRequest,
    GenerationResult,
    StandardOutput,
)
from ai3d.core.storage import ensure_directory

_log = get_logger(__name__)


class MeshDecimator(BasePostprocessor):
    """Reduces polygon count using trimesh simplify_quadric_decimation."""

    name = "decimator"

    def process(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any,
    ) -> GenerationResult:
        request = DecimateRequest(
            input_path=str(input_path),
            output_path=str(output_path),
            target_face_count=kwargs.get("target_face_count"),
            target_ratio=kwargs.get("target_ratio"),
        )
        return self.decimate(request)

    def decimate(self, request: DecimateRequest) -> GenerationResult:
        try:
            import trimesh  # type: ignore[import]
        except ImportError as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="mesh-decimate",
                error=f"trimesh not installed: {exc}",
            )

        input_path = Path(request.input_path)
        output_path = Path(request.output_path)
        ensure_directory(output_path.parent)

        try:
            mesh = trimesh.load(str(input_path), force="mesh")
        except Exception as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="mesh-decimate",
                error=f"Failed to load mesh: {exc}",
            )

        original_faces = len(mesh.faces)

        target: Optional[int] = request.target_face_count
        if target is None and request.target_ratio is not None:
            target = max(1, int(original_faces * request.target_ratio))

        if target is None:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="mesh-decimate",
                error="Either target_face_count or target_ratio must be specified.",
            )

        try:
            decimated = mesh.simplify_quadric_decimation(target)
            decimated.export(str(output_path))
            _log.info(
                "Decimated: %d -> %d faces, saved to %s",
                original_faces,
                len(decimated.faces),
                output_path,
            )
            return GenerationResult(
                success=True,
                provider=self.name,
                task_type="mesh-decimate",
                artifacts=[
                    ArtifactRef(
                        path=str(output_path),
                        kind="mesh",
                        output_type=StandardOutput.CLEANED_MESH,
                    )
                ],
                metadata={"original_faces": original_faces, "final_faces": len(decimated.faces)},
            )
        except Exception as exc:
            _log.error("Decimation failed: %s", exc)
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="mesh-decimate",
                error=str(exc),
            )
