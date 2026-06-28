"""Mesh cleaner — degenerate face removal, winding fix, duplicate removal."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ai3d.core.logging import get_logger
from ai3d.core.models import (
    ArtifactRef,
    BasePostprocessor,
    GenerationResult,
    MeshCleanRequest,
    MeshCleanResult,
    StandardOutput,
)
from ai3d.core.storage import ensure_directory

_log = get_logger(__name__)


class MeshCleaner(BasePostprocessor):
    """Cleans mesh geometry using trimesh: winding, degenerate faces, duplicates."""

    name = "mesh_cleaner"

    def process(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any,
    ) -> GenerationResult:
        request = MeshCleanRequest(
            input_path=str(input_path),
            output_path=str(output_path),
            remove_degenerate_faces=kwargs.get("remove_degenerate_faces", True),
            make_watertight=kwargs.get("make_watertight", False),
            remove_duplicate_vertices=kwargs.get("remove_duplicate_vertices", True),
        )
        result = self.clean(request)
        if result.success and result.output_path:
            return GenerationResult(
                success=True,
                provider=self.name,
                task_type="mesh-clean",
                artifacts=[
                    ArtifactRef(
                        path=result.output_path,
                        kind="cleaned_mesh",
                        output_type=StandardOutput.CLEANED_MESH,
                    )
                ],
                warnings=result.warnings,
            )
        return GenerationResult(
            success=False,
            provider=self.name,
            task_type="mesh-clean",
            error=result.error,
            warnings=result.warnings,
        )

    def clean(self, request: MeshCleanRequest) -> MeshCleanResult:
        try:
            import trimesh  # type: ignore[import]
        except ImportError as exc:
            return MeshCleanResult(
                success=False,
                error=f"trimesh not installed: {exc}. Run: pip install trimesh",
            )

        input_path = Path(request.input_path)
        output_path = Path(request.output_path)
        ensure_directory(output_path.parent)

        try:
            mesh = trimesh.load(str(input_path), force="mesh")
        except Exception as exc:
            return MeshCleanResult(success=False, error=f"Failed to load mesh: {exc}")

        if not isinstance(mesh, trimesh.Trimesh):
            # Scene with multiple meshes — merge
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(
                    [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                )
            else:
                return MeshCleanResult(success=False, error="Loaded object is not a Trimesh or Scene.")

        original_faces = len(mesh.faces)
        warnings: list[str] = []

        try:
            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fix_normals(mesh)

            if request.remove_degenerate_faces:
                import numpy as np
                areas = trimesh.triangles.area(mesh.triangles)
                mesh.update_faces(areas > 1e-10)

            if request.remove_duplicate_vertices:
                mesh.merge_vertices()
                mesh.remove_unreferenced_vertices()

            if request.make_watertight:
                trimesh.repair.fill_holes(mesh)
                if not mesh.is_watertight:
                    warnings.append("Mesh is still not watertight after fill_holes.")

            mesh.export(str(output_path))
            _log.info(
                "Mesh cleaned: %d -> %d faces, saved to %s",
                original_faces,
                len(mesh.faces),
                output_path,
            )

            return MeshCleanResult(
                success=True,
                output_path=str(output_path),
                original_face_count=original_faces,
                cleaned_face_count=len(mesh.faces),
                warnings=warnings,
            )
        except Exception as exc:
            _log.error("Mesh cleaning failed: %s", exc)
            return MeshCleanResult(success=False, error=str(exc), warnings=warnings)
