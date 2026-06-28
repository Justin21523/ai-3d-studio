"""UV unwrapper — xatlas-based UV atlas generation.

Milestone 1: best-effort (xatlas may not be installed).
Milestone 2: production quality with texture baking.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ai3d.core.logging import get_logger
from ai3d.core.models import (
    ArtifactRef,
    BasePostprocessor,
    GenerationResult,
    StandardOutput,
)
from ai3d.core.storage import ensure_directory

_log = get_logger(__name__)


class UVUnwrapper(BasePostprocessor):
    """Generates UV atlas using xatlas and rebuilds trimesh with UV coordinates."""

    name = "uv_unwrapper"

    def process(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any,
    ) -> GenerationResult:
        try:
            import trimesh  # type: ignore[import]
            import xatlas  # type: ignore[import]
            import numpy as np  # type: ignore[import]
        except ImportError as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="uv-unwrap",
                error=f"Missing dependency: {exc}. Run: pip install xatlas trimesh numpy",
            )

        ensure_directory(output_path.parent)

        try:
            mesh = trimesh.load(str(input_path), force="mesh")
        except Exception as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="uv-unwrap",
                error=f"Failed to load mesh: {exc}",
            )

        try:
            vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
            new_vertices = mesh.vertices[vmapping]

            uv_mesh = trimesh.Trimesh(
                vertices=new_vertices,
                faces=indices,
            )
            if hasattr(mesh.visual, "vertex_colors"):
                uv_mesh.visual.vertex_colors = mesh.visual.vertex_colors[vmapping]

            uv_mesh.export(str(output_path))
            _log.info("UV unwrap complete: %d vertices, saved to %s", len(new_vertices), output_path)

            return GenerationResult(
                success=True,
                provider=self.name,
                task_type="uv-unwrap",
                artifacts=[
                    ArtifactRef(
                        path=str(output_path),
                        kind="uv_mesh",
                        output_type=StandardOutput.TEXTURED_MESH,
                        size_bytes=output_path.stat().st_size,
                    )
                ],
                metadata={"uv_vertex_count": len(new_vertices)},
            )
        except Exception as exc:
            _log.error("UV unwrap failed: %s", exc)
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="uv-unwrap",
                error=str(exc),
            )
