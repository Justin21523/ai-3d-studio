"""Mesh format converter — GLB / OBJ / FBX / PLY / STL via trimesh."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ai3d.core.logging import get_logger
from ai3d.core.models import (
    ArtifactRef,
    BasePostprocessor,
    FormatConvertRequest,
    GenerationResult,
    StandardOutput,
)
from ai3d.core.storage import ensure_directory

_log = get_logger(__name__)

_FORMAT_TO_OUTPUT_TYPE = {
    "glb": StandardOutput.GLB,
    "fbx": StandardOutput.FBX,
    "obj": StandardOutput.OBJ,
    "ply": StandardOutput.SPLAT_PLY,
    "stl": StandardOutput.DRAFT_MESH,
}


class FormatConverter(BasePostprocessor):
    """Converts mesh between formats using trimesh (format inferred from extension)."""

    name = "format_converter"

    def process(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any,
    ) -> GenerationResult:
        fmt = kwargs.get("output_format", output_path.suffix.lstrip(".").lower())
        request = FormatConvertRequest(
            input_path=str(input_path),
            output_path=str(output_path),
            output_format=fmt,  # type: ignore[arg-type]
        )
        return self.convert(request)

    def convert(self, request: FormatConvertRequest) -> GenerationResult:
        try:
            import trimesh  # type: ignore[import]
        except ImportError as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="mesh-convert",
                error=f"trimesh not installed: {exc}",
            )

        input_path = Path(request.input_path)
        output_path = Path(request.output_path)
        ensure_directory(output_path.parent)

        try:
            mesh_or_scene = trimesh.load(str(input_path))
            mesh_or_scene.export(str(output_path))
            output_type = _FORMAT_TO_OUTPUT_TYPE.get(request.output_format, StandardOutput.DRAFT_MESH)
            _log.info("Converted %s -> %s", input_path.suffix, output_path.suffix)
            return GenerationResult(
                success=True,
                provider=self.name,
                task_type="mesh-convert",
                artifacts=[
                    ArtifactRef(
                        path=str(output_path),
                        kind="mesh",
                        output_type=output_type,
                        size_bytes=output_path.stat().st_size,
                    )
                ],
            )
        except Exception as exc:
            _log.error("Format conversion failed: %s", exc)
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="mesh-convert",
                error=str(exc),
            )
