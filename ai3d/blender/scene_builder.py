"""BlenderSceneSpec builder — convenience constructors for common operations."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ai3d.core.models import BlenderRenderPass, BlenderSceneSpec, TurntableSpec


def build_turntable_spec(
    asset_path: Path,
    output_dir: Path,
    frame_count: int = 72,
    fps: int = 24,
    resolution_x: int = 1024,
    resolution_y: int = 1024,
    camera_elevation_deg: float = 20.0,
    camera_distance: float = 2.5,
    render_passes: Optional[List[BlenderRenderPass]] = None,
    hdri_path: Optional[Path] = None,
) -> BlenderSceneSpec:
    """Build a BlenderSceneSpec for a turntable render job."""
    turntable = TurntableSpec(
        asset_path=str(asset_path),
        output_dir=str(output_dir),
        frame_count=frame_count,
        fps=fps,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        camera_elevation_deg=camera_elevation_deg,
        camera_distance=camera_distance,
        render_passes=render_passes or [BlenderRenderPass.RGB],
        hdri_path=str(hdri_path) if hdri_path else None,
    )
    return BlenderSceneSpec(
        operation="turntable",
        mesh_path=str(asset_path),
        output_dir=str(output_dir),
        turntable=turntable,
        passes=render_passes or [BlenderRenderPass.RGB],
    )


def build_passes_spec(
    asset_path: Path,
    output_dir: Path,
    passes: List[BlenderRenderPass],
    frame_count: int = 1,
) -> BlenderSceneSpec:
    """Build a BlenderSceneSpec for rendering specific render passes."""
    return BlenderSceneSpec(
        operation="passes",
        mesh_path=str(asset_path),
        output_dir=str(output_dir),
        passes=passes,
        script_overrides={"frame_count": frame_count},
    )
