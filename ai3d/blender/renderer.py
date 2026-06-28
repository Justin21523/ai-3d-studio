"""TurntableRenderer — high-level wrapper around BlenderBridge for turntable renders."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ai3d.core.logging import get_logger
from ai3d.core.models import BlenderRenderPass, BlenderRenderResult, TurntableSpec
from ai3d.core.paths import BLENDER_SCRIPTS_DIR
from ai3d.blender.bridge import BlenderBridge
from ai3d.blender.scene_builder import build_turntable_spec

_log = get_logger(__name__)


class TurntableRenderer:
    """Renders 360-degree turntable animations via headless Blender."""

    def __init__(self, bridge: Optional[BlenderBridge] = None) -> None:
        self._bridge = bridge or BlenderBridge()

    def render(self, spec: TurntableSpec) -> BlenderRenderResult:
        scene_spec = build_turntable_spec(
            asset_path=Path(spec.asset_path),
            output_dir=Path(spec.output_dir),
            frame_count=spec.frame_count,
            fps=spec.fps,
            resolution_x=spec.resolution_x,
            resolution_y=spec.resolution_y,
            camera_elevation_deg=spec.camera_elevation_deg,
            camera_distance=spec.camera_distance,
            render_passes=spec.render_passes,
            hdri_path=Path(spec.hdri_path) if spec.hdri_path else None,
        )
        script = BLENDER_SCRIPTS_DIR / "render_turntable.py"
        _log.info(
            "TurntableRenderer: %d frames, passes=%s -> %s",
            spec.frame_count,
            [p.value for p in spec.render_passes],
            spec.output_dir,
        )
        return self._bridge.launch_headless(scene_spec, script)

    def render_asset(
        self,
        asset_path: Path,
        output_dir: Path,
        frame_count: int = 72,
        fps: int = 24,
        resolution: tuple[int, int] = (1024, 1024),
        passes: Optional[List[BlenderRenderPass]] = None,
        hdri_path: Optional[Path] = None,
    ) -> BlenderRenderResult:
        spec = TurntableSpec(
            asset_path=str(asset_path),
            output_dir=str(output_dir),
            frame_count=frame_count,
            fps=fps,
            resolution_x=resolution[0],
            resolution_y=resolution[1],
            render_passes=passes or [BlenderRenderPass.RGB],
            hdri_path=str(hdri_path) if hdri_path else None,
        )
        return self.render(spec)
