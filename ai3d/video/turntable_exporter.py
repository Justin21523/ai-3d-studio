"""TurntableExporter — drives Blender turntable render and packages results."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

from ai3d.core.logging import get_logger
from ai3d.core.models import BlenderRenderPass, TurntableSpec, VideoConditioningPack
from ai3d.blender.renderer import TurntableRenderer

_log = get_logger(__name__)


class TurntableExporter:
    """Renders a turntable from a 3D asset and returns a VideoConditioningPack."""

    def __init__(self, renderer: Optional[TurntableRenderer] = None) -> None:
        self._renderer = renderer or TurntableRenderer()

    def export(
        self,
        asset_path: Path,
        output_dir: Path,
        frame_count: int = 72,
        fps: int = 24,
        resolution: tuple[int, int] = (1024, 1024),
        passes: Optional[List[BlenderRenderPass]] = None,
        hdri_path: Optional[Path] = None,
        pack_id: Optional[str] = None,
        source_asset_id: Optional[str] = None,
    ) -> VideoConditioningPack:
        if passes is None:
            passes = [BlenderRenderPass.RGB]

        spec = TurntableSpec(
            asset_path=str(asset_path),
            output_dir=str(output_dir),
            frame_count=frame_count,
            fps=fps,
            resolution_x=resolution[0],
            resolution_y=resolution[1],
            render_passes=passes,
            hdri_path=str(hdri_path) if hdri_path else None,
        )

        result = self._renderer.render(spec)

        pid = pack_id or str(uuid.uuid4())[:8]
        sid = source_asset_id or asset_path.stem

        if not result.success:
            _log.warning("Turntable render failed: %s", result.error)

        pass_dirs = result.pass_dirs or {}

        return VideoConditioningPack(
            pack_id=pid,
            source_asset_id=sid,
            rgb_sequence_dir=pass_dirs.get("rgb"),
            depth_sequence_dir=pass_dirs.get("depth"),
            normal_sequence_dir=pass_dirs.get("normal"),
            mask_sequence_dir=pass_dirs.get("mask"),
            frame_count=len(result.frame_paths),
            fps=fps,
            resolution_x=resolution[0],
            resolution_y=resolution[1],
            metadata={
                "blender_success": result.success,
                "render_passes": [p.value for p in passes],
                "asset_path": str(asset_path),
            },
        )
