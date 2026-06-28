"""ConditioningPackBuilder — bundles rgb/depth/normal/mask sequences into a structured pack."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from ai3d.core.logging import get_logger
from ai3d.core.models import VideoConditioningPack
from ai3d.core.storage import ensure_directory, write_model

_log = get_logger(__name__)


class ConditioningPackBuilder:
    """Copies video conditioning sequences into a structured output directory."""

    def build(
        self,
        pack: VideoConditioningPack,
        output_dir: Path,
    ) -> Path:
        """
        Organize conditioning sequences under output_dir/:
            rgb/     depth/     normal/     mask/
        and write pack_manifest.yaml.

        Returns output_dir.
        """
        ensure_directory(output_dir)

        _copy_sequence(pack.rgb_sequence_dir, output_dir / "rgb")
        _copy_sequence(pack.depth_sequence_dir, output_dir / "depth")
        _copy_sequence(pack.normal_sequence_dir, output_dir / "normal")
        _copy_sequence(pack.mask_sequence_dir, output_dir / "mask")

        # Update pack paths to point to the new locations
        updated = pack.model_copy(update={
            "rgb_sequence_dir": str(output_dir / "rgb") if pack.rgb_sequence_dir else None,
            "depth_sequence_dir": str(output_dir / "depth") if pack.depth_sequence_dir else None,
            "normal_sequence_dir": str(output_dir / "normal") if pack.normal_sequence_dir else None,
            "mask_sequence_dir": str(output_dir / "mask") if pack.mask_sequence_dir else None,
        })

        manifest_path = output_dir / "pack_manifest.yaml"
        write_model(manifest_path, updated)
        _log.info("Conditioning pack written to: %s", output_dir)

        return output_dir


def _copy_sequence(src_dir: Optional[str], dest_dir: Path) -> None:
    """Copy all PNG/EXR frames from src_dir to dest_dir if src exists."""
    if not src_dir:
        return
    src = Path(src_dir)
    if not src.exists():
        return
    ensure_directory(dest_dir)
    for frame in sorted(src.glob("*.png")):
        shutil.copy2(frame, dest_dir / frame.name)
    for frame in sorted(src.glob("*.exr")):
        shutil.copy2(frame, dest_dir / frame.name)
    _log.debug("Copied %s -> %s", src, dest_dir)
