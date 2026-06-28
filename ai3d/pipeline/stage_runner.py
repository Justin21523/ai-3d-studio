"""Stage runner — try/except wrapper for individual pipeline stages."""
from __future__ import annotations

import traceback
from typing import Callable

from ai3d.core.logging import get_logger
from ai3d.core.models import PipelineManifest, PipelineStage

_log = get_logger(__name__)


def run_stage(
    stage: PipelineStage,
    manifest: PipelineManifest,
    fn: Callable[[], None],
    skip: bool = False,
) -> bool:
    """
    Execute fn() as a pipeline stage.

    Updates manifest.stages_completed or stages_failed accordingly.
    Returns True if the stage succeeded (or was skipped).
    """
    if skip:
        if stage not in manifest.stages_skipped:
            manifest.stages_skipped.append(stage)
        _log.debug("Stage skipped: %s", stage.value)
        return True

    _log.info("Stage starting: %s", stage.value)
    try:
        fn()
        if stage not in manifest.stages_completed:
            manifest.stages_completed.append(stage)
        _log.info("Stage complete: %s", stage.value)
        return True
    except Exception as exc:
        if stage not in manifest.stages_failed:
            manifest.stages_failed.append(stage)
        tb = traceback.format_exc()
        manifest.warnings.append(f"Stage {stage.value} failed: {exc}")
        _log.error("Stage failed: %s\n%s", stage.value, tb)
        return False
