"""BatchRunner — iterates a list of images through GenerationPipeline."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Optional

from ai3d.core.logging import get_logger
from ai3d.core.models import BatchManifest, GenerationRequest, PipelineStage, StandardOutput
from ai3d.core.storage import ensure_directory, write_model
from ai3d.pipeline.generation_pipeline import GenerationPipeline

_log = get_logger(__name__)


class BatchRunner:
    def __init__(self, pipeline: Optional[GenerationPipeline] = None) -> None:
        self._pipeline = pipeline or GenerationPipeline()

    def run_batch(self, manifest: BatchManifest) -> BatchManifest:
        output_root = ensure_directory(Path(manifest.output_root))

        for image_path_str in manifest.input_images:
            image_path = Path(image_path_str)
            run_id = str(uuid.uuid4())[:8]
            run_output_dir = output_root / run_id

            _log.info("Batch [%s/%d]: %s -> %s",
                      manifest.batch_id, len(manifest.input_images), image_path.name, run_id)

            request = GenerationRequest(
                request_id=run_id,
                input_image_path=str(image_path),
                backend=manifest.backend,
                output_types=manifest.output_types or [StandardOutput.GLB],
                output_dir=str(run_output_dir),
            )

            try:
                result_manifest = self._pipeline.run(request)
                manifest.runs.append(run_id)

                if (result_manifest.final_result and result_manifest.final_result.success) or \
                   PipelineStage.GENERATION_3D in result_manifest.stages_completed:
                    manifest.completed += 1
                else:
                    manifest.failed += 1
            except Exception as exc:
                _log.error("Batch run %s failed: %s", run_id, exc)
                manifest.failed += 1
                manifest.runs.append(run_id)

            # Persist batch manifest after each run for resumability
            write_model(output_root / "batch_manifest.yaml", manifest)

        _log.info("Batch %s done: %d completed, %d failed",
                  manifest.batch_id, manifest.completed, manifest.failed)
        return manifest

    @staticmethod
    def collect_images(input_dir: Path, limit: Optional[int] = None) -> List[str]:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        images = sorted(
            str(p) for p in input_dir.iterdir()
            if p.suffix.lower() in exts
        )
        if limit:
            images = images[:limit]
        return images
