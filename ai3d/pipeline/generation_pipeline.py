"""11-stage generation pipeline — orchestrates the full image-to-3D workflow."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from ai3d.core.logging import get_logger
from ai3d.core.models import (
    ArtifactRef,
    AssetRegistryEntry,
    AssetStatus,
    BlenderRenderPass,
    GenerationRequest,
    GenerationResult,
    MeshCleanRequest,
    PipelineManifest,
    PipelineStage,
    PreprocessRequest,
    StandardOutput,
    TurntableSpec,
)
from ai3d.core.storage import ensure_directory, write_model
from ai3d.pipeline.stage_runner import run_stage

_log = get_logger(__name__)

_SKIP_ALWAYS_STAGES = {
    PipelineStage.MULTIVIEW_GEN,   # scaffold in M1
}


class GenerationPipeline:
    """
    Executes the 11-stage image-to-3D generation pipeline.

    Each stage writes outputs to disk and records them in PipelineManifest.
    The manifest is persisted after each stage for resumability.
    """

    def __init__(
        self,
        backend_registry=None,
        asset_registry=None,
        skip_stages: Optional[List[PipelineStage]] = None,
        skip_blender: bool = False,
        skip_video_pack: bool = False,
        skip_registration: bool = False,
    ) -> None:
        self._skip_stages = set(skip_stages or []) | _SKIP_ALWAYS_STAGES
        self._skip_blender = skip_blender
        self._skip_video_pack = skip_video_pack
        self._skip_registration = skip_registration

        # Lazy imports to avoid hard dep at module load
        self._backend_registry = backend_registry
        self._asset_registry = asset_registry

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, request: GenerationRequest) -> PipelineManifest:
        output_dir = ensure_directory(Path(request.output_dir))
        manifest = PipelineManifest(
            run_id=request.request_id,
            request=request,
        )

        # State threaded between stages
        state: dict = {
            "current_image": Path(request.input_image_path),
            "mesh_path": None,
            "cleaned_mesh_path": None,
            "uv_mesh_path": None,
            "blender_result": None,
            "video_pack": None,
        }

        def save_manifest():
            write_model(output_dir / "pipeline_manifest.yaml", manifest)

        # ── Stage 1: Input ingest ─────────────────────────────────────────────
        def _ingest():
            src = Path(request.input_image_path)
            if not src.exists():
                raise FileNotFoundError(f"Input image not found: {src}")
            dest = output_dir / ("input" + src.suffix)
            shutil.copy2(src, dest)
            state["current_image"] = dest
            manifest.artifacts.append(ArtifactRef(path=str(dest), kind="input_image", label="source"))

        run_stage(PipelineStage.INPUT_INGEST, manifest, _ingest,
                  skip=PipelineStage.INPUT_INGEST in self._skip_stages)
        save_manifest()

        # ── Stage 2: Background removal ───────────────────────────────────────
        def _bg_remove():
            if not request.remove_background:
                return
            from ai3d.preprocessors.background_removal import BackgroundRemovalPreprocessor
            pre_dir = ensure_directory(output_dir / "preprocessed")
            result = BackgroundRemovalPreprocessor().process(
                PreprocessRequest(
                    input_path=str(state["current_image"]),
                    output_dir=str(pre_dir),
                )
            )
            if result.success and result.output_path:
                state["current_image"] = Path(result.output_path)
                manifest.artifacts.append(ArtifactRef(path=result.output_path, kind="nobg_image"))
            elif not result.success:
                manifest.warnings.append(f"Background removal failed: {result.error}")

        run_stage(PipelineStage.BACKGROUND_REMOVAL, manifest, _bg_remove,
                  skip=PipelineStage.BACKGROUND_REMOVAL in self._skip_stages)
        save_manifest()

        # ── Stage 3: Quality check ────────────────────────────────────────────
        def _quality_check():
            if not request.run_quality_check:
                return
            from ai3d.preprocessors.quality_check import QualityChecker
            qc = QualityChecker().check(state["current_image"])
            if not qc.passed:
                for issue in qc.issues:
                    manifest.warnings.append(f"Quality check: {issue}")
            manifest.metadata["quality_check"] = qc.model_dump(mode="json")

        run_stage(PipelineStage.QUALITY_CHECK, manifest, _quality_check,
                  skip=PipelineStage.QUALITY_CHECK in self._skip_stages)
        save_manifest()

        # ── Stage 4: Multi-view gen (scaffold M1) ─────────────────────────────
        run_stage(PipelineStage.MULTIVIEW_GEN, manifest, lambda: None,
                  skip=True)
        save_manifest()

        # ── Stage 5: 3D generation ────────────────────────────────────────────
        def _generate_3d():
            registry = self._get_backend_registry()
            backend = registry.get(request.backend)
            backend_output_types = [
                t for t in request.output_types
                if t not in (StandardOutput.SPLAT_PLY, StandardOutput.SPLAT_KSPLAT)
            ] or [StandardOutput.GLB]
            gen_result = backend.generate(
                GenerationRequest(
                    request_id=request.request_id,
                    input_image_path=str(state["current_image"]),
                    backend=request.backend,
                    output_types=backend_output_types,
                    output_dir=str(output_dir / "generated"),
                    remove_background=False,  # already done above
                    run_quality_check=False,
                    seed=request.seed,
                    device=request.device,
                    backend_params=request.backend_params,
                )
            )
            manifest.final_result = gen_result
            manifest.artifacts.extend(gen_result.artifacts)
            manifest.warnings.extend(gen_result.warnings)

            if not gen_result.success:
                raise RuntimeError(gen_result.error or "3D generation returned failure.")

            # Find the primary mesh artifact
            for artifact in gen_result.artifacts:
                if artifact.output_type in (
                    StandardOutput.GLB, StandardOutput.DRAFT_MESH,
                    StandardOutput.TEXTURED_MESH, StandardOutput.OBJ,
                ):
                    state["mesh_path"] = Path(artifact.path)
                    break

        run_stage(PipelineStage.GENERATION_3D, manifest, _generate_3d,
                  skip=PipelineStage.GENERATION_3D in self._skip_stages)
        save_manifest()

        # ── Stage 6: Mesh cleanup ─────────────────────────────────────────────
        def _mesh_cleanup():
            if state.get("mesh_path") is None:
                manifest.warnings.append("Mesh cleanup skipped: no mesh artifact from stage 5.")
                return
            from ai3d.postprocessors.mesh_cleaner import MeshCleaner
            mesh_path = state["mesh_path"]
            out_path = output_dir / "postprocessed" / ("cleaned" + mesh_path.suffix)
            result = MeshCleaner().clean(
                MeshCleanRequest(
                    input_path=str(mesh_path),
                    output_path=str(out_path),
                )
            )
            if result.success and result.output_path:
                state["cleaned_mesh_path"] = Path(result.output_path)
                manifest.artifacts.append(
                    ArtifactRef(path=result.output_path, kind="cleaned_mesh",
                                output_type=StandardOutput.CLEANED_MESH)
                )
            else:
                manifest.warnings.append(f"Mesh cleanup warning: {result.error}")
                state["cleaned_mesh_path"] = mesh_path  # use original

        run_stage(PipelineStage.MESH_CLEANUP, manifest, _mesh_cleanup,
                  skip=PipelineStage.MESH_CLEANUP in self._skip_stages)
        save_manifest()

        # ── Stage 7: UV / texture ─────────────────────────────────────────────
        def _uv_texture():
            src = state.get("cleaned_mesh_path") or state.get("mesh_path")
            if src is None:
                return
            from ai3d.postprocessors.uv_unwrapper import UVUnwrapper
            out_path = output_dir / "postprocessed" / "uv_unwrapped.glb"
            result = UVUnwrapper().process(src, out_path)
            if result.success and result.artifacts:
                state["uv_mesh_path"] = Path(result.artifacts[0].path)
                manifest.artifacts.extend(result.artifacts)
            else:
                manifest.warnings.append(f"UV unwrap warning: {result.error}")

        run_stage(PipelineStage.UV_TEXTURE, manifest, _uv_texture,
                  skip=PipelineStage.UV_TEXTURE in self._skip_stages)
        save_manifest()

        # ── Stage 8: Mesh-to-splat ───────────────────────────────────────────
        def _mesh_to_splat():
            src = state.get("uv_mesh_path") or state.get("cleaned_mesh_path") or state.get("mesh_path")
            if src is None:
                manifest.warnings.append("Mesh-to-splat skipped: no mesh artifact available.")
                return
            from ai3d.backends.mesh2splat.backend import Mesh2SplatBackend
            result = Mesh2SplatBackend().generate(
                GenerationRequest(
                    request_id=request.request_id,
                    input_image_path=str(src),
                    backend="mesh2splat",
                    output_types=[StandardOutput.SPLAT_PLY],
                    output_dir=str(output_dir / "splats"),
                    remove_background=False,
                    run_quality_check=False,
                    seed=request.seed,
                    device=request.device,
                    backend_params=request.backend_params,
                )
            )
            if result.success:
                manifest.artifacts.extend(result.artifacts)
            else:
                manifest.warnings.append(f"Mesh-to-splat warning: {result.error}")

        run_stage(
            PipelineStage.MESH_TO_SPLAT,
            manifest,
            _mesh_to_splat,
            skip=(
                PipelineStage.MESH_TO_SPLAT in self._skip_stages
                or StandardOutput.SPLAT_PLY not in request.output_types
            ),
        )
        save_manifest()

        # ── Stage 9: Blender render ───────────────────────────────────────────
        def _blender_render():
            asset = state.get("uv_mesh_path") or state.get("cleaned_mesh_path") or state.get("mesh_path")
            if asset is None:
                manifest.warnings.append("Blender render skipped: no mesh artifact available.")
                return
            from ai3d.blender.renderer import TurntableRenderer
            from ai3d.blender.bridge import BlenderBridge
            bridge = BlenderBridge()
            if not bridge.is_available():
                manifest.warnings.append("Blender not available; skipping turntable render.")
                return
            render_dir = ensure_directory(output_dir / "blender_render")
            result = TurntableRenderer(bridge).render_asset(
                asset_path=asset,
                output_dir=render_dir,
                passes=[BlenderRenderPass.RGB, BlenderRenderPass.DEPTH],
            )
            state["blender_result"] = result
            if result.success:
                for fp in result.frame_paths:
                    manifest.artifacts.append(ArtifactRef(path=fp, kind="render_frame"))
            else:
                manifest.warnings.append(f"Blender render warning: {result.error}")

        run_stage(PipelineStage.BLENDER_RENDER, manifest, _blender_render,
                  skip=self._skip_blender or PipelineStage.BLENDER_RENDER in self._skip_stages)
        save_manifest()

        # ── Stage 10: Video conditioning export ───────────────────────────────
        def _video_conditioning():
            asset = state.get("uv_mesh_path") or state.get("cleaned_mesh_path") or state.get("mesh_path")
            if asset is None:
                return
            from ai3d.video.turntable_exporter import TurntableExporter
            from ai3d.video.conditioning_pack import ConditioningPackBuilder
            pack_dir = ensure_directory(output_dir / "video_pack")
            pack = TurntableExporter().export(
                asset_path=asset,
                output_dir=pack_dir / "render",
                source_asset_id=request.request_id,
            )
            ConditioningPackBuilder().build(pack, pack_dir)
            state["video_pack"] = pack
            manifest.artifacts.append(
                ArtifactRef(path=str(pack_dir), kind="video_conditioning_pack")
            )

        run_stage(PipelineStage.VIDEO_CONDITIONING, manifest, _video_conditioning,
                  skip=self._skip_video_pack or PipelineStage.VIDEO_CONDITIONING in self._skip_stages)
        save_manifest()

        # ── Stage 11: Asset registration ─────────────────────────────────────
        def _register():
            registry = self._get_asset_registry()
            output_paths: dict = {}
            for artifact in manifest.artifacts:
                if artifact.output_type:
                    output_paths[artifact.output_type.value] = artifact.path

            entry = AssetRegistryEntry(
                asset_id=request.request_id,
                label=Path(request.input_image_path).stem,
                source_image_path=request.input_image_path,
                backend_used=request.backend,
                output_paths=output_paths,
                status=AssetStatus.COMPLETE if manifest.final_result and manifest.final_result.success
                       else AssetStatus.FAILED,
                pipeline_manifest_path=str(output_dir / "pipeline_manifest.yaml"),
            )
            registry.register(entry)

        run_stage(PipelineStage.ASSET_REGISTRATION, manifest, _register,
                  skip=self._skip_registration or PipelineStage.ASSET_REGISTRATION in self._skip_stages)
        save_manifest()

        return manifest

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_backend_registry(self):
        if self._backend_registry is None:
            from ai3d.backends.registry import get_default_registry
            self._backend_registry = get_default_registry()
        return self._backend_registry

    def _get_asset_registry(self):
        if self._asset_registry is None:
            from ai3d.registry.asset_registry import AssetRegistry
            self._asset_registry = AssetRegistry()
        return self._asset_registry
