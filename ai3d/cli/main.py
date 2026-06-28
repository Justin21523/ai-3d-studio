"""AI 3D Studio CLI — build_parser() + main() dispatch pattern.

Usage:
    python -m ai3d.cli.main <command> [options]
    ai3d <command> [options]          # via console_scripts entry point
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, List, Optional

from ai3d.core.logging import get_logger, setup_logging
from ai3d.core.models import (
    BatchManifest,
    BlenderRenderPass,
    DecimateRequest,
    FormatConvertRequest,
    GenerationRequest,
    MeshCleanRequest,
    StandardOutput,
)

_log = get_logger(__name__)


# ── Output helper ─────────────────────────────────────────────────────────────

def emit(payload: Any) -> int:
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def fail(message: str, code: int = 1) -> int:
    print(json.dumps({"error": message}, indent=2), file=sys.stderr)
    return code


# ── Parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai3d",
        description="AI 3D Studio — image-to-3D generation, Blender integration, video conditioning.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── Backend commands ──────────────────────────────────────────────────────
    p = sub.add_parser("list-backends", help="List all registered backends and their status")
    p.add_argument("--json", dest="json_output", action="store_true")

    p = sub.add_parser("check-backend", help="Check availability of a specific backend")
    p.add_argument("--backend", required=True)

    p = sub.add_parser("run-backend", help="Run a backend on a single input image")
    p.add_argument("--backend", required=True)
    p.add_argument("--input", required=True, metavar="PATH")
    p.add_argument("--output-dir", required=True, metavar="PATH")
    p.add_argument("--output-type", action="append", dest="output_types",
                   choices=[o.value for o in StandardOutput], metavar="TYPE")
    p.add_argument("--no-bg-removal", action="store_true")
    p.add_argument("--no-quality-check", action="store_true")
    p.add_argument("--seed", type=int)
    p.add_argument("--device", default="cuda")

    # ── Batch commands ────────────────────────────────────────────────────────
    p = sub.add_parser("batch-generate", help="Batch generate 3D assets from a directory of images")
    p.add_argument("--backend", required=True)
    p.add_argument("--input-dir", required=True, metavar="PATH")
    p.add_argument("--output-dir", required=True, metavar="PATH")
    p.add_argument("--output-type", action="append", dest="output_types",
                   choices=[o.value for o in StandardOutput], metavar="TYPE")
    p.add_argument("--limit", type=int)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--device", default="cuda")

    # ── Model management ──────────────────────────────────────────────────────
    p = sub.add_parser("install-models", help="Download model weights via huggingface-cli")
    p.add_argument("--backend", help="Install for one backend only (default: all)")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("link-models", help="Symlink models from a source root")
    p.add_argument("--source-root", required=True, metavar="PATH")
    p.add_argument("--dry-run", action="store_true")

    # ── Mesh postprocessing ───────────────────────────────────────────────────
    p = sub.add_parser("mesh-clean", help="Clean mesh geometry (degenerate faces, normals)")
    p.add_argument("--input", required=True, metavar="PATH")
    p.add_argument("--output", required=True, metavar="PATH")
    p.add_argument("--watertight", action="store_true")
    p.add_argument("--no-degenerate-faces", action="store_true")

    p = sub.add_parser("mesh-decimate", help="Reduce polygon count")
    p.add_argument("--input", required=True, metavar="PATH")
    p.add_argument("--output", required=True, metavar="PATH")
    p.add_argument("--face-count", type=int)
    p.add_argument("--ratio", type=float, metavar="0-1")

    p = sub.add_parser("mesh-convert", help="Convert mesh to another format")
    p.add_argument("--input", required=True, metavar="PATH")
    p.add_argument("--output", required=True, metavar="PATH")
    p.add_argument("--format", required=True, choices=["glb", "obj", "fbx", "ply", "stl"])

    p = sub.add_parser("mesh-uv-unwrap", help="Generate UV atlas using xatlas")
    p.add_argument("--input", required=True, metavar="PATH")
    p.add_argument("--output", required=True, metavar="PATH")

    p = sub.add_parser("mesh-to-splat", help="Convert mesh to 3D Gaussian Splat")
    p.add_argument("--input", required=True, metavar="PATH")
    p.add_argument("--output", required=True, metavar="PATH")
    p.add_argument("--format", choices=["ply", "ksplat"], default="ply")

    p = sub.add_parser("rig-model", help="Auto-rig a GLB/OBJ mesh with an installed rigging provider")
    p.add_argument("--input", required=True, metavar="PATH")
    p.add_argument("--output-dir", required=True, metavar="PATH")
    p.add_argument("--provider", default="riganything", choices=["riganything"])
    p.add_argument("--simplify", action="store_true")
    p.add_argument("--target-faces", type=int, default=8192)
    p.add_argument("--dry-run", action="store_true")

    # ── Blender commands ──────────────────────────────────────────────────────
    p = sub.add_parser("blender-check", help="Check Blender availability and version")

    p = sub.add_parser("blender-render", help="Render turntable animation via headless Blender")
    p.add_argument("--asset", required=True, metavar="PATH")
    p.add_argument("--output-dir", required=True, metavar="PATH")
    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--resolution", default="1024x1024",
                   help="WxH (default: 1024x1024)")
    p.add_argument("--elevation", type=float, default=20.0,
                   metavar="DEG", help="Camera elevation in degrees")
    p.add_argument("--pass", action="append", dest="passes",
                   choices=["rgb", "depth", "normal", "mask"],
                   metavar="PASS")
    p.add_argument("--hdri", metavar="PATH", help="HDRI environment map path")

    # ── Video conditioning ────────────────────────────────────────────────────
    p = sub.add_parser("export-video-pack",
                       help="Export video conditioning pack (turntable + depth/normal)")
    p.add_argument("--asset", required=True, metavar="PATH")
    p.add_argument("--output-dir", required=True, metavar="PATH")
    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--resolution", default="1024x1024")
    p.add_argument("--with-depth", action="store_true")
    p.add_argument("--with-normal", action="store_true")
    p.add_argument("--with-mask", action="store_true")

    # ── ComfyUI commands ──────────────────────────────────────────────────────
    p = sub.add_parser("comfyui-check", help="Check ComfyUI server connectivity")
    p.add_argument("--base-url", default="http://127.0.0.1:8188")

    p = sub.add_parser("list-workflows", help="List registered ComfyUI workflows")
    p.add_argument("--json", dest="json_output", action="store_true")

    p = sub.add_parser("run-workflow", help="Execute a ComfyUI workflow")
    p.add_argument("--workflow", required=True)
    p.add_argument("--input", required=True, metavar="PATH")
    p.add_argument("--output-dir", required=True, metavar="PATH")
    p.add_argument("--param", action="append", dest="params",
                   metavar="KEY=VALUE", help="Workflow placeholder override")
    p.add_argument("--base-url", default="http://127.0.0.1:8188")
    p.add_argument("--timeout", type=int, default=1800)

    p = sub.add_parser("sync-comfyui-workflows", help="Export enabled repo workflows into ComfyUI's visible workflow folder")
    p.add_argument("--workflow", help="Sync one workflow only")
    p.add_argument("--base-url", default="http://127.0.0.1:8188")
    p.add_argument("--output-dir", metavar="PATH", help="ComfyUI workflows folder override")

    p = sub.add_parser("list-3d-models", help="List high-level 3D model routes")
    p.add_argument("--json", dest="json_output", action="store_true")

    p = sub.add_parser("run-3d", help="Generate a 3D model from an image with a simple model selector")
    p.add_argument("--input", required=True, metavar="PATH")
    p.add_argument("--output-dir", required=True, metavar="PATH")
    p.add_argument("--model", default="hunyuan3d21",
                   help="hunyuan3d21, triposr, sf3d, trellis, or hunyuan3d")
    p.add_argument("--splat", action="store_true", help="Also export a 3DGS PLY splat")
    p.add_argument("--turntable", action="store_true", help="Also render Blender turntable frames")
    p.add_argument("--frames", type=int, default=72)
    p.add_argument("--resolution", default="1024x1024")
    p.add_argument("--no-clean", action="store_true")
    p.add_argument("--no-uv", action="store_true")
    p.add_argument("--no-register", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--latent-resolution", type=int, default=4096)
    p.add_argument("--octree-resolution", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--base-url", default="http://127.0.0.1:8188")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--device", default="cuda")

    # ── Registry commands ─────────────────────────────────────────────────────
    p = sub.add_parser("list-assets", help="List registered 3D assets")
    p.add_argument("--backend")
    p.add_argument("--tag")
    p.add_argument("--json", dest="json_output", action="store_true")

    p = sub.add_parser("show-asset", help="Show details for one registered asset")
    p.add_argument("--asset-id", required=True)

    p = sub.add_parser("register-asset", help="Manually register a 3D asset")
    p.add_argument("--asset-id", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--source-image", required=True, metavar="PATH")
    p.add_argument("--backend", required=True)
    p.add_argument("--output-path", action="append", dest="output_paths",
                   metavar="TYPE=PATH")
    p.add_argument("--tag", action="append", dest="tags")

    p = sub.add_parser("delete-asset", help="Remove a registered asset from the registry")
    p.add_argument("--asset-id", required=True)
    p.add_argument("--confirm", action="store_true")

    # ── Pipeline commands ─────────────────────────────────────────────────────
    p = sub.add_parser("run-pipeline", help="Run the full 11-stage generation pipeline")
    p.add_argument("--input", required=True, metavar="PATH")
    p.add_argument("--backend", required=True)
    p.add_argument("--output-dir", required=True, metavar="PATH")
    p.add_argument("--output-type", action="append", dest="output_types",
                   choices=[o.value for o in StandardOutput], metavar="TYPE")
    p.add_argument("--skip-stage", action="append", dest="skip_stages",
                   choices=_SKIPPABLE_STAGES, metavar="STAGE")
    p.add_argument("--no-blender", action="store_true")
    p.add_argument("--no-video-pack", action="store_true")
    p.add_argument("--no-register", action="store_true")
    p.add_argument("--seed", type=int)
    p.add_argument("--device", default="cuda")

    p = sub.add_parser("run-batch-pipeline",
                       help="Run the full pipeline on all images in a directory")
    p.add_argument("--input-dir", required=True, metavar="PATH")
    p.add_argument("--backend", required=True)
    p.add_argument("--output-root", required=True, metavar="PATH")
    p.add_argument("--output-type", action="append", dest="output_types",
                   choices=[o.value for o in StandardOutput], metavar="TYPE")
    p.add_argument("--limit", type=int)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--no-blender", action="store_true")
    p.add_argument("--no-video-pack", action="store_true")

    p = sub.add_parser("show-pipeline", help="Show a pipeline manifest by run-id or path")
    p.add_argument("--run-id")
    p.add_argument("--manifest-path", metavar="PATH")

    return parser


_SKIPPABLE_STAGES = [
    "input_ingest", "background_removal", "quality_check",
    "generation_3d", "mesh_cleanup", "uv_texture",
    "blender_render", "video_conditioning", "asset_registration",
]


_COMFYUI_3D_MODELS = {
    "hunyuan3d21": "hunyuan3d21_image_to_model",
    "hunyuan3d2.1": "hunyuan3d21_image_to_model",
    "hunyuan3d-2.1": "hunyuan3d21_image_to_model",
    "comfyui_hunyuan3d21": "hunyuan3d21_image_to_model",
}


def _parse_params(params: Optional[List[str]]) -> dict:
    replacements = {}
    for param in (params or []):
        if "=" not in param:
            continue
        key, _, val = param.partition("=")
        if val.isdigit():
            replacements[key.upper()] = int(val)
            continue
        try:
            replacements[key.upper()] = float(val)
        except ValueError:
            replacements[key.upper()] = val
    return replacements


def _default_workflow_replacements(input_path: Path, workflow: str) -> dict:
    return {
        "INPUT": str(input_path),
        "INPUT_PATH": str(input_path),
        "INPUT_IMAGE": input_path.name,
        "CKPT_NAME": "hunyuan_3d_v2.1.safetensors",
        "OUTPUT_PREFIX": f"ai3d/{workflow}",
        "SEED": 1,
        "STEPS": 30,
        "CFG": 5.0,
        "LATENT_RESOLUTION": 4096,
        "OCTREE_RESOLUTION": 256,
        "THRESHOLD": 0.6,
    }


def _comfyui_workflows_dir() -> Path:
    from ai3d.core.paths import comfyui_root
    return comfyui_root() / "user" / "default" / "workflows"


def _standard_3d_model_routes() -> list[dict]:
    return [
        {
            "name": "hunyuan3d21",
            "engine": "comfyui",
            "workflow": "hunyuan3d21_image_to_model",
            "outputs": ["glb", "cleaned_mesh", "textured_mesh", "splat_ply"],
        },
        {
            "name": "triposr",
            "engine": "backend",
            "backend": "triposr",
            "outputs": ["glb", "cleaned_mesh", "textured_mesh", "splat_ply"],
        },
        {
            "name": "sf3d",
            "engine": "backend",
            "backend": "sf3d",
            "outputs": ["glb", "cleaned_mesh", "textured_mesh", "splat_ply"],
        },
        {
            "name": "trellis",
            "engine": "backend",
            "backend": "trellis",
            "outputs": ["glb", "splat_ply"],
            "status": "requires local TRELLIS install and weights",
        },
        {
            "name": "hunyuan3d",
            "engine": "backend",
            "backend": "hunyuan3d",
            "outputs": ["glb"],
            "status": "native backend path, separate from ComfyUI Hunyuan3D 2.1",
        },
    ]


# ── Command handlers ──────────────────────────────────────────────────────────

def _handle_list_backends(args) -> int:
    from ai3d.backends.registry import get_default_registry
    registry = get_default_registry()
    capabilities = registry.capabilities()
    return emit(capabilities)


def _handle_check_backend(args) -> int:
    from ai3d.backends.registry import get_default_registry
    registry = get_default_registry()
    try:
        backend = registry.get(args.backend)
    except KeyError as exc:
        return fail(str(exc))
    avail = backend.check_availability()
    reqs = backend.estimate_requirements()
    return emit({
        "availability": avail.model_dump(mode="json"),
        "requirements": reqs.model_dump(mode="json"),
    })


def _handle_run_backend(args) -> int:
    from ai3d.backends.registry import get_default_registry
    registry = get_default_registry()
    try:
        backend = registry.get(args.backend)
    except KeyError as exc:
        return fail(str(exc))

    output_types = [StandardOutput(t) for t in (args.output_types or ["glb"])]
    request = GenerationRequest(
        request_id=str(uuid.uuid4())[:8],
        input_image_path=args.input,
        backend=args.backend,
        output_types=output_types,
        output_dir=args.output_dir,
        remove_background=not args.no_bg_removal,
        run_quality_check=not args.no_quality_check,
        seed=args.seed,
        device=args.device,
    )
    result = backend.generate(request)
    return emit(result.model_dump(mode="json"))


def _handle_batch_generate(args) -> int:
    from ai3d.pipeline.batch_runner import BatchRunner
    from ai3d.pipeline.generation_pipeline import GenerationPipeline

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        return fail(f"Input directory not found: {input_dir}")

    images = BatchRunner.collect_images(input_dir, limit=args.limit)
    if not images:
        return fail(f"No images found in: {input_dir}")

    output_types = [StandardOutput(t) for t in (args.output_types or ["glb"])]
    batch = BatchManifest(
        batch_id=str(uuid.uuid4())[:8],
        input_images=images,
        backend=args.backend,
        output_types=output_types,
        output_root=args.output_dir,
    )
    pipeline = GenerationPipeline(skip_blender=True, skip_video_pack=True)
    result = BatchRunner(pipeline).run_batch(batch)
    return emit(result.model_dump(mode="json"))


def _handle_install_models(args) -> int:
    from ai3d.registry.model_registry import ModelRegistry
    registry = ModelRegistry()
    entries = registry.list_enabled()
    if args.backend:
        entries = [e for e in entries if e.backend_adapter == args.backend]

    results = []
    for entry in entries:
        path = Path(entry.local_path)
        if path.exists() and not args.force:
            results.append({"model": entry.name, "status": "already_present", "path": str(path)})
            continue
        # Provide the download command rather than running it automatically
        notes = entry.notes
        download_cmd = next((n for n in notes if n.startswith("Download:")), None)
        results.append({
            "model": entry.name,
            "status": "download_required",
            "path": str(path),
            "download_cmd": download_cmd or f"huggingface-cli download {entry.source_repo} --local-dir {path}",
        })
    return emit(results)


def _handle_link_models(args) -> int:
    from ai3d.registry.model_registry import ModelRegistry
    source_root = Path(args.source_root)
    registry = ModelRegistry()
    results = []
    for entry in registry.list_enabled():
        target = Path(entry.local_path)
        source_name = target.name
        source = source_root / source_name
        if not source.exists():
            results.append({"model": entry.name, "status": "source_not_found", "source": str(source)})
            continue
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(source)
        results.append({
            "model": entry.name,
            "status": "dry_run" if args.dry_run else "linked",
            "source": str(source),
            "target": str(target),
        })
    return emit(results)


def _handle_mesh_clean(args) -> int:
    from ai3d.postprocessors.mesh_cleaner import MeshCleaner
    result = MeshCleaner().clean(MeshCleanRequest(
        input_path=args.input,
        output_path=args.output,
        remove_degenerate_faces=not args.no_degenerate_faces,
        make_watertight=args.watertight,
    ))
    return emit(result.model_dump(mode="json"))


def _handle_mesh_decimate(args) -> int:
    from ai3d.postprocessors.decimator import MeshDecimator
    result = MeshDecimator().decimate(DecimateRequest(
        input_path=args.input,
        output_path=args.output,
        target_face_count=args.face_count,
        target_ratio=args.ratio,
    ))
    return emit(result.model_dump(mode="json"))


def _handle_mesh_convert(args) -> int:
    from ai3d.postprocessors.format_converter import FormatConverter
    result = FormatConverter().convert(FormatConvertRequest(
        input_path=args.input,
        output_path=args.output,
        output_format=args.format,
    ))
    return emit(result.model_dump(mode="json"))


def _handle_mesh_uv_unwrap(args) -> int:
    from ai3d.postprocessors.uv_unwrapper import UVUnwrapper
    result = UVUnwrapper().process(Path(args.input), Path(args.output))
    return emit(result.model_dump(mode="json"))


def _handle_mesh_to_splat(args) -> int:
    from ai3d.backends.mesh2splat.backend import Mesh2SplatBackend
    output_path = Path(args.output)
    result = Mesh2SplatBackend().generate(
        GenerationRequest(
            request_id=output_path.stem or str(uuid.uuid4())[:8],
            input_image_path=args.input,
            backend="mesh2splat",
            output_types=[StandardOutput.SPLAT_PLY],
            output_dir=str(output_path.parent),
        )
    )
    return emit(result.model_dump(mode="json"))


def _handle_rig_model(args) -> int:
    import shutil
    import subprocess
    from ai3d.core.paths import ai_tools_root

    input_path = Path(args.input)
    if not input_path.exists():
        return fail(f"Input mesh not found: {input_path}")

    source_root = ai_tools_root() / "RigAnything"
    ckpt_path = source_root / "ckpt" / "riganything_ckpt.pt"
    if not source_root.exists():
        return fail(f"RigAnything source not found: {source_root}")
    if not ckpt_path.exists():
        return fail(f"RigAnything checkpoint not found: {ckpt_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash",
        "scripts/inference.sh",
        str(input_path),
        "1" if args.simplify else "0",
        str(args.target_faces),
    ]
    if args.dry_run:
        return emit({
            "provider": args.provider,
            "source_root": str(source_root),
            "command": cmd,
            "cwd": str(source_root),
            "expected_output_root": str(source_root / "outputs" / input_path.stem),
        })

    proc = subprocess.run(cmd, cwd=source_root, text=True, capture_output=True, timeout=3600)
    expected = source_root / "outputs" / input_path.stem / f"{input_path.stem}_simplified_rig.glb"
    copied = None
    if expected.exists():
        copied = output_dir / expected.name
        shutil.copy2(expected, copied)
    return emit({
        "success": proc.returncode == 0 and copied is not None,
        "provider": args.provider,
        "source_output": str(expected),
        "output": str(copied) if copied else None,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    })


def _handle_blender_check(args) -> int:
    from ai3d.blender.bridge import BlenderBridge
    bridge = BlenderBridge()
    return emit({
        "available": bridge.is_available(),
        "version": bridge.get_version(),
        "executable": str(bridge._blender_exe),
    })


def _handle_blender_render(args) -> int:
    from ai3d.blender.renderer import TurntableRenderer
    res_parts = args.resolution.split("x")
    res_x, res_y = int(res_parts[0]), int(res_parts[1])
    passes = [BlenderRenderPass(p) for p in (args.passes or ["rgb"])]
    from ai3d.core.models import TurntableSpec
    spec = TurntableSpec(
        asset_path=args.asset,
        output_dir=args.output_dir,
        frame_count=args.frames,
        fps=args.fps,
        resolution_x=res_x,
        resolution_y=res_y,
        camera_elevation_deg=args.elevation,
        render_passes=passes,
        hdri_path=args.hdri,
    )
    result = TurntableRenderer().render(spec)
    return emit(result.model_dump(mode="json"))


def _handle_export_video_pack(args) -> int:
    from ai3d.video.turntable_exporter import TurntableExporter
    from ai3d.video.conditioning_pack import ConditioningPackBuilder
    res_parts = args.resolution.split("x")
    res_x, res_y = int(res_parts[0]), int(res_parts[1])
    passes = [BlenderRenderPass.RGB]
    if args.with_depth:
        passes.append(BlenderRenderPass.DEPTH)
    if args.with_normal:
        passes.append(BlenderRenderPass.NORMAL)
    if args.with_mask:
        passes.append(BlenderRenderPass.MASK)

    output_dir = Path(args.output_dir)
    pack = TurntableExporter().export(
        asset_path=Path(args.asset),
        output_dir=output_dir / "render",
        frame_count=args.frames,
        fps=args.fps,
        resolution=(res_x, res_y),
        passes=passes,
    )
    ConditioningPackBuilder().build(pack, output_dir)
    return emit(pack.model_dump(mode="json"))


def _handle_comfyui_check(args) -> int:
    from ai3d.comfyui.client import ComfyUIClient
    client = ComfyUIClient(base_url=args.base_url)
    available, reason = client.health_check()
    result: dict = {"available": available, "base_url": args.base_url}
    if reason:
        result["reason"] = reason
    if available:
        try:
            result["server_stats"] = client.get_server_stats()
        except Exception:
            pass
    return emit(result)


def _handle_list_workflows(args) -> int:
    from ai3d.registry.workflow_registry import WorkflowRegistry
    entries = WorkflowRegistry().list()
    return emit([e.model_dump(mode="json") for e in entries])


def _handle_run_workflow(args) -> int:
    import shutil
    from ai3d.comfyui.client import ComfyUIClient
    from ai3d.comfyui.workflow_manager import WorkflowManager
    from ai3d.comfyui.result_poller import ResultPoller
    from ai3d.registry.workflow_registry import WorkflowRegistry
    from ai3d.core.paths import REPO_ROOT, comfyui_input_dir

    try:
        entry = WorkflowRegistry().get(args.workflow)
    except KeyError as exc:
        return fail(str(exc))

    if not entry.enabled:
        return fail(f"Workflow '{args.workflow}' is disabled (scaffold stub).")

    template_path = REPO_ROOT / entry.template_path
    input_path = Path(args.input)
    replacements = _default_workflow_replacements(input_path, args.workflow)
    if input_path.exists() and input_path.is_file():
        input_dir = comfyui_input_dir()
        input_dir.mkdir(parents=True, exist_ok=True)
        comfy_input = input_dir / input_path.name
        if input_path.resolve() != comfy_input.resolve():
            shutil.copy2(input_path, comfy_input)
        replacements["INPUT_IMAGE"] = input_path.name
    replacements.update(_parse_params(args.params))

    client = ComfyUIClient(base_url=args.base_url)
    available, reason = client.health_check()
    if not available:
        return fail(f"ComfyUI server not reachable: {reason}")

    wm = WorkflowManager()
    try:
        filled = wm.prepare(template_path, replacements)
    except Exception as exc:
        return fail(f"Failed to prepare workflow: {exc}")

    prompt_id = client.queue_prompt(filled)
    result = ResultPoller(client).wait_and_download(
        prompt_id,
        output_dir=Path(args.output_dir),
        timeout=args.timeout,
    )
    return emit(result.model_dump(mode="json"))


def _handle_sync_comfyui_workflows(args) -> int:
    from ai3d.comfyui.client import ComfyUIClient
    from ai3d.comfyui.workflow_exporter import export_ui_workflow
    from ai3d.comfyui.workflow_manager import WorkflowManager
    from ai3d.registry.workflow_registry import WorkflowRegistry
    from ai3d.core.paths import REPO_ROOT

    registry = WorkflowRegistry()
    entries = [registry.get(args.workflow)] if args.workflow else registry.list_enabled()
    output_dir = Path(args.output_dir) if args.output_dir else _comfyui_workflows_dir()

    object_info = {}
    client = ComfyUIClient(base_url=args.base_url)
    available, _reason = client.health_check()
    if available:
        try:
            object_info = client.get_object_info()
        except Exception:
            object_info = {}

    wm = WorkflowManager()
    synced = []
    for entry in entries:
        if not entry.enabled:
            continue
        template_path = REPO_ROOT / entry.template_path
        replacements = _default_workflow_replacements(Path("ai3d_example_input.png"), entry.name)
        graph = wm.prepare(template_path, replacements)
        out_path = output_dir / f"AI3D - {entry.name}.json"
        export_ui_workflow(graph, out_path, object_info=object_info)
        synced.append({
            "workflow": entry.name,
            "path": str(out_path),
            "ui_format": True,
        })

    return emit({
        "success": bool(synced),
        "output_dir": str(output_dir),
        "workflows": synced,
        "restart_or_reload_comfyui": True,
    })


def _handle_list_3d_models(args) -> int:
    return emit(_standard_3d_model_routes())


def _run_comfyui_3d(args, workflow_name: str) -> dict:
    import shutil
    from ai3d.comfyui.client import ComfyUIClient
    from ai3d.comfyui.result_poller import ResultPoller
    from ai3d.comfyui.workflow_manager import WorkflowManager
    from ai3d.core.paths import REPO_ROOT, comfyui_input_dir
    from ai3d.registry.workflow_registry import WorkflowRegistry

    run_id = str(uuid.uuid4())[:8]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    input_dir = comfyui_input_dir()
    input_dir.mkdir(parents=True, exist_ok=True)
    comfy_input = input_dir / input_path.name
    if input_path.resolve() != comfy_input.resolve():
        shutil.copy2(input_path, comfy_input)

    entry = WorkflowRegistry().get(workflow_name)
    replacements = _default_workflow_replacements(input_path, workflow_name)
    replacements.update({
        "INPUT_IMAGE": input_path.name,
        "OUTPUT_PREFIX": f"ai3d/{run_id}/generated",
        "SEED": args.seed,
        "STEPS": args.steps,
        "CFG": args.cfg,
        "LATENT_RESOLUTION": args.latent_resolution,
        "OCTREE_RESOLUTION": args.octree_resolution,
        "THRESHOLD": args.threshold,
    })

    client = ComfyUIClient(base_url=args.base_url)
    available, reason = client.health_check()
    if not available:
        raise RuntimeError(f"ComfyUI server not reachable: {reason}")

    graph = WorkflowManager().prepare(REPO_ROOT / entry.template_path, replacements)
    prompt_id = client.queue_prompt(graph)
    comfy_result = ResultPoller(client).wait_and_download(
        prompt_id,
        output_dir=output_dir / "generated",
        timeout=args.timeout,
    )
    if not comfy_result.success:
        raise RuntimeError(comfy_result.error or "ComfyUI 3D workflow returned no output.")

    artifacts = comfy_result.output_files[:]
    warnings = comfy_result.warnings[:]
    mesh_artifact = next(
        (a for a in artifacts if Path(a.path).suffix.lower() in (".glb", ".gltf", ".obj", ".fbx")),
        None,
    )
    if mesh_artifact is None:
        raise RuntimeError("ComfyUI workflow completed but did not return a mesh file.")

    mesh_path = Path(mesh_artifact.path)
    final_mesh = mesh_path
    if not args.no_clean:
        from ai3d.postprocessors.mesh_cleaner import MeshCleaner
        clean_path = output_dir / "postprocessed" / f"cleaned{mesh_path.suffix}"
        result = MeshCleaner().process(mesh_path, clean_path)
        if result.success and result.artifacts:
            final_mesh = Path(result.artifacts[0].path)
            artifacts.extend(result.artifacts)
        else:
            warnings.append(f"Mesh cleanup warning: {result.error}")

    if not args.no_uv:
        from ai3d.postprocessors.uv_unwrapper import UVUnwrapper
        uv_path = output_dir / "postprocessed" / "uv_unwrapped.glb"
        result = UVUnwrapper().process(final_mesh, uv_path)
        if result.success and result.artifacts:
            final_mesh = Path(result.artifacts[0].path)
            artifacts.extend(result.artifacts)
        else:
            warnings.append(f"UV unwrap warning: {result.error}")

    if args.splat:
        from ai3d.backends.mesh2splat.backend import Mesh2SplatBackend
        result = Mesh2SplatBackend().generate(GenerationRequest(
            request_id=run_id,
            input_image_path=str(final_mesh),
            backend="mesh2splat",
            output_types=[StandardOutput.SPLAT_PLY],
            output_dir=str(output_dir / "splats"),
            remove_background=False,
            run_quality_check=False,
            seed=args.seed,
            device=args.device,
        ))
        if result.success:
            artifacts.extend(result.artifacts)
        else:
            warnings.append(f"Mesh-to-splat warning: {result.error}")

    turntable = None
    if args.turntable:
        from ai3d.blender.bridge import BlenderBridge
        from ai3d.blender.renderer import TurntableRenderer
        res_x, res_y = [int(part) for part in args.resolution.split("x")]
        bridge = BlenderBridge()
        if bridge.is_available():
            render = TurntableRenderer(bridge).render_asset(
                asset_path=final_mesh,
                output_dir=output_dir / "turntable",
                frame_count=args.frames,
                resolution=(res_x, res_y),
            )
            turntable = render.model_dump(mode="json")
            if not render.success:
                warnings.append(f"Turntable render warning: {render.error}")
        else:
            warnings.append("Blender not available; turntable skipped.")

    return {
        "success": True,
        "run_id": run_id,
        "model": args.model,
        "engine": "comfyui",
        "workflow": workflow_name,
        "prompt_id": comfy_result.prompt_id,
        "input": str(input_path),
        "final_mesh": str(final_mesh),
        "artifacts": [a.model_dump(mode="json") for a in artifacts],
        "warnings": warnings,
        "turntable": turntable,
    }


def _run_backend_3d(args) -> dict:
    from ai3d.pipeline.generation_pipeline import GenerationPipeline
    from ai3d.core.models import PipelineStage

    output_types = [StandardOutput.GLB]
    if args.splat:
        output_types.append(StandardOutput.SPLAT_PLY)
    skip_stages = []
    if args.no_clean:
        skip_stages.append(PipelineStage.MESH_CLEANUP)
    if args.no_uv:
        skip_stages.append(PipelineStage.UV_TEXTURE)

    request = GenerationRequest(
        request_id=str(uuid.uuid4())[:8],
        input_image_path=args.input,
        backend=args.model,
        output_types=output_types,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
    )
    manifest = GenerationPipeline(
        skip_stages=skip_stages,
        skip_blender=not args.turntable,
        skip_video_pack=True,
        skip_registration=args.no_register,
    ).run(request)
    return manifest.model_dump(mode="json")


def _handle_run_3d(args) -> int:
    model_key = args.model.lower().replace("_", "-")
    comfy_key = args.model.lower().replace("-", "").replace("_", "")
    try:
        comfy_aliases = {k.replace("-", "").replace("_", ""): v for k, v in _COMFYUI_3D_MODELS.items()}
        if comfy_key in comfy_aliases:
            workflow_name = comfy_aliases[comfy_key]
            return emit(_run_comfyui_3d(args, workflow_name))
        args.model = model_key
        return emit(_run_backend_3d(args))
    except Exception as exc:
        return fail(str(exc))


def _handle_list_assets(args) -> int:
    from ai3d.registry.asset_registry import AssetRegistry
    registry = AssetRegistry()
    entries = registry.list()
    if args.backend:
        entries = [e for e in entries if e.backend_used == args.backend]
    if args.tag:
        entries = [e for e in entries if args.tag in e.tags]
    return emit([e.model_dump(mode="json") for e in entries])


def _handle_show_asset(args) -> int:
    from ai3d.registry.asset_registry import AssetRegistry
    try:
        entry = AssetRegistry().get(args.asset_id)
        return emit(entry.model_dump(mode="json"))
    except KeyError as exc:
        return fail(str(exc))


def _handle_register_asset(args) -> int:
    from ai3d.registry.asset_registry import AssetRegistry
    from ai3d.core.models import AssetRegistryEntry

    output_paths: dict = {}
    for op in (args.output_paths or []):
        if "=" in op:
            t, _, p = op.partition("=")
            output_paths[t] = p

    entry = AssetRegistryEntry(
        asset_id=args.asset_id,
        label=args.label,
        source_image_path=args.source_image,
        backend_used=args.backend,
        output_paths=output_paths,
        tags=args.tags or [],
    )
    path = AssetRegistry().register(entry)
    return emit({"registered": True, "path": str(path), **entry.model_dump(mode="json")})


def _handle_delete_asset(args) -> int:
    if not args.confirm:
        return fail("Use --confirm to delete an asset from the registry.")
    from ai3d.registry.asset_registry import AssetRegistry
    deleted = AssetRegistry().delete(args.asset_id)
    return emit({"deleted": deleted, "asset_id": args.asset_id})


def _handle_run_pipeline(args) -> int:
    from ai3d.pipeline.generation_pipeline import GenerationPipeline
    from ai3d.core.models import PipelineStage

    output_types = [StandardOutput(t) for t in (args.output_types or ["glb"])]
    skip_stages = [PipelineStage(s) for s in (args.skip_stages or [])]

    request = GenerationRequest(
        request_id=str(uuid.uuid4())[:8],
        input_image_path=args.input,
        backend=args.backend,
        output_types=output_types,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
    )
    pipeline = GenerationPipeline(
        skip_stages=skip_stages,
        skip_blender=args.no_blender,
        skip_video_pack=args.no_video_pack,
        skip_registration=args.no_register,
    )
    manifest = pipeline.run(request)
    return emit(manifest.model_dump(mode="json"))


def _handle_run_batch_pipeline(args) -> int:
    from ai3d.pipeline.batch_runner import BatchRunner
    from ai3d.pipeline.generation_pipeline import GenerationPipeline

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        return fail(f"Input directory not found: {input_dir}")

    images = BatchRunner.collect_images(input_dir, limit=args.limit)
    if not images:
        return fail(f"No images found in: {input_dir}")

    output_types = [StandardOutput(t) for t in (args.output_types or ["glb"])]
    batch = BatchManifest(
        batch_id=str(uuid.uuid4())[:8],
        input_images=images,
        backend=args.backend,
        output_types=output_types,
        output_root=args.output_root,
    )
    pipeline = GenerationPipeline(
        skip_blender=args.no_blender,
        skip_video_pack=args.no_video_pack,
    )
    result = BatchRunner(pipeline).run_batch(batch)
    return emit(result.model_dump(mode="json"))


def _handle_show_pipeline(args) -> int:
    from ai3d.core.models import PipelineManifest
    from ai3d.core.storage import read_model

    if args.manifest_path:
        path = Path(args.manifest_path)
    elif args.run_id:
        from ai3d.core.paths import outputs_root
        path = outputs_root() / args.run_id / "pipeline_manifest.yaml"
    else:
        return fail("Provide --run-id or --manifest-path")

    if not path.exists():
        return fail(f"Manifest not found: {path}")

    manifest = read_model(path, PipelineManifest)
    return emit(manifest.model_dump(mode="json"))


# ── Dispatch ──────────────────────────────────────────────────────────────────

_HANDLERS = {
    "list-backends": _handle_list_backends,
    "check-backend": _handle_check_backend,
    "run-backend": _handle_run_backend,
    "batch-generate": _handle_batch_generate,
    "install-models": _handle_install_models,
    "link-models": _handle_link_models,
    "mesh-clean": _handle_mesh_clean,
    "mesh-decimate": _handle_mesh_decimate,
    "mesh-convert": _handle_mesh_convert,
    "mesh-uv-unwrap": _handle_mesh_uv_unwrap,
    "mesh-to-splat": _handle_mesh_to_splat,
    "rig-model": _handle_rig_model,
    "blender-check": _handle_blender_check,
    "blender-render": _handle_blender_render,
    "export-video-pack": _handle_export_video_pack,
    "comfyui-check": _handle_comfyui_check,
    "list-workflows": _handle_list_workflows,
    "run-workflow": _handle_run_workflow,
    "sync-comfyui-workflows": _handle_sync_comfyui_workflows,
    "list-3d-models": _handle_list_3d_models,
    "run-3d": _handle_run_3d,
    "list-assets": _handle_list_assets,
    "show-asset": _handle_show_asset,
    "register-asset": _handle_register_asset,
    "delete-asset": _handle_delete_asset,
    "run-pipeline": _handle_run_pipeline,
    "run-batch-pipeline": _handle_run_batch_pipeline,
    "show-pipeline": _handle_show_pipeline,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(level=args.log_level)

    handler = _HANDLERS.get(args.command)
    if handler is None:
        return fail(f"Unknown command: {args.command}")

    try:
        return handler(args)
    except SystemExit:
        raise
    except Exception as exc:
        _log.exception("Command '%s' raised an unexpected error", args.command)
        return fail(str(exc))


if __name__ == "__main__":
    sys.exit(main())
