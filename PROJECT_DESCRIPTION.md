# AI 3D Studio — Project Description

## Purpose

AI 3D Studio is a production-oriented Python framework that unifies multiple open-source image-to-3D generation models behind a single consistent interface. It bridges the gap between raw image inputs and video-ready 3D asset pipelines.

## Problem Solved

Running image-to-3D generation backends (TripoSR, SF3D, TRELLIS, Hunyuan3D, InstantMesh, CRM) requires per-project boilerplate: environment setup, model loading, format conversion, and downstream integration. AI 3D Studio provides one installable package that handles all of this, from preprocessing through Blender rendering to ComfyUI workflow orchestration.

## Architecture

```
Input Image
  → PreProcessor (background removal, quality check)
  → Backend Adapter (TripoSR / SF3D / TRELLIS / Hunyuan3D / InstantMesh / CRM)
  → PostProcessor (mesh clean, UV unwrap, format convert, texture bake)
  → Blender Bridge (headless turntable render, depth / normal / mask passes)
  → Video Conditioning Pack (rgb / depth / normal / mask frame sequences)
  → ComfyUI Client (workflow template fill, execution, result download)
  → Asset Registry (YAML-backed provenance tracking)
```

All stages are wired by an 11-stage `GenerationPipeline` that saves a `pipeline_manifest.yaml` after each stage for reproducibility and resume support.

## Design Principles

- **Consistent ABC interface** — every backend implements `BaseBackend` (`generate`, `check_availability`, `estimate_requirements`, `export_metadata`). New backends add one file.
- **Pydantic v2 throughout** — all data flows through typed models; YAML I/O via `write_model`/`read_model`.
- **No hardcoded paths** — all filesystem paths externalized to `configs/paths.yaml` → `PathConfig`.
- **Lazy model loading** — weights loaded on first inference call; availability checked without loading.
- **Subprocess isolation for Blender** — Blender runs in its own process; communication via JSON spec file + `AI3D_RESULT:` stdout marker.
- **Lightweight ComfyUI integration** — pure HTTP client; no ComfyUI Python package required.

## Target Use Cases

1. **Batch draft generation** — TripoSR over a directory of product images; export GLBs for review.
2. **Final production asset** — SF3D or Hunyuan3D for PBR-textured GLBs ready for Blender editing and rigging.
3. **Highest quality** — TRELLIS for simultaneous 3DGS + textured mesh output.
4. **Video conditioning** — turntable renders → depth/normal passes → CogVideoX / Wan2.1 / LTX-Video conditioning packs.
5. **ComfyUI video pipeline** — fill workflow templates, submit to running ComfyUI server, poll and download results.

## Milestones

| Milestone | Status | Deliverables |
|-----------|--------|-------------|
| M1 | Complete | TripoSR + SF3D full backends; Blender bridge; ComfyUI client; video pack; 11-stage pipeline; CLI; 70 tests |
| M2 | In Progress | TRELLIS, Hunyuan3D, InstantMesh, CRM full backends; UV texture baker; render passes |
| M3 | Planned | Wonder3D, Mesh2Splat; 3DGS pipeline; EXR depth; camera motion presets |
| M4 | Planned | FastAPI layer; async batch; benchmark docs |

## Technology Stack

| Layer | Libraries |
|-------|-----------|
| Models | Pydantic v2 |
| 3D processing | trimesh, open3d, xatlas |
| Image preprocessing | Pillow, rembg, opencv |
| ML backends | PyTorch (CUDA), diffusers |
| Blender | bpy (subprocess), headless render |
| HTTP | requests (ComfyUI client) |
| API (M4) | FastAPI, uvicorn |
| Testing | pytest, pytest-mock |
