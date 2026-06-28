# AI 3D Studio

A production-oriented, open-source system for image-to-3D generation, 3D asset post-processing, Blender integration, ComfyUI workflow orchestration, and video-generation downstream conditioning.

## Overview

AI 3D Studio unifies multiple state-of-the-art 3D generation backends behind a single consistent interface, with a full processing pipeline from raw image to video-ready conditioning pack.

```
Input Image
    │
    ├── Background removal (rembg)
    ├── Quality check
    │
    ▼
3D Generation Backend (TripoSR / SF3D / TRELLIS / Hunyuan3D / ...)
    │
    ├── Mesh cleanup (trimesh)
    ├── UV unwrapping (xatlas)
    ├── Format conversion (GLB / OBJ / FBX / PLY)
    │
    ▼
Blender (headless)
    ├── Turntable render (RGB)
    ├── Depth pass
    ├── Normal pass
    └── Object mask
    │
    ▼
Video Conditioning Pack
    ├── rgb/     ← turntable frames
    ├── depth/   ← normalized depth maps
    ├── normal/  ← world normals
    └── pack_manifest.yaml
    │
    ▼
ComfyUI Workflow (I2V / first-frame controlled video)
```

## Quick Start

```bash
# Install (editable)
pip install -e .
pip install -r requirements.txt

# Check your environment
python scripts/check_environment.py

# Download model weights
bash scripts/install_models.sh

# List available backends
ai3d list-backends

# Check a backend
ai3d check-backend --backend sf3d

# Generate from a single image
ai3d run-backend \
    --backend sf3d \
    --input /path/to/object.jpg \
    --output-dir /mnt/data/3d-studio/outputs/my_run \
    --output-type glb

# Run the full pipeline (11 stages)
ai3d run-pipeline \
    --input /path/to/object.jpg \
    --backend sf3d \
    --output-dir /mnt/data/3d-studio/outputs/full_run_001

# Batch generate
ai3d batch-generate \
    --backend triposr \
    --input-dir /path/to/images/ \
    --output-dir /mnt/data/3d-studio/outputs/batch_001

# Render turntable via Blender
ai3d blender-render \
    --asset /mnt/data/3d-studio/outputs/my_run/output.glb \
    --output-dir /mnt/data/3d-studio/outputs/my_run/turntable \
    --frames 72 --pass rgb --pass depth

# Export video conditioning pack
ai3d export-video-pack \
    --asset /mnt/data/3d-studio/outputs/my_run/output.glb \
    --output-dir /mnt/data/3d-studio/outputs/my_run/video_pack \
    --with-depth

# Run tests
pytest tests/ -v
```

## Supported Backends

| Backend | Status | VRAM | Output | Best For |
|---------|--------|------|--------|----------|
| **SF3D** (Stable Fast 3D) | Full (M1) | 6 GB | PBR GLB | Fast HQ preview, Blender editing |
| **TripoSR** | Full (M1) | 8 GB | Mesh + vertex color | Batch draft generation |
| **TRELLIS** | Scaffold (M2) | 16 GB | 3DGS + textured mesh | Highest quality |
| **Hunyuan3D-2** | Scaffold (M2) | 24 GB | Textured mesh | Multi-view quality |
| **InstantMesh** | Scaffold (M2) | 23 GB | Textured mesh | LRM sparse-view |
| **CRM** | Scaffold (M2) | 8 GB | Textured mesh | Low-VRAM alternative |
| **Wonder3D** | Scaffold (M3) | 16 GB | Multiview + normals | Ambiguous inputs |
| **Mesh2Splat** | Scaffold (M3) | 4 GB | 3DGS .ply | Web/splat visualization |

## Backend Selection Rules

- **Draft previews / batch**: TripoSR (fast, low VRAM)
- **Final asset for Blender**: SF3D (PBR materials) or Hunyuan3D (M2+)
- **Highest quality**: TRELLIS (M2+)
- **Ambiguous/incomplete source**: Wonder3D multi-view → downstream mesh
- **Web visualization**: Mesh2Splat → 3DGS .ply
- **Video conditioning prep**: Blender depth/normal passes on any mesh

## Directory Structure

```
ai3d/
├── core/          # Pydantic models, storage, paths, logging, config
├── backends/      # One adapter per 3D backend + BackendRegistry
├── preprocessors/ # Background removal, masking, quality checks
├── postprocessors/# Mesh clean, decimate, format convert, UV unwrap
├── blender/       # Headless Blender bridge, turntable renderer
├── comfyui/       # ComfyUI HTTP client, workflow manager, poller
├── video/         # Turntable exporter, depth sequence, conditioning pack
├── registry/      # Asset, model, workflow registries (YAML-backed)
├── pipeline/      # 11-stage orchestration, batch runner
├── cli/           # argparse CLI (all commands)
└── api/           # Optional FastAPI layer
configs/           # paths.yaml, models.yaml, workflows.yaml
workflows/         # ComfyUI JSON templates
blender_templates/ # .blend starter files
docs/              # Architecture, backend docs, integration guides
tests/             # pytest unit tests
scripts/           # Environment check, model install/link
```

## Path Configuration

Edit `configs/paths.yaml` to match your local environment:

```yaml
ai_models_root: /mnt/c/ai_models
outputs_root:   /mnt/data/3d-studio/outputs
blender_executable: /usr/bin/blender
comfyui_root:   /mnt/c/ai_tools/comfyui
comfyui_base_url: http://127.0.0.1:8188
```

## Milestones

- **M1 (current)**: SF3D + TripoSR fully implemented, Blender bridge, ComfyUI client, video pack
- **M2**: TRELLIS, Hunyuan3D, InstantMesh, CRM — full inference
- **M3**: Wonder3D, Mesh2Splat, full 3DGS pipeline, EXR depth sequences
- **M4**: FastAPI layer, async batch, full test coverage, benchmark docs

## Documentation

- [Architecture](docs/architecture.md)
- [Backends Guide](docs/backends.md)
- [Blender Integration](docs/blender_integration.md)
- [ComfyUI Workflows](docs/comfyui_workflows.md)
- [Video Pipeline](docs/video_pipeline.md)
- [Model Matrix](docs/model_matrix.md)
