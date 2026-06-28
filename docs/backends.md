# Backends Guide

## Overview

Each backend is wrapped behind `BaseBackend` and registered in `BackendRegistry`.
All backends return standardized `GenerationResult` and `ArtifactRef` objects.

## Decision Rules

| Situation | Recommended Backend |
|-----------|-------------------|
| Fast draft preview, batch processing | TripoSR |
| Final asset for Blender editing (needs real materials) | SF3D |
| Highest quality, 3DGS output | TRELLIS (M2+) |
| High quality + texture baking | Hunyuan3D-2 (M2+) |
| Ambiguous or incomplete source image | Wonder3D → InstantMesh (M3) |
| Low VRAM constraint (<10 GB) | TripoSR or CRM (M2+) |
| Web visualization / splat viewer | any backend → Mesh2Splat (M3) |

---

## TripoSR

- **Source**: stabilityai/TripoSR
- **Status**: Fully implemented (Milestone 1)
- **VRAM**: ~8 GB
- **Output**: Mesh + vertex colors (GLB, OBJ)
- **Strengths**: Very fast (~30s), good topology, reliable on clean product images
- **Limitations**: No UV/PBR textures (vertex color only); less detail than TRELLIS
- **Use for**: Batch draft generation, quick previews, concept validation

---

## Stable Fast 3D (SF3D)

- **Source**: stabilityai/stable-fast-3d
- **Status**: Fully implemented (Milestone 1)
- **VRAM**: ~6 GB
- **Output**: PBR-textured GLB (native UV + albedo/roughness/metallic maps)
- **Strengths**: Best-in-class textures for M1 backends; output is Blender-ready immediately
- **Limitations**: Input must be 512×512 RGBA; less topological quality than TRELLIS
- **Use for**: Final assets destined for Blender editing, rendering, or rigging

---

## TRELLIS (Scaffold — M2)

- **Source**: microsoft/TRELLIS
- **Status**: Scaffold stub
- **VRAM**: ~16 GB (TRELLIS), ~20 GB (TRELLIS.2)
- **Output**: 3DGS .ply + textured mesh
- **Strengths**: State-of-the-art quality; structured latent diffusion approach; native 3DGS output
- **Use for**: Final production assets; when quality is more important than speed

---

## Hunyuan3D-2 / 2.1 (Scaffold — M2)

- **Source**: tencent/Hunyuan3D-2
- **Status**: Scaffold stub
- **VRAM**: ~24 GB
- **Output**: Textured mesh (multi-stage: DiT → reconstruction → Hunyuan3D-Paint)
- **Strengths**: Excellent textures via Paint V2; handles complex geometry well
- **Use for**: High-quality final assets, especially characters and objects with fine detail

---

## InstantMesh (Scaffold — M2)

- **Source**: TencentARC/InstantMesh
- **Status**: Scaffold stub
- **VRAM**: ~23 GB
- **Output**: Textured mesh (Zero123++ multi-view → LRM reconstruction)
- **Use for**: Multi-view quality when TRELLIS is unavailable

---

## CRM (Scaffold — M2)

- **Source**: Zhengyi-Wang/CRM
- **Status**: Scaffold stub
- **VRAM**: ~8 GB
- **Output**: Textured mesh (RGBN-based CRM)
- **Use for**: Low-VRAM alternative to InstantMesh; hardware-constrained environments

---

## Wonder3D (Scaffold — M3)

- **Source**: xxlong0/Wonder3D
- **Status**: Scaffold stub
- **VRAM**: ~16 GB
- **Output**: 6 multi-view RGB images + 6 normal maps (NOT a direct mesh generator)
- **Use for**: Pre-processing ambiguous images; outputs feed downstream mesh reconstruction

---

## Mesh2Splat (Scaffold — M3)

- **Source**: mesh2splat/mesh2splat
- **Status**: Scaffold stub
- **VRAM**: ~4 GB
- **Input**: Cleaned mesh (GLB/OBJ) — NOT an image
- **Output**: 3DGS .ply / .ksplat
- **Use for**: Converting any generated mesh to 3D Gaussian Splat for web viewing or splat rendering experiments

---

## Output Type Reference

| `StandardOutput` | Format | Produced By |
|-----------------|--------|-------------|
| `draft_mesh` | GLB (no texture) | TripoSR |
| `cleaned_mesh` | GLB | MeshCleaner |
| `textured_mesh` | GLB (PBR) | SF3D, Hunyuan3D, TRELLIS |
| `glb` | .glb | All |
| `obj` | .obj | TripoSR, SF3D |
| `fbx` | .fbx | FormatConverter |
| `splat_ply` | .ply (3DGS) | TRELLIS, Mesh2Splat |
| `splat_ksplat` | .ksplat | Mesh2Splat |
| `multiview_images` | PNG × N | Wonder3D, InstantMesh |
| `normal_maps` | PNG × N | Wonder3D, CRM |
| `depth_maps` | PNG / EXR | Blender depth pass |
