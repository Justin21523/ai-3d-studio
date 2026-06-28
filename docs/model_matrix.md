# Model Matrix

Full capability and performance matrix for all supported backends.

> Note: Benchmark numbers (M1 column) are estimates based on published papers and community reports.
> Actual performance may vary by GPU and input complexity.
> M2/M3 numbers will be filled in when those backends are implemented.

## Capability Matrix

| Backend | Mesh | PBR Texture | 3DGS | Multiview | Normal | Depth | Status |
|---------|:----:|:-----------:|:----:|:---------:|:------:|:-----:|--------|
| TripoSR | ✓ | ✗ (vertex color) | ✗ | ✗ | ✗ | ✗ | M1 Full |
| SF3D | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | M1 Full |
| TRELLIS | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | M2 Scaffold |
| Hunyuan3D-2 | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | M2 Scaffold |
| InstantMesh | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | M2 Scaffold |
| CRM | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | M2 Scaffold |
| Wonder3D | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | M3 Scaffold |
| Mesh2Splat | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | M3 Scaffold |

## Performance Estimates (RTX 3090, single image)

| Backend | VRAM | Time | Output Size |
|---------|------|------|-------------|
| TripoSR | 8 GB | ~30s | 2–10 MB GLB |
| SF3D | 6 GB | ~20s | 5–25 MB GLB (with textures) |
| TRELLIS | 16 GB | ~120s | 20–50 MB (mesh + splat) |
| Hunyuan3D-2 | 24 GB | ~180s | 15–40 MB GLB |
| InstantMesh | 23 GB | ~90s | 10–30 MB GLB |
| CRM | 8 GB | ~45s | 5–20 MB GLB |
| Wonder3D | 16 GB | ~60s (diffusion only) | 6× PNG images |
| Mesh2Splat | 4 GB | ~30s | 10–100 MB .ply |

## Downstream Use Compatibility

| Backend Output | Blender Edit | Rigging | 3DGS Viewer | Video Conditioning |
|---------------|:------------:|:-------:|:-----------:|:-----------------:|
| Vertex-color mesh (TripoSR) | ✓ | ✓ | ✗ | ✓ (via Blender) |
| PBR mesh (SF3D) | ✓✓ | ✓✓ | ✗ | ✓✓ |
| Textured mesh (TRELLIS/Hunyuan3D) | ✓✓ | ✓✓ | ✗ | ✓✓ |
| 3DGS .ply (TRELLIS/Mesh2Splat) | ✗ | ✗ | ✓✓ | ✓ (Blender 3DGS addon) |
| Multiview images (Wonder3D) | ✗ | ✗ | ✗ | ✓ (direct frame use) |

## Notes

- **SF3D** is the recommended default for M1 workflows where PBR materials are needed.
- **TripoSR** is the recommended default for large-batch draft generation.
- **TRELLIS** (M2) is expected to become the recommended default for final production assets.
- **Wonder3D** outputs are useful as conditioning inputs for other models, not standalone 3D assets.
- Blender turntable renders work with any mesh output regardless of texture quality.
