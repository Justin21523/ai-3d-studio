# Video Generation Downstream Pipeline

## Overview

AI 3D Studio prepares assets and conditioning data for downstream video generation models
(e.g. CogVideoX, Wan2.1, LTX-Video, SVD, etc.).

A **video conditioning pack** bundles:
- `rgb/` — turntable frame sequence (PNG)
- `depth/` — normalized depth maps (PNG)
- `normal/` — world-space normal maps (PNG) *(M2+)*
- `mask/` — object mask sequence (PNG) *(M2+)*
- `pack_manifest.yaml` — metadata

## Quick Start

```bash
# Export conditioning pack for an existing GLB
ai3d export-video-pack \
    --asset /path/to/output.glb \
    --output-dir /path/to/video_pack \
    --frames 72 \
    --fps 24 \
    --with-depth

# The pipeline handles this automatically (stage 10)
ai3d run-pipeline \
    --input /path/to/object.jpg \
    --backend sf3d \
    --output-dir /path/to/output
```

## Pipeline Stages (video-relevant)

| Stage | What Happens |
|-------|-------------|
| GENERATION_3D | Mesh generated from image |
| BLENDER_RENDER | Turntable + depth pass via headless Blender |
| VIDEO_CONDITIONING | Frames organized into conditioning pack |

## Using the Pack for Video Generation

### CogVideoX / Wan2.1 (first-frame controlled)

```bash
# Export first frame
ai3d blender-render \
    --asset output.glb \
    --output-dir first_frame_render \
    --frames 1 --pass rgb

# Then feed first_frame_render/rgb/frame_0001.png to your video model
```

### Depth-conditioned video (ControlNet-style)

```bash
ai3d blender-render \
    --asset output.glb \
    --output-dir conditioning \
    --frames 72 --pass rgb --pass depth

# Feed conditioning/rgb/ as reference frames
# Feed conditioning/depth/ as depth conditioning to ControlNet-video models
```

## Depth Sequence Processing

Depth frames output by Blender's compositor are normalized to 0–255 PNG automatically.
For raw EXR depth export (Milestone 3): `ai3d.video.depth_sequence.DepthSequenceExporter`.

## Camera Motion Presets (Planned — M3)

- `turntable_360` — standard 0→360° rotation
- `dolly_in` — camera moves toward the object
- `orbit_tilt` — tilted orbit at variable elevation
- `spiral_rise` — spiral upward motion

These will be configurable as `camera_preset` in `TurntableSpec`.
