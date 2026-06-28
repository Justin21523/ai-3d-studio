# Blender Integration

## Overview

The Blender integration launches Blender in headless (`--background`) mode via subprocess.
Communication uses a JSON spec file passed as a `--` argument to the Blender Python script.

## Architecture

```
TurntableRenderer
    │ render(TurntableSpec)
    ▼
BlenderBridge.launch_headless(BlenderSceneSpec, script_path)
    │ subprocess.run([blender, --background, --python, script.py, --, --scene-spec, spec.json])
    ▼
render_turntable.py  (runs INSIDE Blender's Python interpreter, uses bpy)
    │ Reads spec.json, imports mesh, builds scene, renders animation
    │ Prints: AI3D_RESULT:<json> on last stdout line
    ▼
BlenderBridge parses AI3D_RESULT → BlenderRenderResult
```

## Requirements

- Blender 3.6+ or 4.x installed (set path in `configs/paths.yaml` → `blender_executable`)
- GLB/OBJ/FBX/PLY mesh to render
- Writable output directory

## Quick Start

```bash
# Check Blender availability
ai3d blender-check

# Render 72-frame turntable (RGB + depth)
ai3d blender-render \
    --asset /path/to/output.glb \
    --output-dir /path/to/render_output \
    --frames 72 \
    --fps 24 \
    --resolution 1024x1024 \
    --pass rgb \
    --pass depth
```

## Render Passes

| Pass | Status | Description |
|------|--------|-------------|
| `rgb` | Implemented (M1) | Standard RGB render |
| `depth` | Implemented (M1) | Normalized depth pass via compositor |
| `normal` | Scaffold (M2) | World-space normals via Cycles compositor |
| `mask` | Scaffold (M2) | Per-object index pass |

Passes are rendered simultaneously in a single Blender session when possible.

## Output Structure

```
output_dir/
├── rgb/
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
└── depth/
    ├── frame_0001.png
    └── ...
```

## Blender Script Protocol

The `render_turntable.py` script:
1. Reads `--scene-spec <path>` JSON argument.
2. Clears the default scene.
3. Imports the mesh (auto-detects format from extension).
4. Centers and normalizes the mesh to a 2-unit bounding box.
5. Creates a camera at the specified elevation/distance.
6. Sets up compositor nodes for requested passes.
7. Animates the camera 0→360° over `frame_count` frames.
8. Renders with `bpy.ops.render.render(animation=True)`.
9. Prints `AI3D_RESULT:<json>` to stdout.

## Extending Blender Functionality

To add custom scene operations:
1. Create a new script in `ai3d/blender/scripts/`.
2. Add a new `operation` type to `BlenderSceneSpec` if needed.
3. Build the spec via `scene_builder.py` helpers.
4. Launch via `BlenderBridge.launch_headless()`.
