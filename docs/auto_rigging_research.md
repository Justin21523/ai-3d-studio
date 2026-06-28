# Auto-Rigging Options For AI3D

This note tracks open-source rigging routes that can make generated `.glb` assets animation-ready.

## Installed Locally

### RigAnything

- Source: `https://github.com/Isabella98Liu/RigAnything`
- Local source: `/mnt/c/ai_tools/RigAnything`
- Weights: `/mnt/c/ai_models/vision/riganything/riganything_ckpt.pt`
- Expected input: `.glb` or `.obj`
- Expected output: rigged `.glb`
- CLI wrapper:

```bash
python -m ai3d.cli.main rig-model \
  --input /path/to/model.glb \
  --output-dir /tmp/rigged \
  --simplify \
  --target-faces 8192
```

RigAnything is the most direct fit for `image -> glb -> rigged glb` because it accepts GLB input and exports a rigged GLB.

### ComfyUI-UniRig

- Source: `https://github.com/PozzettiAndrea/ComfyUI-UniRig`
- Local source: `/mnt/c/ai_tools/comfyui/custom_nodes/ComfyUI-UniRig`
- Role: ComfyUI custom nodes for skeleton extraction, skinning, pose manipulation, and animation workflows.
- Restart ComfyUI after installation so its prestartup/install hooks can register the nodes.
- Bundled example workflows:
  - `workflows/unirig_humanoid.json`
  - `workflows/unirig_bird.json`
  - `workflows/mia_humanoid.json`
  - `workflows/apply_animation.json`

This is the best ComfyUI-native path once the custom node has finished installing inside ComfyUI.

### UniRig

- Source: `https://github.com/VAST-AI-Research/UniRig`
- Local source: `/mnt/c/ai_tools/UniRig`
- Expected input: `.obj`, `.fbx`, `.glb`, `.vrm`
- Outputs are typically `.fbx` skeleton/skinning artifacts.

UniRig is useful as the underlying research implementation. For this project, ComfyUI-UniRig is the cleaner integration route.

## Researched Candidates

### RigNet / bRigNet

- Source: `https://github.com/zhan-xu/RigNet`
- Blender add-on wrapper: `https://github.com/pKrime/brignet`
- Older but practical for character meshes. It usually requires mesh simplification and a separate Blender workflow.

### MagicArticulate

- Source: `https://github.com/Seed3D/MagicArticulate`
- Focus: articulation-ready 3D models, skeleton generation, and Blender utilities.
- Useful to watch, but less immediately plug-and-play for this repo than RigAnything.

### Anymate

- Project: `https://anymate3d.github.io/`
- Large-scale object rigging dataset and baseline models.
- Important research direction, but not the first integration target until inference packaging is simpler.

## Recommended Integration Order

1. Use `run-3d --model hunyuan3d21` or `run-pipeline --backend triposr` to produce GLB.
2. Run `rig-model --provider riganything` to produce a rigged GLB.
3. Use Blender or ComfyUI-UniRig to apply/retarget animations.
4. Add a later `animate-model` command once the preferred animation retargeting route is chosen.
