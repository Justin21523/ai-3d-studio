# ComfyUI Workflows

## Overview

The ComfyUI integration uses a lightweight HTTP client to submit workflow graphs,
poll for completion, and download outputs. No ComfyUI Python package is required —
only the ComfyUI server running locally or on a remote host.

## Architecture

```
WorkflowRegistry.get(name) → WorkflowRegistryEntry
    │
WorkflowManager.prepare(template_path, replacements)
    │ Fills __PLACEHOLDER__ tokens in the JSON graph
    ▼
ComfyUIClient.queue_prompt(filled_graph) → prompt_id
    │
ResultPoller.wait_and_download(prompt_id, output_dir)
    │ Polls GET /history/{prompt_id}
    │ Downloads outputs via GET /view
    ▼
ComfyJobResult { output_files: [ArtifactRef, ...] }
```

## Quick Start

```bash
# Check ComfyUI connectivity
ai3d comfyui-check

# List available workflows
ai3d list-workflows

# Execute a workflow
ai3d run-workflow \
    --workflow sf3d_turntable_to_i2v \
    --input /path/to/output.glb \
    --output-dir /path/to/comfy_output \
    --param PROMPT="turntable rotation, 3d object" \
    --param FRAME_COUNT=72
```

## Placeholder Convention

Templates use `__UPPER_SNAKE_CASE__` tokens.
Override from CLI with `--param KEY=VALUE` (key without underscores, matched case-insensitively).

## Available Workflows (M1)

| Name | Type | Backend |
|------|------|---------|
| `triposr_turntable_to_i2v` | turntable_conditioning | triposr |
| `sf3d_turntable_to_i2v` | turntable_conditioning | sf3d |

Note: The JSON templates in `workflows/` are structural scaffolds.
Replace node definitions with your actual ComfyUI workflow exported in API format.

## Exporting Your Own Template

1. Build your workflow in ComfyUI.
2. Open DevTools → Network, submit the workflow, capture the `/prompt` POST body.
3. Extract the `"prompt"` object from the request body.
4. Replace specific values with `__PLACEHOLDER__` tokens.
5. Save as `workflows/<name>.json`.
6. Add to `configs/workflows.yaml`.

## ComfyUI Server Configuration

Edit `configs/paths.yaml`:
```yaml
comfyui_root: /mnt/c/ai_tools/comfyui
comfyui_input_dir: /mnt/c/ai_tools/comfyui/input
comfyui_base_url: http://127.0.0.1:8188
```

For remote servers: set `comfyui_base_url` to the remote address.
