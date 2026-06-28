# ComfyUI Workflow Templates

This directory contains ComfyUI workflow JSON templates used by `ai3d run-workflow`.

## Placeholder Convention

All variable tokens use `__UPPER_SNAKE_CASE__` format.
The WorkflowManager fills them programmatically before submitting to ComfyUI.

## Available Templates

| File | Workflow Name | Status |
|------|--------------|--------|
| `triposr_to_video_i2v.json` | `triposr_turntable_to_i2v` | Enabled (M1) |
| `sf3d_to_video_i2v.json` | `sf3d_turntable_to_i2v` | Enabled (M1) |

## Adding a New Template

1. Export your workflow from ComfyUI as JSON (API format).
2. Replace variable values with `__PLACEHOLDER_NAME__` tokens.
3. Add an entry to `configs/workflows.yaml` referencing the template path.
4. Use `ai3d run-workflow --workflow <name> --param KEY=VALUE` to execute.
