# Blender Templates

This directory stores `.blend` starter files used by the Blender integration.

## turntable_base.blend (placeholder)

The `turntable_base.blend` file is not included in version control (binary).

The `render_turntable.py` Blender script creates its scene programmatically,
so no template `.blend` is strictly required for the turntable workflow.

For custom scene setups (custom HDRIs, specific lighting rigs, camera presets),
place your `.blend` files here and reference them via `configs/paths.yaml`:

```yaml
blender_templates_dir: /path/to/your/blender_templates
```

## Adding a Custom Template

1. Save your `.blend` with the desired scene setup.
2. Update `BlenderSceneSpec.script_overrides` to reference it.
3. Modify `render_turntable.py` to open the template instead of a blank scene.
