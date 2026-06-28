"""Blender turntable render script — executed INSIDE Blender's Python interpreter.

Usage (via BlenderBridge):
    blender --background --python render_turntable.py -- --scene-spec /path/to/spec.json

This script runs with bpy available. It must NOT import ai3d packages — only stdlib + bpy.
It prints AI3D_RESULT:<json> on the last line for BlenderBridge to parse.
"""
import json
import math
import os
import sys
from pathlib import Path


def parse_args() -> dict:
    argv = sys.argv
    if "--" not in argv:
        return {}
    extra = argv[argv.index("--") + 1:]
    result = {}
    i = 0
    while i < len(extra):
        if extra[i] == "--scene-spec" and i + 1 < len(extra):
            result["scene_spec"] = extra[i + 1]
            i += 2
        else:
            i += 1
    return result


def load_spec(spec_path: str) -> dict:
    with open(spec_path, encoding="utf-8") as f:
        return json.load(f)


def emit_result(data: dict) -> None:
    print(f"AI3D_RESULT:{json.dumps(data)}", flush=True)


def setup_scene(context):
    import bpy
    # Remove default objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # Clean orphan data
    for block_type in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for block in list(block_type):
            if block.users == 0:
                block_type.remove(block)


def import_mesh(mesh_path: str):
    import bpy
    ext = Path(mesh_path).suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif ext == ".obj":
        bpy.ops.import_scene.obj(filepath=mesh_path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=mesh_path)
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=mesh_path)
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")

    # Return all imported objects
    return [obj for obj in bpy.context.selected_objects]


def center_and_normalize(objects):
    import bpy
    import mathutils

    if not objects:
        return

    # Compute bounding box center
    min_c = [float("inf")] * 3
    max_c = [float("-inf")] * 3
    for obj in objects:
        for v in obj.bound_box:
            world_v = obj.matrix_world @ mathutils.Vector(v)
            for i in range(3):
                min_c[i] = min(min_c[i], world_v[i])
                max_c[i] = max(max_c[i], world_v[i])

    center = mathutils.Vector([(min_c[i] + max_c[i]) / 2 for i in range(3)])
    size = max(max_c[i] - min_c[i] for i in range(3))
    scale_factor = 2.0 / max(size, 1e-6)

    for obj in objects:
        obj.location -= center
        obj.scale *= scale_factor

    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.ops.object.transform_apply(scale=True, location=True)


def add_turntable_camera(elevation_deg: float, distance: float, res_x: int, res_y: int):
    import bpy
    import mathutils

    scene = bpy.context.scene
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y

    # Camera data
    cam_data = bpy.data.cameras.new("TurntableCamera")
    cam_data.lens = 50
    cam_obj = bpy.data.objects.new("TurntableCamera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Position camera at elevation on unit sphere, pointing at origin
    elevation_rad = math.radians(elevation_deg)
    cam_obj.location = mathutils.Vector((
        distance * math.cos(elevation_rad),
        0.0,
        distance * math.sin(elevation_rad),
    ))

    # Point camera at origin
    direction = mathutils.Vector((0, 0, 0)) - cam_obj.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot_quat.to_euler()

    # Add empty at origin as rotation pivot
    empty = bpy.data.objects.new("TurntablePivot", None)
    scene.collection.objects.link(empty)
    empty.location = (0, 0, 0)

    # Parent camera to empty
    cam_obj.parent = empty

    return cam_obj, empty


def add_world_lighting():
    import bpy
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bpy.context.scene.world = world

    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Strength"].default_value = 1.0


def setup_render_passes(render_passes: list[str], output_dir: str):
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32

    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = "depth" in render_passes
    view_layer.use_pass_normal = "normal" in render_passes
    view_layer.use_pass_object_index = "mask" in render_passes

    if len(render_passes) > 1 or "depth" in render_passes or "normal" in render_passes:
        scene.use_nodes = True
        _setup_compositor_nodes(scene, render_passes, output_dir)


def _setup_compositor_nodes(scene, render_passes: list[str], output_dir: str):
    import bpy

    tree = scene.node_tree
    tree.nodes.clear()
    links = tree.links

    render_layers = tree.nodes.new("CompositorNodeRLayers")
    render_layers.location = (0, 0)

    # RGB output
    if "rgb" in render_passes:
        rgb_out = tree.nodes.new("CompositorNodeOutputFile")
        rgb_out.base_path = str(Path(output_dir) / "rgb")
        rgb_out.file_slots[0].path = "frame_"
        rgb_out.format.file_format = "PNG"
        rgb_out.location = (400, 100)
        links.new(render_layers.outputs["Image"], rgb_out.inputs[0])

    # Depth output
    if "depth" in render_passes:
        depth_out = tree.nodes.new("CompositorNodeOutputFile")
        depth_out.base_path = str(Path(output_dir) / "depth")
        depth_out.file_slots[0].path = "frame_"
        depth_out.format.file_format = "PNG"
        depth_out.location = (400, -100)
        # Normalize depth
        normalize = tree.nodes.new("CompositorNodeNormalize")
        normalize.location = (200, -100)
        links.new(render_layers.outputs["Depth"], normalize.inputs[0])
        links.new(normalize.outputs[0], depth_out.inputs[0])

    # Normal output
    if "normal" in render_passes:
        normal_out = tree.nodes.new("CompositorNodeOutputFile")
        normal_out.base_path = str(Path(output_dir) / "normal")
        normal_out.file_slots[0].path = "frame_"
        normal_out.format.file_format = "PNG"
        normal_out.location = (400, -300)
        links.new(render_layers.outputs["Normal"], normal_out.inputs[0])


def set_keyframes(empty, frame_count: int, fps: int):
    import bpy
    import math

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frame_count
    scene.render.fps = fps

    # Animate empty Z rotation from 0 to 360 degrees
    empty.rotation_euler = (0, 0, 0)
    empty.keyframe_insert(data_path="rotation_euler", frame=1)
    empty.rotation_euler = (0, 0, math.radians(360))
    empty.keyframe_insert(data_path="rotation_euler", frame=frame_count)

    # Set interpolation to linear
    if empty.animation_data and empty.animation_data.action:
        for fcurve in empty.animation_data.action.fcurves:
            for kf in fcurve.keyframe_points:
                kf.interpolation = "LINEAR"


def render_animation(output_dir: str, render_passes: list[str]):
    import bpy

    scene = bpy.context.scene
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if "rgb" in render_passes and len(render_passes) == 1:
        # Simple RGB-only render
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(output_path / "rgb" / "frame_")
        Path(str(output_path / "rgb")).mkdir(parents=True, exist_ok=True)

    bpy.ops.render.render(animation=True)


def collect_frame_paths(output_dir: str, render_passes: list[str]) -> list[str]:
    paths = []
    out = Path(output_dir)
    for pass_name in render_passes:
        pass_dir = out / pass_name
        if pass_dir.exists():
            paths.extend(sorted(str(p) for p in pass_dir.glob("*.png")))
    return paths


def main():
    args = parse_args()
    spec_path = args.get("scene_spec")

    if not spec_path:
        emit_result({"success": False, "output_dir": "", "error": "No --scene-spec argument provided."})
        return

    try:
        spec = load_spec(spec_path)
    except Exception as exc:
        emit_result({"success": False, "output_dir": "", "error": f"Failed to load spec: {exc}"})
        return

    turntable = spec.get("turntable") or {}
    output_dir = spec.get("output_dir", "")
    mesh_path = spec.get("mesh_path", "")
    render_passes = [p for p in (turntable.get("render_passes") or spec.get("passes") or ["rgb"])]

    try:
        import bpy
        context = bpy.context

        setup_scene(context)
        objects = import_mesh(mesh_path)
        center_and_normalize(objects)

        frame_count = turntable.get("frame_count", 72)
        fps = turntable.get("fps", 24)
        elevation = turntable.get("camera_elevation_deg", 20.0)
        distance = turntable.get("camera_distance", 2.5)
        res_x = turntable.get("resolution_x", 1024)
        res_y = turntable.get("resolution_y", 1024)
        hdri_path = turntable.get("hdri_path")

        cam, pivot = add_turntable_camera(elevation, distance, res_x, res_y)
        add_world_lighting()

        if hdri_path and Path(hdri_path).exists():
            _apply_hdri(hdri_path)

        setup_render_passes(render_passes, output_dir)
        set_keyframes(pivot, frame_count, fps)
        render_animation(output_dir, render_passes)

        frame_paths = collect_frame_paths(output_dir, render_passes)
        pass_dirs = {p: str(Path(output_dir) / p) for p in render_passes}

        emit_result({
            "success": True,
            "output_dir": output_dir,
            "frame_paths": frame_paths,
            "pass_dirs": pass_dirs,
            "warnings": [],
            "metadata": {"frame_count": frame_count, "render_passes": render_passes},
        })

    except Exception as exc:
        import traceback
        emit_result({
            "success": False,
            "output_dir": output_dir,
            "error": str(exc),
            "warnings": [traceback.format_exc()],
        })


def _apply_hdri(hdri_path: str):
    import bpy
    world = bpy.context.scene.world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()
    bg = tree.nodes.new("ShaderNodeBackground")
    env = tree.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(hdri_path)
    output = tree.nodes.new("ShaderNodeOutputWorld")
    tree.links.new(env.outputs["Color"], bg.inputs["Color"])
    tree.links.new(bg.outputs["Background"], output.inputs["Surface"])


main()
