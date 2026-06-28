"""Blender single-frame render pass script — executed inside Blender.

Usage:
    blender --background --python render_passes.py -- --scene-spec /path/to/spec.json

This script runs with bpy available. It must not import ai3d packages.
It prints AI3D_RESULT:<json> on the last line for BlenderBridge to parse.
"""
import json
import math
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


def setup_scene():
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
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
    return [obj for obj in bpy.context.selected_objects]


def center_and_normalize(objects):
    import bpy
    import mathutils

    if not objects:
        return

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


def assign_object_indices(objects):
    for idx, obj in enumerate(objects, start=1):
        obj.pass_index = idx


def add_camera(elevation_deg: float, distance: float, res_x: int, res_y: int):
    import bpy
    import mathutils

    scene = bpy.context.scene
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y

    cam_data = bpy.data.cameras.new("PassCamera")
    cam_data.lens = 50
    cam_obj = bpy.data.objects.new("PassCamera", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    elevation_rad = math.radians(elevation_deg)
    cam_obj.location = mathutils.Vector((
        distance * math.cos(elevation_rad),
        0.0,
        distance * math.sin(elevation_rad),
    ))
    direction = mathutils.Vector((0, 0, 0)) - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return cam_obj


def add_world_lighting():
    import bpy

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bpy.context.scene.world = world
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Strength"].default_value = 1.0

    light_data = bpy.data.lights.new("KeyLight", type="AREA")
    light_data.energy = 500
    light_data.size = 5
    light_obj = bpy.data.objects.new("KeyLight", light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = (0.0, -3.0, 4.0)


def apply_hdri(hdri_path: str):
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


def setup_render_passes(render_passes: list[str], output_dir: str, passes_format: list[str]):
    import bpy

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 1
    scene.frame_set(1)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32

    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = "depth" in render_passes
    view_layer.use_pass_normal = "normal" in render_passes
    view_layer.use_pass_object_index = "mask" in render_passes

    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    links = tree.links

    render_layers = tree.nodes.new("CompositorNodeRLayers")
    render_layers.location = (0, 0)

    file_format = "OPEN_EXR" if "exr" in {p.lower() for p in passes_format} else "PNG"
    extension = "exr" if file_format == "OPEN_EXR" else "png"

    def add_output(pass_name: str, source_socket, y: int):
        out_dir = Path(output_dir) / pass_name
        out_dir.mkdir(parents=True, exist_ok=True)
        node = tree.nodes.new("CompositorNodeOutputFile")
        node.base_path = str(out_dir)
        node.file_slots[0].path = "frame_"
        node.format.file_format = file_format
        if file_format == "OPEN_EXR":
            node.format.color_depth = "32"
        node.location = (420, y)
        links.new(source_socket, node.inputs[0])

    if "rgb" in render_passes:
        add_output("rgb", render_layers.outputs["Image"], 150)

    if "depth" in render_passes:
        normalize = tree.nodes.new("CompositorNodeNormalize")
        normalize.location = (210, -50)
        links.new(render_layers.outputs["Depth"], normalize.inputs[0])
        add_output("depth", normalize.outputs[0], -50)

    if "normal" in render_passes:
        add_output("normal", render_layers.outputs["Normal"], -250)

    if "mask" in render_passes:
        add_output("mask", render_layers.outputs["IndexOB"], -450)

    return extension


def render_single(output_dir: str, render_passes: list[str], extension: str):
    import bpy

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(animation=False)
    return collect_frame_paths(output_dir, render_passes, extension)


def collect_frame_paths(output_dir: str, render_passes: list[str], extension: str) -> list[str]:
    paths = []
    out = Path(output_dir)
    for pass_name in render_passes:
        pass_dir = out / pass_name
        if pass_dir.exists():
            paths.extend(sorted(str(p) for p in pass_dir.glob(f"*.{extension}")))
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
    output_dir = spec.get("output_dir", turntable.get("output_dir", ""))
    mesh_path = spec.get("mesh_path") or spec.get("asset_path") or turntable.get("asset_path", "")
    render_passes = [str(p) for p in (spec.get("passes") or turntable.get("render_passes") or ["rgb"])]
    passes_format = [str(p) for p in spec.get("passes_format", [])]

    try:
        setup_scene()
        objects = import_mesh(mesh_path)
        center_and_normalize(objects)
        assign_object_indices(objects)

        elevation = turntable.get("camera_elevation_deg", spec.get("camera_elevation_deg", 20.0))
        distance = turntable.get("camera_distance", spec.get("camera_distance", 2.5))
        res_x = turntable.get("resolution_x", spec.get("resolution_x", 1024))
        res_y = turntable.get("resolution_y", spec.get("resolution_y", 1024))
        hdri_path = turntable.get("hdri_path") or spec.get("hdri_path")

        add_camera(elevation, distance, res_x, res_y)
        add_world_lighting()
        if hdri_path and Path(hdri_path).exists():
            apply_hdri(hdri_path)

        extension = setup_render_passes(render_passes, output_dir, passes_format)
        frame_paths = render_single(output_dir, render_passes, extension)
        pass_dirs = {p: str(Path(output_dir) / p) for p in render_passes}

        emit_result({
            "success": True,
            "output_dir": output_dir,
            "frame_paths": frame_paths,
            "pass_dirs": pass_dirs,
            "warnings": [],
            "metadata": {
                "frame_count": 1,
                "render_passes": render_passes,
                "format": extension,
            },
        })
    except Exception as exc:
        import traceback

        emit_result({
            "success": False,
            "output_dir": output_dir,
            "error": str(exc),
            "warnings": [traceback.format_exc()],
        })


main()
