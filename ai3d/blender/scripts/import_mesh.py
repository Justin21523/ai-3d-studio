"""Blender mesh import helpers — used inside Blender's Python interpreter.

This module provides reusable import utilities for other Blender scripts.
It must NOT import ai3d packages — only stdlib + bpy.
"""
from pathlib import Path


def import_mesh_by_extension(mesh_path: str) -> list:
    """Import a mesh file into the active Blender scene based on file extension."""
    import bpy

    ext = Path(mesh_path).suffix.lower()
    bpy.ops.object.select_all(action="DESELECT")

    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif ext == ".obj":
        bpy.ops.import_scene.obj(filepath=mesh_path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=mesh_path)
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=mesh_path)
    elif ext == ".stl":
        bpy.ops.import_mesh.stl(filepath=mesh_path)
    else:
        raise ValueError(f"Unsupported mesh extension: {ext}")

    return list(bpy.context.selected_objects)


def select_mesh_objects() -> list:
    """Return all MESH-type objects in the scene."""
    import bpy
    return [obj for obj in bpy.data.objects if obj.type == "MESH"]
