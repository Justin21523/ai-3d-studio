"""Texture baker — vertex-color-to-UV-texture baking via barycentric rasterization.

For TripoSR and other backends that produce vertex-color meshes, this bakes the
vertex colors into a UV texture atlas using xatlas UV coordinates.

For PBR-textured backends (SF3D, Hunyuan3D) this is a no-op since they already
produce UV-mapped texture maps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ai3d.core.logging import get_logger
from ai3d.core.models import ArtifactRef, BasePostprocessor, GenerationResult, StandardOutput
from ai3d.core.storage import ensure_directory

_log = get_logger(__name__)

_DEFAULT_TEXTURE_SIZE = 1024


class TextureBaker(BasePostprocessor):
    """Bakes vertex colors onto a UV-mapped texture atlas."""

    name = "texture_baker"

    def __init__(self, texture_size: int = _DEFAULT_TEXTURE_SIZE) -> None:
        self._texture_size = texture_size

    def process(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any,
    ) -> GenerationResult:
        texture_size: int = kwargs.get("texture_size", self._texture_size)
        ensure_directory(output_path.parent)

        try:
            import trimesh
            import numpy as np
        except ImportError as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="texture-bake",
                error=f"trimesh/numpy not installed: {exc}",
            )

        try:
            mesh = trimesh.load(str(input_path), force="mesh")
        except Exception as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="texture-bake",
                error=f"Failed to load mesh: {exc}",
            )

        if not isinstance(mesh, trimesh.Trimesh):
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="texture-bake",
                error="Input must be a single Trimesh (not a scene).",
            )

        # If the mesh already has UV coords + material texture, just export it
        if _has_uv_texture(mesh):
            _log.info("Mesh already has UV texture; exporting as-is to %s", output_path)
            mesh.export(str(output_path))
            return GenerationResult(
                success=True,
                provider=self.name,
                task_type="texture-bake",
                artifacts=[ArtifactRef(
                    path=str(output_path),
                    kind="mesh",
                    label="UV-textured mesh (passthrough)",
                    output_type=StandardOutput.TEXTURED_MESH,
                    size_bytes=output_path.stat().st_size if output_path.exists() else 0,
                )],
            )

        # Check for vertex colors
        if not _has_vertex_colors(mesh):
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="texture-bake",
                error="Mesh has neither vertex colors nor UV texture; nothing to bake.",
            )

        _log.info("Baking vertex colors to %dx%d UV texture", texture_size, texture_size)

        # UV unwrap with xatlas
        try:
            import xatlas
        except ImportError as exc:
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="texture-bake",
                error=f"xatlas not installed: {exc}",
            )

        vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
        vertices_new = mesh.vertices[vmapping]
        vertex_colors_new = _get_vertex_colors(mesh)[vmapping]  # shape: (N, 4) RGBA

        # Rasterize vertex colors into texture atlas
        texture_img = _rasterize_vertex_colors(
            vertices_new, indices, uvs, vertex_colors_new, texture_size
        )

        # Build new textured mesh
        try:
            from PIL import Image as PILImage
            import io

            # Encode texture as PNG bytes
            buf = io.BytesIO()
            texture_img.save(buf, format="PNG")
            texture_bytes = buf.getvalue()

            # Build trimesh with UV material
            material = trimesh.visual.texture.SimpleMaterial(
                image=texture_img,
            )
            uv_visual = trimesh.visual.TextureVisuals(
                uv=uvs,
                material=material,
            )
            textured_mesh = trimesh.Trimesh(
                vertices=vertices_new,
                faces=indices,
                visual=uv_visual,
                process=False,
            )

            textured_mesh.export(str(output_path))
            size = output_path.stat().st_size if output_path.exists() else 0

            return GenerationResult(
                success=True,
                provider=self.name,
                task_type="texture-bake",
                artifacts=[ArtifactRef(
                    path=str(output_path),
                    kind="mesh",
                    label=f"UV-baked texture ({texture_size}px)",
                    output_type=StandardOutput.TEXTURED_MESH,
                    size_bytes=size,
                )],
                metadata={"texture_size": texture_size},
            )

        except Exception as exc:
            _log.exception("Texture export failed")
            return GenerationResult(
                success=False,
                provider=self.name,
                task_type="texture-bake",
                error=f"Failed to export textured mesh: {exc}",
            )


def _has_uv_texture(mesh: "trimesh.Trimesh") -> bool:
    import trimesh
    v = mesh.visual
    return isinstance(v, trimesh.visual.TextureVisuals) and v.uv is not None


def _has_vertex_colors(mesh: "trimesh.Trimesh") -> bool:
    import trimesh
    import numpy as np
    v = mesh.visual
    if isinstance(v, trimesh.visual.ColorVisuals):
        vc = v.vertex_colors
        return vc is not None and len(vc) == len(mesh.vertices)
    return False


def _get_vertex_colors(mesh: "trimesh.Trimesh") -> "np.ndarray":
    import numpy as np
    colors = mesh.visual.vertex_colors  # (N, 4) uint8
    return np.array(colors, dtype=np.float32) / 255.0


def _rasterize_vertex_colors(
    vertices: "np.ndarray",
    faces: "np.ndarray",
    uvs: "np.ndarray",
    vertex_colors: "np.ndarray",
    texture_size: int,
) -> "PILImage.Image":
    """Barycentric rasterization of vertex colors into a UV atlas image."""
    import numpy as np
    from PIL import Image, ImageDraw

    tex = np.zeros((texture_size, texture_size, 4), dtype=np.float32)
    weight = np.zeros((texture_size, texture_size), dtype=np.float32)

    uv_px = uvs * (texture_size - 1)  # scale to pixel coords

    for face in faces:
        i0, i1, i2 = face
        p0 = uv_px[i0]
        p1 = uv_px[i1]
        p2 = uv_px[i2]
        c0 = vertex_colors[i0]
        c1 = vertex_colors[i1]
        c2 = vertex_colors[i2]

        # Bounding box of triangle in UV space
        min_x = max(0, int(min(p0[0], p1[0], p2[0])))
        max_x = min(texture_size - 1, int(max(p0[0], p1[0], p2[0])) + 1)
        min_y = max(0, int(min(p0[1], p1[1], p2[1])))
        max_y = min(texture_size - 1, int(max(p0[1], p1[1], p2[1])) + 1)

        if max_x <= min_x or max_y <= min_y:
            continue

        # Rasterize pixels inside triangle bounding box
        xs = np.arange(min_x, max_x + 1)
        ys = np.arange(min_y, max_y + 1)
        px, py = np.meshgrid(xs, ys)
        px = px.flatten().astype(np.float32)
        py = py.flatten().astype(np.float32)

        # Barycentric coordinates
        v0 = p1 - p0
        v1 = p2 - p0
        v2 = np.stack([px - p0[0], py - p0[1]], axis=1)

        d00 = v0 @ v0
        d01 = v0 @ v1
        d11 = v1 @ v1
        denom = d00 * d11 - d01 * d01

        if abs(denom) < 1e-10:
            continue

        d20 = v2 @ v0
        d21 = v2 @ v1
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w

        inside = (u >= 0) & (v >= 0) & (w >= 0)

        px_in = px[inside].astype(int)
        py_in = py[inside].astype(int)
        u_in = u[inside, None]
        v_in = v[inside, None]
        w_in = w[inside, None]

        colors = u_in * c0 + v_in * c1 + w_in * c2
        tex[py_in, px_in] += colors
        weight[py_in, px_in] += 1.0

    # Normalize and convert
    valid = weight > 0
    tex[valid] /= weight[valid, None]
    tex_uint8 = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(tex_uint8, mode="RGBA")
