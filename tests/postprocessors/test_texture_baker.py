"""Unit tests for TextureBaker."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_vertex_color_mesh():
    """Return a minimal trimesh.Trimesh with vertex colors."""
    import numpy as np
    import trimesh

    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    colors = np.array([
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ], dtype=np.uint8)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
    return mesh


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("trimesh"),
    reason="trimesh not installed",
)
def test_baker_missing_input(tmp_path):
    from ai3d.postprocessors.texture_baker import TextureBaker
    baker = TextureBaker(texture_size=64)
    result = baker.process(tmp_path / "nonexistent.glb", tmp_path / "out.glb")
    assert result.success is False
    assert "Failed to load mesh" in result.error or "trimesh" in result.error.lower()


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("trimesh"),
    reason="trimesh not installed",
)
def test_baker_no_colors_no_uv(tmp_path):
    """A mesh with no vertex colors (mocked so trimesh doesn't assign defaults)."""
    import trimesh
    import numpy as np
    from unittest.mock import patch, MagicMock
    from ai3d.postprocessors.texture_baker import TextureBaker

    mesh = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    in_path = tmp_path / "plain.glb"
    in_path.write_bytes(b"placeholder")

    baker = TextureBaker(texture_size=64)
    # Patch trimesh.load to return a mesh with no colors and no UV
    with patch("trimesh.load", return_value=mesh), \
         patch("ai3d.postprocessors.texture_baker._has_uv_texture", return_value=False), \
         patch("ai3d.postprocessors.texture_baker._has_vertex_colors", return_value=False):
        result = baker.process(in_path, tmp_path / "out.glb")
    assert result.success is False
    assert "neither vertex colors nor UV texture" in result.error


@pytest.mark.skipif(
    not (__import__("importlib").util.find_spec("trimesh")
         and __import__("importlib").util.find_spec("xatlas")),
    reason="trimesh or xatlas not installed",
)
def test_baker_vertex_color_to_uv(tmp_path):
    import trimesh
    from ai3d.postprocessors.texture_baker import TextureBaker

    mesh = _make_vertex_color_mesh()
    in_path = tmp_path / "vc_mesh.glb"
    mesh.export(str(in_path))

    baker = TextureBaker(texture_size=64)
    out_path = tmp_path / "textured.glb"
    result = baker.process(in_path, out_path)

    assert result.success is True
    assert out_path.exists()
    assert result.artifacts[0].path == str(out_path)
