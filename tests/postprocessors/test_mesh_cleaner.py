"""Tests for MeshCleaner."""
from __future__ import annotations

import pytest
from ai3d.postprocessors.mesh_cleaner import MeshCleaner
from ai3d.core.models import MeshCleanRequest


def test_missing_input(tmp_path):
    result = MeshCleaner().clean(MeshCleanRequest(
        input_path=str(tmp_path / "ghost.glb"),
        output_path=str(tmp_path / "out.glb"),
    ))
    assert not result.success
    assert result.error is not None


def test_clean_basic_mesh(tmp_path):
    """Create a minimal trimesh, save it, then clean it."""
    pytest.importorskip("trimesh")
    import trimesh
    import numpy as np

    # Simple tetrahedron
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1],
    ], dtype=float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    input_path = tmp_path / "input.glb"
    output_path = tmp_path / "cleaned.glb"
    mesh.export(str(input_path))

    result = MeshCleaner().clean(MeshCleanRequest(
        input_path=str(input_path),
        output_path=str(output_path),
    ))

    assert result.success
    assert output_path.exists()
    assert result.cleaned_face_count >= 0
