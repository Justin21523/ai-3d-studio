"""Tests for BlenderBridge."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ai3d.blender.bridge import BlenderBridge, _parse_result_from_stdout
from ai3d.core.models import BlenderSceneSpec


def test_is_available_blender_missing(tmp_path):
    bridge = BlenderBridge(blender_exe=tmp_path / "nonexistent_blender")
    assert bridge.is_available() is False


def test_get_version_missing(tmp_path):
    bridge = BlenderBridge(blender_exe=tmp_path / "nonexistent_blender")
    version = bridge.get_version()
    assert "error" in version.lower()


def test_parse_result_from_stdout_with_marker():
    stdout = 'some output\nAI3D_RESULT:{"success": true, "output_dir": "/tmp", "frame_paths": []}'
    result = _parse_result_from_stdout(stdout, "/tmp")
    assert result.success is True
    assert result.output_dir == "/tmp"


def test_parse_result_no_marker():
    result = _parse_result_from_stdout("no marker here", "/tmp/fallback")
    assert result.success is True
    assert result.output_dir == "/tmp/fallback"
    assert result.warnings


def test_launch_headless_blender_unavailable(tmp_path):
    bridge = BlenderBridge(blender_exe=tmp_path / "ghost_blender")
    spec = BlenderSceneSpec(
        operation="turntable",
        mesh_path="/tmp/mesh.glb",
        output_dir=str(tmp_path / "out"),
    )
    result = bridge.launch_headless(spec, script_path=Path("/tmp/script.py"))
    assert not result.success
    assert "not found" in result.error.lower()
