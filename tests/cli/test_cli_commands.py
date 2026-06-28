"""Smoke tests for CLI commands."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ai3d.cli.main import main


def _run(args: list[str]) -> tuple[int, dict]:
    """Run CLI and capture JSON stdout."""
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(args)
    output = buf.getvalue().strip()
    try:
        data = json.loads(output) if output else {}
    except json.JSONDecodeError:
        data = {"raw": output}
    return code, data


def test_list_backends_returns_dict():
    code, data = _run(["list-backends"])
    assert code == 0
    # Should return a dict of backend names -> capability dicts
    assert isinstance(data, dict)
    assert "trellis" in data
    assert "mesh2splat" in data


def test_check_backend_scaffold():
    code, data = _run(["check-backend", "--backend", "trellis"])
    assert code == 0
    assert data["availability"]["available"] is False


def test_check_backend_unknown():
    code, data = _run(["check-backend", "--backend", "nonexistent_xyz"])
    assert code != 0


def test_blender_check_returns_dict():
    code, data = _run(["blender-check"])
    assert code == 0
    assert "available" in data
    assert "version" in data


def test_comfyui_check_unreachable():
    code, data = _run(["comfyui-check", "--base-url", "http://127.0.0.1:19999"])
    assert code == 0
    assert data["available"] is False


def test_list_workflows():
    code, data = _run(["list-workflows"])
    assert code == 0
    assert isinstance(data, list)
    names = [w["name"] for w in data]
    assert "triposr_turntable_to_i2v" in names
    assert "sf3d_turntable_to_i2v" in names
    assert "hunyuan3d21_image_to_model" in names


def test_list_3d_models():
    code, data = _run(["list-3d-models"])
    assert code == 0
    names = [m["name"] for m in data]
    assert "hunyuan3d21" in names
    assert "triposr" in names


def test_list_assets_empty(tmp_path):
    with patch("ai3d.registry.asset_registry.asset_registry_dir", return_value=tmp_path / "reg"):
        code, data = _run(["list-assets"])
    assert code == 0
    assert data == []


def test_mesh_clean_missing_input(tmp_path):
    code, data = _run([
        "mesh-clean",
        "--input", str(tmp_path / "ghost.glb"),
        "--output", str(tmp_path / "out.glb"),
    ])
    assert code == 0
    assert data["success"] is False


def test_rig_model_missing_input(tmp_path):
    code, data = _run([
        "rig-model",
        "--input", str(tmp_path / "ghost.glb"),
        "--output-dir", str(tmp_path / "out"),
        "--dry-run",
    ])
    assert code != 0


def test_install_models_shows_commands():
    code, data = _run(["install-models"])
    assert code == 0
    assert isinstance(data, list)
    # All enabled models should appear
    names = [r["model"] for r in data]
    assert "triposr" in names
    assert "sf3d" in names
