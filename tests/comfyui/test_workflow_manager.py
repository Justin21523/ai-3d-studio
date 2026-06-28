"""Tests for WorkflowManager placeholder fill and validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from ai3d.comfyui.workflow_manager import WorkflowManager, PLACEHOLDER_PATTERN


def _write_template(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_collect_placeholders():
    wm = WorkflowManager()
    graph = {
        "node1": {"inputs": {"image": "__INPUT_IMAGE__", "seed": "__SEED__"}},
        "node2": {"inputs": {"prompt": "a photo of __SUBJECT__"}},
    }
    found = wm.collect_placeholders(graph)
    assert "__INPUT_IMAGE__" in found
    assert "__SEED__" in found
    assert "__SUBJECT__" in found


def test_fill_placeholders_string():
    wm = WorkflowManager()
    graph = {"inputs": {"path": "__MESH_PATH__"}}
    filled = wm.fill_placeholders(graph, {"MESH_PATH": "/tmp/output.glb"})
    assert filled["inputs"]["path"] == "/tmp/output.glb"


def test_fill_placeholders_preserves_type():
    wm = WorkflowManager()
    graph = {"inputs": {"seed": "__SEED__", "count": "__FRAME_COUNT__"}}
    filled = wm.fill_placeholders(graph, {"SEED": 42, "FRAME_COUNT": 72})
    assert filled["inputs"]["seed"] == 42
    assert filled["inputs"]["count"] == 72


def test_fill_placeholders_nested():
    wm = WorkflowManager()
    graph = {"a": {"b": {"c": "__DEEP__"}}}
    filled = wm.fill_placeholders(graph, {"DEEP": "hello"})
    assert filled["a"]["b"]["c"] == "hello"


def test_fill_placeholders_list():
    wm = WorkflowManager()
    graph = {"items": ["__A__", "__B__", "static"]}
    filled = wm.fill_placeholders(graph, {"A": "x", "B": "y"})
    assert filled["items"] == ["x", "y", "static"]


def test_fill_does_not_mutate_original():
    wm = WorkflowManager()
    original = {"key": "__VAL__"}
    wm.fill_placeholders(original, {"VAL": "replaced"})
    assert original["key"] == "__VAL__"


def test_validate_missing_placeholder(tmp_path: Path):
    wm = WorkflowManager()
    template = _write_template(tmp_path / "wf.json", {"x": "__PRESENT__"})
    result = wm.validate(template, required_placeholders=["PRESENT", "MISSING"])
    assert not result["valid"]
    assert any("MISSING" in e for e in result["errors"])


def test_validate_passes(tmp_path: Path):
    wm = WorkflowManager()
    template = _write_template(tmp_path / "wf.json", {"x": "__INPUT__", "y": "__OUTPUT__"})
    result = wm.validate(template, required_placeholders=["INPUT", "OUTPUT"])
    assert result["valid"]
    assert not result["errors"]


def test_load_template_not_found(tmp_path: Path):
    wm = WorkflowManager()
    with pytest.raises(FileNotFoundError):
        wm.load_template(tmp_path / "nonexistent.json")


def test_prepare_roundtrip(tmp_path: Path):
    wm = WorkflowManager()
    template = _write_template(tmp_path / "wf.json", {
        "inputs": {"image": "__INPUT__", "frames": "__FRAME_COUNT__"}
    })
    filled = wm.prepare(template, {"INPUT": "/path/img.png", "FRAME_COUNT": 24})
    assert filled["inputs"]["image"] == "/path/img.png"
    assert filled["inputs"]["frames"] == 24


def test_prepare_strips_top_level_metadata(tmp_path: Path):
    wm = WorkflowManager()
    template = _write_template(tmp_path / "wf.json", {
        "_comment": "template metadata",
        "_placeholders": {"INPUT": "source image"},
        "1": {"class_type": "LoadImage", "inputs": {"image": "__INPUT__"}},
    })
    filled = wm.prepare(template, {"INPUT": "chair.png"})
    assert "_comment" not in filled
    assert "_placeholders" not in filled
    assert filled["1"]["inputs"]["image"] == "chair.png"
