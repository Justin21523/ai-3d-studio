"""Tests for storage helpers."""
from __future__ import annotations

from pathlib import Path

import pytest
from ai3d.core.models import AssetRegistryEntry
from ai3d.core.storage import ensure_directory, read_model, read_yaml, write_model, write_yaml


def test_write_and_read_yaml(tmp_path: Path):
    data = {"key": "value", "list": [1, 2, 3]}
    p = tmp_path / "test.yaml"
    write_yaml(p, data)
    loaded = read_yaml(p)
    assert loaded["key"] == "value"
    assert loaded["list"] == [1, 2, 3]


def test_write_and_read_model(tmp_path: Path):
    entry = AssetRegistryEntry(
        asset_id="test-001",
        label="My Asset",
        source_image_path="/tmp/img.png",
        backend_used="sf3d",
    )
    p = tmp_path / "asset.yaml"
    write_model(p, entry)
    loaded = read_model(p, AssetRegistryEntry)
    assert loaded.asset_id == "test-001"
    assert loaded.label == "My Asset"
    assert loaded.backend_used == "sf3d"


def test_ensure_directory(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "c"
    result = ensure_directory(nested)
    assert result.exists()
    assert result.is_dir()


def test_write_yaml_creates_parent_dirs(tmp_path: Path):
    p = tmp_path / "deep" / "nested" / "file.yaml"
    write_yaml(p, {"x": 1})
    assert p.exists()
