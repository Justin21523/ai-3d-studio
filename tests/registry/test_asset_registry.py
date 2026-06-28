"""Tests for AssetRegistry CRUD operations."""
from __future__ import annotations

import pytest
from pathlib import Path
from ai3d.registry.asset_registry import AssetRegistry
from ai3d.core.models import AssetRegistryEntry, AssetStatus


@pytest.fixture()
def registry(tmp_path: Path) -> AssetRegistry:
    return AssetRegistry(registry_dir=tmp_path / "registry")


@pytest.fixture()
def sample_entry() -> AssetRegistryEntry:
    return AssetRegistryEntry(
        asset_id="asset-001",
        label="Test Object",
        source_image_path="/tmp/obj.png",
        backend_used="sf3d",
        output_paths={"glb": "/tmp/out/output.glb"},
        tags=["test", "object"],
    )


def test_register_and_get(registry, sample_entry):
    registry.register(sample_entry)
    loaded = registry.get("asset-001")
    assert loaded.asset_id == "asset-001"
    assert loaded.label == "Test Object"


def test_get_missing_raises(registry):
    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_exists(registry, sample_entry):
    assert not registry.exists("asset-001")
    registry.register(sample_entry)
    assert registry.exists("asset-001")


def test_list_empty(registry):
    assert registry.list() == []


def test_list_returns_all(registry, sample_entry):
    registry.register(sample_entry)
    entries = registry.list()
    assert len(entries) == 1
    assert entries[0].asset_id == "asset-001"


def test_list_by_backend(registry, sample_entry):
    registry.register(sample_entry)
    entries = registry.list_by_backend("sf3d")
    assert len(entries) == 1
    assert registry.list_by_backend("triposr") == []


def test_list_by_tag(registry, sample_entry):
    registry.register(sample_entry)
    assert len(registry.list_by_tag("test")) == 1
    assert registry.list_by_tag("missing") == []


def test_delete(registry, sample_entry):
    registry.register(sample_entry)
    assert registry.exists("asset-001")
    result = registry.delete("asset-001")
    assert result is True
    assert not registry.exists("asset-001")


def test_delete_nonexistent(registry):
    assert registry.delete("ghost") is False


def test_update(registry, sample_entry):
    registry.register(sample_entry)
    updated = registry.update("asset-001", label="Updated Label", status=AssetStatus.FAILED)
    assert updated.label == "Updated Label"
    assert updated.status == AssetStatus.FAILED
    # Verify persisted
    loaded = registry.get("asset-001")
    assert loaded.label == "Updated Label"


def test_roundtrip_yaml(registry, sample_entry):
    registry.register(sample_entry)
    loaded = registry.get("asset-001")
    assert loaded.output_paths["glb"] == "/tmp/out/output.glb"
    assert "test" in loaded.tags
