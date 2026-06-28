"""Asset registry — CRUD over per-asset YAML files in the registry directory."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ai3d.core.logging import get_logger
from ai3d.core.models import AssetRegistryEntry, AssetStatus
from ai3d.core.paths import asset_registry_dir
from ai3d.core.storage import ensure_directory, read_model, write_model

_log = get_logger(__name__)


class AssetRegistry:
    def __init__(self, registry_dir: Optional[Path] = None) -> None:
        self._dir = registry_dir or asset_registry_dir()

    def _path_for(self, asset_id: str) -> Path:
        return self._dir / f"{asset_id}.yaml"

    # ── Write ─────────────────────────────────────────────────────────────────

    def register(self, entry: AssetRegistryEntry) -> Path:
        ensure_directory(self._dir)
        path = self._path_for(entry.asset_id)
        write_model(path, entry)
        _log.info("Registered asset: %s (%s)", entry.asset_id, entry.label)
        return path

    def update(self, asset_id: str, **fields) -> AssetRegistryEntry:
        entry = self.get(asset_id)
        updated = entry.model_copy(update=fields)
        self.register(updated)
        return updated

    def delete(self, asset_id: str) -> bool:
        path = self._path_for(asset_id)
        if path.exists():
            path.unlink()
            _log.info("Deleted asset: %s", asset_id)
            return True
        return False

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, asset_id: str) -> AssetRegistryEntry:
        path = self._path_for(asset_id)
        if not path.exists():
            raise KeyError(f"Asset not found in registry: {asset_id}")
        return read_model(path, AssetRegistryEntry)

    def exists(self, asset_id: str) -> bool:
        return self._path_for(asset_id).exists()

    def list(self) -> List[AssetRegistryEntry]:
        if not self._dir.exists():
            return []
        entries: List[AssetRegistryEntry] = []
        for yaml_path in sorted(self._dir.glob("*.yaml")):
            try:
                entries.append(read_model(yaml_path, AssetRegistryEntry))
            except Exception as exc:
                _log.warning("Skipping malformed asset file %s: %s", yaml_path.name, exc)
        return entries

    def list_by_backend(self, backend: str) -> List[AssetRegistryEntry]:
        return [e for e in self.list() if e.backend_used == backend]

    def list_by_tag(self, tag: str) -> List[AssetRegistryEntry]:
        return [e for e in self.list() if tag in e.tags]

    def list_by_status(self, status: AssetStatus) -> List[AssetRegistryEntry]:
        return [e for e in self.list() if e.status == status]
