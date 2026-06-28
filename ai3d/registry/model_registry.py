"""Model registry — loads and queries configs/models.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ai3d.core.logging import get_logger
from ai3d.core.models import ModelRegistryEntry
from ai3d.core.paths import CONFIGS_ROOT
from ai3d.core.storage import read_yaml

_log = get_logger(__name__)


class ModelRegistry:
    def __init__(self, manifest_path: Optional[Path] = None) -> None:
        self._path = manifest_path or (CONFIGS_ROOT / "models.yaml")
        self._entries: Dict[str, ModelRegistryEntry] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def load(self) -> None:
        raw = read_yaml(self._path)
        models_list = raw.get("models", [])
        self._entries = {}
        for item in models_list:
            try:
                entry = ModelRegistryEntry.model_validate(item)
                self._entries[entry.name] = entry
            except Exception as exc:
                _log.warning("Skipping malformed model entry: %s — %s", item.get("name", "?"), exc)
        self._loaded = True
        _log.debug("ModelRegistry loaded %d entries from %s", len(self._entries), self._path)

    def get(self, name: str) -> ModelRegistryEntry:
        self._ensure_loaded()
        if name not in self._entries:
            raise KeyError(f"Model '{name}' not in registry. Known: {sorted(self._entries)}")
        return self._entries[name]

    def list(self) -> List[ModelRegistryEntry]:
        self._ensure_loaded()
        return list(self._entries.values())

    def list_enabled(self) -> List[ModelRegistryEntry]:
        return [e for e in self.list() if e.enabled]

    def is_local(self, name: str) -> bool:
        entry = self.get(name)
        return Path(entry.local_path).exists()

    def get_local_path(self, name: str) -> Optional[Path]:
        entry = self.get(name)
        p = Path(entry.local_path)
        return p if p.exists() else None
