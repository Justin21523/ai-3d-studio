"""Workflow registry — loads and queries configs/workflows.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ai3d.core.logging import get_logger
from ai3d.core.models import WorkflowRegistryEntry
from ai3d.core.paths import CONFIGS_ROOT
from ai3d.core.storage import read_yaml

_log = get_logger(__name__)


class WorkflowRegistry:
    def __init__(self, manifest_path: Optional[Path] = None) -> None:
        self._path = manifest_path or (CONFIGS_ROOT / "workflows.yaml")
        self._entries: Dict[str, WorkflowRegistryEntry] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def load(self) -> None:
        raw = read_yaml(self._path)
        workflows_list = raw.get("workflows", [])
        self._entries = {}
        for item in workflows_list:
            try:
                entry = WorkflowRegistryEntry.model_validate(item)
                self._entries[entry.name] = entry
            except Exception as exc:
                _log.warning("Skipping malformed workflow entry: %s — %s", item.get("name", "?"), exc)
        self._loaded = True
        _log.debug("WorkflowRegistry loaded %d entries", len(self._entries))

    def get(self, name: str) -> WorkflowRegistryEntry:
        self._ensure_loaded()
        if name not in self._entries:
            raise KeyError(f"Workflow '{name}' not found. Known: {sorted(self._entries)}")
        return self._entries[name]

    def list(self) -> List[WorkflowRegistryEntry]:
        self._ensure_loaded()
        return list(self._entries.values())

    def list_enabled(self) -> List[WorkflowRegistryEntry]:
        return [e for e in self.list() if e.enabled]

    def list_by_type(self, workflow_type: str) -> List[WorkflowRegistryEntry]:
        return [e for e in self.list() if e.type == workflow_type]

    def list_by_backend(self, backend: str) -> List[WorkflowRegistryEntry]:
        return [e for e in self.list() if e.backend_dependency == backend]
