"""Path configuration loader — reads configs/paths.yaml into PathConfig."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ai3d.core.models import PathConfig
from ai3d.core.storage import read_yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "paths.yaml"
_cached: Optional[PathConfig] = None


def load_path_config(config_path: Path | None = None) -> PathConfig:
    raw = read_yaml(config_path or _DEFAULT_CONFIG_PATH)

    # Resolve blender_templates_dir to repo default if not set
    if not raw.get("blender_templates_dir"):
        raw["blender_templates_dir"] = str(_REPO_ROOT / "blender_templates")

    # Resolve asset_registry_dir to outputs sibling if not set
    if not raw.get("asset_registry_dir"):
        outputs = raw.get("outputs_root", "/mnt/data/3d-studio/outputs")
        raw["asset_registry_dir"] = str(Path(outputs).parent / "registry" / "assets")

    return PathConfig.model_validate(raw)


def get_path_config() -> PathConfig:
    global _cached
    if _cached is None:
        _cached = load_path_config()
    return _cached


def invalidate_cache() -> None:
    global _cached
    _cached = None
