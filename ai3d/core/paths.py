"""Canonical path helpers — repo-relative constants and config-resolved directories."""
from __future__ import annotations

from pathlib import Path

# Repo-relative constants (always valid regardless of environment)
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_ROOT = REPO_ROOT / "configs"
WORKFLOWS_ROOT = REPO_ROOT / "workflows"
BLENDER_SCRIPTS_DIR = REPO_ROOT / "ai3d" / "blender" / "scripts"
BLENDER_TEMPLATES_ROOT = REPO_ROOT / "blender_templates"


def _cfg():
    from ai3d.core.config import get_path_config
    return get_path_config()


def outputs_root() -> Path:
    return Path(_cfg().outputs_root)


def cache_root() -> Path:
    return Path(_cfg().cache_root)


def previews_root() -> Path:
    return Path(_cfg().previews_root)


def asset_registry_dir() -> Path:
    return Path(_cfg().asset_registry_dir)


def blender_executable() -> Path:
    return Path(_cfg().blender_executable)


def blender_templates_dir() -> Path:
    d = _cfg().blender_templates_dir
    return Path(d) if d else BLENDER_TEMPLATES_ROOT


def comfyui_root() -> Path:
    return Path(_cfg().comfyui_root)


def comfyui_input_dir() -> Path:
    return Path(_cfg().comfyui_input_dir)


def ai_models_root() -> Path:
    return Path(_cfg().ai_models_root)


def ai_tools_root() -> Path:
    return Path(_cfg().ai_tools_root)
