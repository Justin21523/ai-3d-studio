"""Symlink model directories from a source root into the configured local paths.

Usage:
    python scripts/link_models.py --source-root /mnt/c/ai_models [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Symlink models from source root")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from ai3d.registry.model_registry import ModelRegistry

    source_root = Path(args.source_root)
    registry = ModelRegistry()

    for entry in registry.list_enabled():
        target = Path(entry.local_path)
        source_name = target.name
        source = source_root / source_name

        if not source.exists():
            print(f"  [SKIP] {entry.name}: source not found at {source}")
            continue

        if args.dry_run:
            print(f"  [DRY ] {entry.name}: {source} -> {target}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()

        target.symlink_to(source)
        print(f"  [LINK] {entry.name}: {source} -> {target}")


if __name__ == "__main__":
    main()
