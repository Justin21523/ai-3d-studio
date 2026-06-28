"""Export JSON schemas for all workflow templates.

Usage:
    python scripts/export_workflow_schema.py [--output-dir <path>]

Outputs one <workflow_name>_schema.json per registered workflow, listing:
  - workflow name, type, backend
  - required and optional __PLACEHOLDER__ tokens with their descriptions
  - example values from the workflow input_schema in workflows.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai3d.comfyui.workflow_manager import WorkflowManager
from ai3d.registry.workflow_registry import WorkflowRegistry


def build_schema(entry, manager: WorkflowManager) -> dict:
    schema: dict = {
        "name": entry.name,
        "type": entry.type,
        "backend_dependency": entry.backend_dependency,
        "template_path": entry.template_path,
        "enabled": entry.enabled,
        "placeholders": {},
        "input_schema": entry.input_schema or {},
        "output_schema": entry.output_schema or {},
    }

    # Extract placeholders from template if it exists
    template_path = Path(entry.template_path)
    if not template_path.is_absolute():
        template_path = Path(__file__).parent.parent / template_path

    if template_path.exists():
        try:
            found = manager.collect_placeholders(
                json.loads(template_path.read_text(encoding="utf-8"))
            )
            schema["placeholders"] = {
                token: entry.input_schema.get(token, "No description")
                for token in sorted(found)
            }
        except Exception as exc:
            schema["placeholders"] = {"error": str(exc)}
    else:
        schema["placeholders"] = {"warning": f"Template not found: {template_path}"}

    return schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export workflow template schemas to JSON")
    parser.add_argument(
        "--output-dir",
        default="schemas",
        metavar="PATH",
        help="Directory to write schema files into (default: schemas/)",
    )
    parser.add_argument(
        "--workflow",
        metavar="NAME",
        help="Export only this workflow (default: all)",
    )
    parser.add_argument(
        "--include-disabled",
        action="store_true",
        help="Include workflows with enabled=false",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = WorkflowRegistry()
    manager = WorkflowManager()

    try:
        entries = registry.list()
    except Exception as exc:
        print(f"ERROR: failed to load workflow registry: {exc}", file=sys.stderr)
        return 1

    if args.workflow:
        try:
            entries = [registry.get(args.workflow)]
        except KeyError:
            print(f"ERROR: workflow '{args.workflow}' not found", file=sys.stderr)
            return 1

    if not args.include_disabled:
        entries = [e for e in entries if e.enabled]

    if not entries:
        print("No workflows to export (use --include-disabled to include disabled workflows)")
        return 0

    for entry in entries:
        schema = build_schema(entry, manager)
        out_path = output_dir / f"{entry.name}_schema.json"
        out_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        print(f"  {entry.name:40s} → {out_path}")

    print(f"\nExported {len(entries)} workflow schema(s) to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
