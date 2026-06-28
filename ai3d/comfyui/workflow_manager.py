"""ComfyUI workflow manager — loads, validates, and fills JSON workflow templates."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

from ai3d.core.logging import get_logger

_log = get_logger(__name__)

PLACEHOLDER_PATTERN = re.compile(r"__[A-Z0-9_]+__")


class WorkflowManager:
    """Loads ComfyUI JSON workflow templates and fills __PLACEHOLDER__ tokens."""

    def load_template(self, template_path: Path) -> Dict[str, Any]:
        if not template_path.exists():
            raise FileNotFoundError(f"Workflow template not found: {template_path}")
        with open(template_path, encoding="utf-8") as f:
            return json.load(f)

    def collect_placeholders(self, graph: Any) -> Set[str]:
        """Recursively collect all __PLACEHOLDER__ tokens in the graph."""
        found: Set[str] = set()
        self._walk(graph, found)
        return found

    def _walk(self, node: Any, found: Set[str]) -> None:
        if isinstance(node, str):
            for match in PLACEHOLDER_PATTERN.findall(node):
                found.add(match)
        elif isinstance(node, dict):
            for v in node.values():
                self._walk(v, found)
        elif isinstance(node, list):
            for item in node:
                self._walk(item, found)

    def fill_placeholders(
        self,
        graph: Any,
        replacements: Dict[str, Any],
    ) -> Any:
        """Return a deep copy of the graph with all placeholder tokens replaced."""
        graph = copy.deepcopy(graph)
        return self._replace(graph, replacements)

    def _replace(self, node: Any, replacements: Dict[str, Any]) -> Any:
        if isinstance(node, str):
            for key, value in replacements.items():
                token = f"__{key.upper()}__"
                if token in node:
                    if node == token:
                        return value  # full replacement — preserve type (int, float…)
                    node = node.replace(token, str(value))
            return node
        if isinstance(node, dict):
            return {k: self._replace(v, replacements) for k, v in node.items()}
        if isinstance(node, list):
            return [self._replace(item, replacements) for item in node]
        return node

    def strip_metadata(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in graph.items() if not str(k).startswith("_")}

    def validate(
        self,
        template_path: Path,
        required_placeholders: List[str],
    ) -> Dict[str, Any]:
        """Validate that all required placeholders exist in the template."""
        try:
            graph = self.load_template(template_path)
        except Exception as exc:
            return {"valid": False, "errors": [str(exc)], "warnings": [], "found": []}

        found = self.collect_placeholders(graph)
        errors: List[str] = []
        warnings: List[str] = []

        for req in required_placeholders:
            token = f"__{req.upper()}__"
            if token not in found:
                errors.append(f"Required placeholder {token} not found in template.")

        for token in found:
            upper = token.lstrip("_").rstrip("_")
            if upper not in [r.upper() for r in required_placeholders]:
                warnings.append(f"Template contains unreferenced placeholder: {token}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "found": sorted(found),
        }

    def prepare(
        self,
        template_path: Path,
        replacements: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Load template and fill placeholders in one call."""
        graph = self.load_template(template_path)
        _log.debug("Filling workflow template: %s with %d replacements", template_path.name, len(replacements))
        filled = self.fill_placeholders(graph, replacements)
        return self.strip_metadata(filled)
