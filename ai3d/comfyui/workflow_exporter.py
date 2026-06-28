"""ComfyUI workflow export helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ai3d.comfyui.workflow_manager import WorkflowManager


def _node_input_type(class_info: Dict[str, Any], input_name: str) -> str:
    for group in ("required", "optional"):
        spec = class_info.get("input", {}).get(group, {}).get(input_name)
        if spec:
            first = spec[0]
            if isinstance(first, str):
                return first
            return "COMBO"
    return "*"


def _node_outputs(class_info: Dict[str, Any], link_map: Dict[int, list[int]]) -> list[Dict[str, Any]]:
    output_types = class_info.get("output") or []
    output_names = class_info.get("output_name") or output_types
    outputs: list[Dict[str, Any]] = []
    for index, output_type in enumerate(output_types):
        outputs.append({
            "name": output_names[index] if index < len(output_names) else str(output_type),
            "type": output_type,
            "links": link_map.get(index) or None,
        })
    return outputs


def api_workflow_to_ui(api_graph: Dict[str, Any], object_info: Dict[str, Any] | None = None) -> Dict[str, Any]:
    graph = WorkflowManager().strip_metadata(api_graph)
    node_ids = list(graph.keys())
    id_map = {node_id: int(node_id) if str(node_id).isdigit() else index for index, node_id in enumerate(node_ids, 1)}
    link_id = 0
    links: list[list[Any]] = []
    node_input_slots: Dict[str, list[Dict[str, Any]]] = {node_id: [] for node_id in node_ids}
    output_link_map: Dict[str, Dict[int, list[int]]] = {node_id: {} for node_id in node_ids}

    for target_id, node_data in graph.items():
        class_info = (object_info or {}).get(node_data.get("class_type", ""), {})
        for input_name, value in node_data.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in id_map:
                source_id = str(value[0])
                source_slot = int(value[1])
                target_slot = len(node_input_slots[target_id])
                input_type = _node_input_type(class_info, input_name)
                link_id += 1
                links.append([link_id, id_map[source_id], source_slot, id_map[target_id], target_slot, input_type])
                node_input_slots[target_id].append({
                    "name": input_name,
                    "type": input_type,
                    "link": link_id,
                })
                output_link_map[source_id].setdefault(source_slot, []).append(link_id)

    nodes: list[Dict[str, Any]] = []
    for order, node_id in enumerate(node_ids):
        node_data = graph[node_id]
        class_type = node_data.get("class_type", "Unknown")
        class_info = (object_info or {}).get(class_type, {})
        widgets_values = [
            value
            for value in node_data.get("inputs", {}).values()
            if not (isinstance(value, list) and len(value) == 2 and str(value[0]) in id_map)
        ]
        x = (order % 4) * 360
        y = (order // 4) * 280
        nodes.append({
            "id": id_map[node_id],
            "type": class_type,
            "pos": [x, y],
            "size": [300, 120],
            "flags": {},
            "order": order,
            "mode": 0,
            "inputs": node_input_slots[node_id],
            "outputs": _node_outputs(class_info, output_link_map[node_id]),
            "properties": {
                "Node name for S&R": class_type,
            },
            "widgets_values": widgets_values,
        })

    return {
        "id": "ai3d-hunyuan3d21-image-to-model",
        "revision": 0,
        "last_node_id": max(id_map.values()) if id_map else 0,
        "last_link_id": link_id,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 1, "offset": [0, 0]}},
        "version": 0.4,
    }


def export_ui_workflow(
    api_graph: Dict[str, Any],
    output_path: Path,
    object_info: Dict[str, Any] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ui_graph = api_workflow_to_ui(api_graph, object_info=object_info)
    output_path.write_text(json.dumps(ui_graph, indent=2), encoding="utf-8")
    return output_path
