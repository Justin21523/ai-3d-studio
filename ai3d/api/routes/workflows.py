"""Workflow registry routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/")
async def list_workflows():
    from ai3d.registry.workflow_registry import WorkflowRegistry
    return [e.model_dump(mode="json") for e in WorkflowRegistry().list()]


@router.get("/{workflow_name}")
async def get_workflow(workflow_name: str):
    from ai3d.registry.workflow_registry import WorkflowRegistry
    try:
        return WorkflowRegistry().get(workflow_name).model_dump(mode="json")
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_name}' not found.")
