"""Backend inspection routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/")
async def list_backends():
    from ai3d.backends.registry import get_default_registry
    return get_default_registry().capabilities()


@router.get("/{backend_name}/check")
async def check_backend(backend_name: str):
    from ai3d.backends.registry import get_default_registry
    registry = get_default_registry()
    try:
        backend = registry.get(backend_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Backend '{backend_name}' not found.")
    return {
        "availability": backend.check_availability().model_dump(mode="json"),
        "requirements": backend.estimate_requirements().model_dump(mode="json"),
    }
