"""Asset registry routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/")
async def list_assets(backend: str | None = None, tag: str | None = None):
    from ai3d.registry.asset_registry import AssetRegistry
    registry = AssetRegistry()
    entries = registry.list()
    if backend:
        entries = [e for e in entries if e.backend_used == backend]
    if tag:
        entries = [e for e in entries if tag in e.tags]
    return [e.model_dump(mode="json") for e in entries]


@router.get("/{asset_id}")
async def get_asset(asset_id: str):
    from ai3d.registry.asset_registry import AssetRegistry
    try:
        return AssetRegistry().get(asset_id).model_dump(mode="json")
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Asset '{asset_id}' not found.")
