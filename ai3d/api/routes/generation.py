"""Generation routes — single image and batch."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ai3d.core.models import GenerationRequest, StandardOutput

router = APIRouter()


class RunBackendRequest(BaseModel):
    input_image_path: str
    backend: str
    output_dir: str
    output_types: list[str] = ["glb"]
    remove_background: bool = True
    device: str = "cuda"
    seed: int | None = None


@router.post("/run")
async def run_backend(body: RunBackendRequest):
    from ai3d.backends.registry import get_default_registry
    registry = get_default_registry()
    try:
        backend = registry.get(body.backend)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Backend '{body.backend}' not found.")

    request = GenerationRequest(
        request_id=str(uuid.uuid4())[:8],
        input_image_path=body.input_image_path,
        backend=body.backend,
        output_types=[StandardOutput(t) for t in body.output_types],
        output_dir=body.output_dir,
        remove_background=body.remove_background,
        device=body.device,
        seed=body.seed,
    )
    result = backend.generate(request)
    return result.model_dump(mode="json")
