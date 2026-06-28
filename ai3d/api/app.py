"""FastAPI application factory — Milestone 1 minimal stub.

All routes delegate to the same service layer used by the CLI.
Full API expansion is planned for Milestone 4.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from ai3d.core.logging import get_logger, setup_logging

_log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    _log.info("AI 3D Studio API starting up")
    yield
    _log.info("AI 3D Studio API shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI 3D Studio",
        version="0.1.0",
        description="Image-to-3D generation, Blender integration, video conditioning pipeline.",
        lifespan=lifespan,
    )

    from ai3d.api.routes import backends, generation, assets, workflows
    app.include_router(backends.router, prefix="/api/v1/backends", tags=["backends"])
    app.include_router(generation.router, prefix="/api/v1/generation", tags=["generation"])
    app.include_router(assets.router, prefix="/api/v1/assets", tags=["assets"])
    app.include_router(workflows.router, prefix="/api/v1/workflows", tags=["workflows"])

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    return app


app = create_app()
