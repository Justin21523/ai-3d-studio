"""BackendRegistry — runtime Python registry for all 3D generation backends."""
from __future__ import annotations

from typing import Dict, List

from ai3d.core.logging import get_logger
from ai3d.core.models import AvailabilityResult, BaseBackend, ProviderCapability

_log = get_logger(__name__)


class BackendRegistry:
    def __init__(self) -> None:
        self._backends: Dict[str, BaseBackend] = {}

    def register(self, backend: BaseBackend) -> None:
        self._backends[backend.name] = backend
        _log.debug("Registered backend: %s", backend.name)

    def get(self, name: str) -> BaseBackend:
        if name not in self._backends:
            raise KeyError(f"Backend '{name}' not found. Available: {self.list_names()}")
        return self._backends[name]

    def list_names(self) -> List[str]:
        return sorted(self._backends.keys())

    def capabilities(self) -> Dict[str, dict]:
        result: Dict[str, dict] = {}
        for name, backend in self._backends.items():
            try:
                cap: ProviderCapability = backend.get_capabilities()
                result[name] = cap.model_dump(mode="json")
            except Exception as exc:
                _log.warning("capabilities() failed for %s: %s", name, exc)
                result[name] = {"provider": name, "available": False, "error": str(exc)}
        return result

    def check_all(self) -> Dict[str, AvailabilityResult]:
        result: Dict[str, AvailabilityResult] = {}
        for name, backend in self._backends.items():
            try:
                result[name] = backend.check_availability()
            except Exception as exc:
                _log.warning("check_availability() failed for %s: %s", name, exc)
                result[name] = AvailabilityResult(
                    available=False,
                    backend=name,
                    reason=f"Exception during check: {exc}",
                )
        return result


def _build_default_registry() -> BackendRegistry:
    registry = BackendRegistry()

    # Milestone 1 — fully implemented
    try:
        from ai3d.backends.triposr.backend import TripoSRBackend
        registry.register(TripoSRBackend())
    except ImportError as exc:
        _log.debug("TripoSRBackend import skipped: %s", exc)

    try:
        from ai3d.backends.sf3d.backend import SF3DBackend
        registry.register(SF3DBackend())
    except ImportError as exc:
        _log.debug("SF3DBackend import skipped: %s", exc)

    # Scaffold stubs — registered but unavailable
    from ai3d.backends.trellis.backend import TRELLISBackend
    from ai3d.backends.hunyuan3d.backend import Hunyuan3DBackend
    from ai3d.backends.instantmesh.backend import InstantMeshBackend
    from ai3d.backends.crm.backend import CRMBackend
    from ai3d.backends.wonder3d.backend import Wonder3DBackend
    from ai3d.backends.mesh2splat.backend import Mesh2SplatBackend

    for stub in (
        TRELLISBackend(),
        Hunyuan3DBackend(),
        InstantMeshBackend(),
        CRMBackend(),
        Wonder3DBackend(),
        Mesh2SplatBackend(),
    ):
        registry.register(stub)

    return registry


_DEFAULT_REGISTRY: BackendRegistry | None = None


def get_default_registry() -> BackendRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = _build_default_registry()
    return _DEFAULT_REGISTRY
