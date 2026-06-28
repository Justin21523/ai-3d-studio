"""Tests for BackendRegistry."""
from __future__ import annotations

import pytest
from ai3d.backends.registry import BackendRegistry, get_default_registry
from ai3d.core.models import (
    AvailabilityResult,
    BackendMetadata,
    BackendStatus,
    BaseBackend,
    GenerationRequest,
    GenerationResult,
    ProviderCapability,
    ResourceEstimate,
    StandardOutput,
)


class _StubBackend(BaseBackend):
    name = "stub_test"

    def check_availability(self) -> AvailabilityResult:
        return AvailabilityResult(available=True, backend=self.name)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(success=True, provider=self.name, task_type="test")

    def estimate_requirements(self) -> ResourceEstimate:
        return ResourceEstimate(vram_gb=1.0, ram_gb=2.0, estimated_seconds=1.0)

    def export_metadata(self) -> BackendMetadata:
        return BackendMetadata(
            name="Stub", version="0.0", source_repo="test/stub",
            supported_output_types=[StandardOutput.GLB], required_vram_gb=1.0,
        )


def test_register_and_get():
    reg = BackendRegistry()
    stub = _StubBackend()
    reg.register(stub)
    assert reg.get("stub_test") is stub


def test_get_missing_raises():
    reg = BackendRegistry()
    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_list_names():
    reg = BackendRegistry()
    reg.register(_StubBackend())
    assert "stub_test" in reg.list_names()


def test_capabilities_returns_dict():
    reg = BackendRegistry()
    reg.register(_StubBackend())
    caps = reg.capabilities()
    assert "stub_test" in caps
    assert caps["stub_test"]["available"] is True


def test_check_all():
    reg = BackendRegistry()
    reg.register(_StubBackend())
    results = reg.check_all()
    assert results["stub_test"].available is True


def test_default_registry_has_scaffold_stubs():
    reg = get_default_registry()
    names = reg.list_names()
    for expected in ("trellis", "hunyuan3d", "instantmesh", "crm", "wonder3d", "mesh2splat"):
        assert expected in names, f"Expected scaffold '{expected}' in default registry"


def test_scaffold_stubs_unavailable():
    reg = get_default_registry()
    for stub_name in ("trellis", "hunyuan3d", "crm"):
        backend = reg.get(stub_name)
        avail = backend.check_availability()
        assert not avail.available
        cap = backend.get_capabilities()
        assert cap.status == BackendStatus.SCAFFOLD
