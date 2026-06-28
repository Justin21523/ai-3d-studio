"""
Core Pydantic models and ABC interfaces for ai-3d-studio.

All extensible provider/processor patterns use ABC base classes.
All data transfer uses Pydantic v2 BaseModel with .model_dump(mode="json")
and .model_validate() for round-trip YAML/JSON serialization.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Type, TypeVar

from pydantic import BaseModel, Field


T = TypeVar("T", bound=BaseModel)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class StandardOutput(str, Enum):
    """Canonical output type identifiers for 3D generation pipelines."""
    DRAFT_MESH       = "draft_mesh"
    CLEANED_MESH     = "cleaned_mesh"
    TEXTURED_MESH    = "textured_mesh"
    GLB              = "glb"
    FBX              = "fbx"
    OBJ              = "obj"
    SPLAT_PLY        = "splat_ply"
    SPLAT_KSPLAT     = "splat_ksplat"
    MULTIVIEW_IMAGES = "multiview_images"
    NORMAL_MAPS      = "normal_maps"
    DEPTH_MAPS       = "depth_maps"


class PipelineStage(str, Enum):
    """Ordered stages of the full generation pipeline."""
    INPUT_INGEST       = "input_ingest"
    BACKGROUND_REMOVAL = "background_removal"
    QUALITY_CHECK      = "quality_check"
    MULTIVIEW_GEN      = "multiview_gen"
    GENERATION_3D      = "generation_3d"
    MESH_CLEANUP       = "mesh_cleanup"
    UV_TEXTURE         = "uv_texture"
    MESH_TO_SPLAT      = "mesh_to_splat"
    BLENDER_RENDER     = "blender_render"
    VIDEO_CONDITIONING = "video_conditioning"
    ASSET_REGISTRATION = "asset_registration"


class BackendStatus(str, Enum):
    READY       = "ready"
    SCAFFOLD    = "scaffold"
    UNAVAILABLE = "unavailable"
    ERROR       = "error"


class AssetStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    COMPLETE   = "complete"
    FAILED     = "failed"


class BlenderRenderPass(str, Enum):
    RGB    = "rgb"
    DEPTH  = "depth"
    NORMAL = "normal"
    MASK   = "mask"


# ─────────────────────────────────────────────
# Shared Result Primitives
# ─────────────────────────────────────────────

class ArtifactRef(BaseModel):
    """One produced or referenced file artifact."""
    path: str
    kind: str = "file"
    label: Optional[str] = None
    output_type: Optional[StandardOutput] = None
    size_bytes: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    """Normalized generation or task execution result."""
    success: bool
    provider: str
    task_type: str
    created_at: str = Field(default_factory=utc_now_iso)
    artifacts: List[ArtifactRef] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class ProviderCapability(BaseModel):
    """Capability metadata for a backend provider."""
    provider: str
    label: str
    available: bool = True
    enabled: bool = True
    status: BackendStatus = BackendStatus.READY
    supports_textured_mesh: bool = False
    supports_glb: bool = False
    supports_splat: bool = False
    supports_multiview: bool = False
    supports_normal_maps: bool = False
    supports_depth_maps: bool = False
    supported_output_types: List[StandardOutput] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Backend Request / Response Models
# ─────────────────────────────────────────────

class GenerationRequest(BaseModel):
    """Provider-agnostic 3D generation request."""
    request_id: str
    input_image_path: str
    backend: str
    output_types: List[StandardOutput] = Field(
        default_factory=lambda: [StandardOutput.GLB]
    )
    output_dir: str
    remove_background: bool = True
    run_quality_check: bool = True
    seed: Optional[int] = None
    device: str = "cuda"
    backend_params: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResourceEstimate(BaseModel):
    """Estimated resource requirements for a backend run."""
    vram_gb: float
    ram_gb: float
    estimated_seconds: float
    supports_cpu_fallback: bool = False
    notes: List[str] = Field(default_factory=list)


class BackendMetadata(BaseModel):
    """Exportable backend identity record."""
    name: str
    version: str
    source_repo: str
    modality: str = "image-to-3d"
    supported_output_types: List[StandardOutput] = Field(default_factory=list)
    required_vram_gb: float
    notes: List[str] = Field(default_factory=list)


class AvailabilityResult(BaseModel):
    """Result of a backend availability check."""
    available: bool
    backend: str
    reason: Optional[str] = None
    checked_at: str = Field(default_factory=utc_now_iso)
    model_paths_found: List[str] = Field(default_factory=list)
    model_paths_missing: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────
# Preprocessor Models
# ─────────────────────────────────────────────

class PreprocessRequest(BaseModel):
    input_path: str
    output_dir: str
    target_size: Optional[tuple[int, int]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PreprocessResult(BaseModel):
    success: bool
    output_path: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QualityCheckResult(BaseModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    has_alpha: bool = False
    estimated_subject_coverage: float = 0.0


# ─────────────────────────────────────────────
# Postprocessor Models
# ─────────────────────────────────────────────

class MeshCleanRequest(BaseModel):
    input_path: str
    output_path: str
    remove_degenerate_faces: bool = True
    make_watertight: bool = False
    remove_duplicate_vertices: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MeshCleanResult(BaseModel):
    success: bool
    output_path: Optional[str] = None
    original_face_count: int = 0
    cleaned_face_count: int = 0
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class DecimateRequest(BaseModel):
    input_path: str
    output_path: str
    target_face_count: Optional[int] = None
    target_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FormatConvertRequest(BaseModel):
    input_path: str
    output_path: str
    output_format: Literal["glb", "obj", "fbx", "ply", "stl"]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Blender Models
# ─────────────────────────────────────────────

class TurntableSpec(BaseModel):
    """Parameters for a turntable render job."""
    asset_path: str
    output_dir: str
    frame_count: int = 72
    fps: int = 24
    resolution_x: int = 1024
    resolution_y: int = 1024
    camera_elevation_deg: float = 20.0
    camera_distance: float = 2.5
    render_passes: List[BlenderRenderPass] = Field(
        default_factory=lambda: [BlenderRenderPass.RGB]
    )
    hdri_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BlenderRenderResult(BaseModel):
    success: bool
    output_dir: str
    frame_paths: List[str] = Field(default_factory=list)
    pass_dirs: Dict[str, str] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BlenderSceneSpec(BaseModel):
    """Full scene specification sent to headless Blender via JSON stdin."""
    operation: Literal["turntable", "passes", "import_mesh", "custom"]
    mesh_path: str
    output_dir: str
    turntable: Optional[TurntableSpec] = None
    passes: List[BlenderRenderPass] = Field(default_factory=list)
    script_overrides: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# ComfyUI Models
# ─────────────────────────────────────────────

class ComfyWorkflowSpec(BaseModel):
    """A resolved and filled ComfyUI workflow ready for queuing."""
    workflow_name: str
    template_path: str
    filled_graph: Dict[str, Any] = Field(default_factory=dict)
    placeholders_used: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ComfyJobResult(BaseModel):
    success: bool
    prompt_id: Optional[str] = None
    output_files: List[ArtifactRef] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Video Conditioning Models
# ─────────────────────────────────────────────

class VideoConditioningPack(BaseModel):
    """A bundle of conditioning assets for downstream video generation."""
    pack_id: str
    source_asset_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    rgb_sequence_dir: Optional[str] = None
    depth_sequence_dir: Optional[str] = None
    normal_sequence_dir: Optional[str] = None
    mask_sequence_dir: Optional[str] = None
    frame_count: int = 0
    fps: int = 24
    resolution_x: int = 1024
    resolution_y: int = 1024
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Registry Models
# ─────────────────────────────────────────────

class ModelRegistryEntry(BaseModel):
    """One entry from configs/models.yaml."""
    name: str
    source_repo: str
    local_path: str
    modality: str = "image-to-3d"
    backend_adapter: str
    format: str = "safetensors"
    required_vram_gb: float
    notes: List[str] = Field(default_factory=list)
    enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowRegistryEntry(BaseModel):
    """One entry from configs/workflows.yaml."""
    name: str
    type: str
    backend_dependency: str
    template_path: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Dict[str, Any] = Field(default_factory=dict)
    notes: List[str] = Field(default_factory=list)
    enabled: bool = True


class AssetRegistryEntry(BaseModel):
    """One registered 3D asset in the local catalog."""
    asset_id: str
    label: str
    source_image_path: str
    backend_used: str
    output_paths: Dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)
    status: AssetStatus = AssetStatus.COMPLETE
    pipeline_manifest_path: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Pipeline Models
# ─────────────────────────────────────────────

class PipelineManifest(BaseModel):
    """Full reproducible manifest for one generation run."""
    run_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    request: GenerationRequest
    stages_completed: List[PipelineStage] = Field(default_factory=list)
    stages_skipped: List[PipelineStage] = Field(default_factory=list)
    stages_failed: List[PipelineStage] = Field(default_factory=list)
    artifacts: List[ArtifactRef] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    final_result: Optional[GenerationResult] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchManifest(BaseModel):
    """Manifest for a batch generation job."""
    batch_id: str
    created_at: str = Field(default_factory=utc_now_iso)
    input_images: List[str] = Field(default_factory=list)
    backend: str
    output_types: List[StandardOutput] = Field(default_factory=list)
    output_root: str
    runs: List[str] = Field(default_factory=list)
    completed: int = 0
    failed: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Path Configuration Model
# ─────────────────────────────────────────────

class PathConfig(BaseModel):
    """Canonical path configuration loaded from configs/paths.yaml."""
    ai_models_root: str = "/mnt/c/ai_models"
    ai_tools_root: str = "/mnt/c/ai_tools"
    data_root: str = "/mnt/data"
    outputs_root: str = "/mnt/data/3d-studio/outputs"
    cache_root: str = "/mnt/c/ai_cache/3d-studio"
    previews_root: str = "/mnt/data/3d-studio/previews"
    blender_executable: str = "/usr/bin/blender"
    blender_templates_dir: str = ""
    comfyui_root: str = "/mnt/c/ai_tools/comfyui"
    comfyui_input_dir: str = "/mnt/c/ai_tools/comfyui/input"
    comfyui_base_url: str = "http://127.0.0.1:8188"
    asset_registry_dir: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# ABC Interfaces
# ─────────────────────────────────────────────

class BaseBackend(ABC):
    """Abstract base class for all 3D generation backends."""

    name: str

    @abstractmethod
    def check_availability(self) -> AvailabilityResult:
        """Check that model weights exist and runtime is usable."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run 3D generation for one input image."""

    @abstractmethod
    def estimate_requirements(self) -> ResourceEstimate:
        """Return static resource estimates for this backend."""

    @abstractmethod
    def export_metadata(self) -> BackendMetadata:
        """Return identity/version metadata for this backend."""

    def get_capabilities(self) -> ProviderCapability:
        """Return capability metadata. Override for dynamic status."""
        meta = self.export_metadata()
        avail = self.check_availability()
        return ProviderCapability(
            provider=self.name,
            label=meta.name,
            available=avail.available,
            status=BackendStatus.READY if avail.available else BackendStatus.UNAVAILABLE,
            supported_output_types=meta.supported_output_types,
            metadata={"source_repo": meta.source_repo},
        )


class BasePreprocessor(ABC):
    """Abstract base class for image preprocessors."""

    name: str

    @abstractmethod
    def process(self, request: PreprocessRequest) -> PreprocessResult:
        """Apply this preprocessing step to an input image."""


class BasePostprocessor(ABC):
    """Abstract base class for mesh/asset postprocessors."""

    name: str

    @abstractmethod
    def process(
        self,
        input_path: Path,
        output_path: Path,
        **kwargs: Any,
    ) -> GenerationResult:
        """Apply this postprocessing step."""
