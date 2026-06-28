# Architecture

## Design Principles

- **Feature-based package hierarchy** — each domain (backends, blender, comfyui, video…) is a self-contained package.
- **Strong abstract interfaces** — `BaseBackend`, `BasePreprocessor`, `BasePostprocessor` define strict contracts. All new backends and processors must implement these.
- **Pydantic v2 throughout** — every request, result, and configuration object is a `BaseModel`. Round-trip via `.model_dump(mode="json")` / `.model_validate()`.
- **YAML manifest persistence** — every pipeline run writes a `pipeline_manifest.yaml` to disk for reproducibility and debugging.
- **Config-driven paths** — all filesystem paths come from `configs/paths.yaml`. Nothing is hardcoded.
- **CLI as primary interface** — all functionality is accessible via `ai3d <command>`. The FastAPI layer wraps the same service layer.

## Package Map

```
ai3d/
├── core/
│   ├── models.py       ← All Pydantic models + ABC interfaces (single source of truth)
│   ├── storage.py      ← write_model / read_model / write_yaml / read_yaml
│   ├── paths.py        ← Repo-relative constants + config-resolved path functions
│   ├── logging.py      ← setup_logging / get_logger
│   └── config.py       ← load_path_config() → PathConfig
│
├── backends/
│   ├── base.py         ← re-exports BaseBackend
│   ├── registry.py     ← BackendRegistry + get_default_registry()
│   └── <name>/
│       ├── loader.py   ← weight resolution + lazy model loading
│       └── backend.py  ← Concrete BaseBackend implementation
│
├── preprocessors/      ← Image prep before 3D generation
├── postprocessors/     ← Mesh processing after generation
├── blender/            ← Headless Blender subprocess bridge
├── comfyui/            ← ComfyUI HTTP client + workflow template filler
├── video/              ← Turntable export + conditioning pack builder
├── registry/           ← YAML-backed asset / model / workflow registries
├── pipeline/           ← 11-stage orchestration + batch runner
├── cli/                ← argparse CLI (build_parser + main dispatch)
└── api/                ← FastAPI factory (Milestone 4)
```

## Data Flow

```
GenerationRequest
    │ backend.generate()
    ▼
GenerationResult { artifacts: [ArtifactRef, ...] }
    │
    ├── MeshCleaner.clean()        → cleaned mesh on disk
    ├── UVUnwrapper.process()      → UV mesh on disk
    ├── TurntableRenderer.render() → frame sequence on disk
    ├── TurntableExporter.export() → VideoConditioningPack
    └── AssetRegistry.register()   → asset_{id}.yaml
    │
    ▼
PipelineManifest (persisted as pipeline_manifest.yaml)
```

## Extension Points

### Adding a new backend

1. Create `ai3d/backends/<name>/loader.py` and `backend.py`.
2. Implement all four abstract methods from `BaseBackend`.
3. Register in `ai3d/backends/registry.py` `_build_default_registry()`.
4. Add entry to `configs/models.yaml`.

### Adding a new preprocessor

1. Create `ai3d/preprocessors/<name>.py`.
2. Subclass `BasePreprocessor` and implement `process()`.
3. Wire into `GenerationPipeline` stage 2 or 3 as needed.

### Adding a new ComfyUI workflow

1. Export the workflow from ComfyUI in API format.
2. Replace variable values with `__PLACEHOLDER__` tokens.
3. Save as `workflows/<name>.json`.
4. Add entry to `configs/workflows.yaml`.
5. Use `ai3d run-workflow --workflow <name>`.
