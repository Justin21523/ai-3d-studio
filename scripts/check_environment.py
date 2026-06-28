"""Environment sanity checker — GPU, models, Blender, ComfyUI."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def check_torch():
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_ok else "N/A"
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if cuda_ok else 0
        print(f"  [{'OK' if cuda_ok else 'WARN'}] PyTorch {torch.__version__} | CUDA: {cuda_ok} | "
              f"GPU: {device_name} | VRAM: {vram:.1f} GB")
        return cuda_ok
    except ImportError:
        print("  [FAIL] PyTorch not installed")
        return False


def check_models():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ai3d.registry.model_registry import ModelRegistry
    registry = ModelRegistry()
    for entry in registry.list_enabled():
        p = Path(entry.local_path)
        status = "OK  " if p.exists() else "MISS"
        print(f"  [{status}] {entry.name:20s} -> {p}")


def check_blender():
    from ai3d.blender.bridge import BlenderBridge
    bridge = BlenderBridge()
    available = bridge.is_available()
    version = bridge.get_version() if available else "not found"
    print(f"  [{'OK' if available else 'MISS'}] Blender: {version} at {bridge._blender_exe}")


def check_comfyui():
    from ai3d.comfyui.client import ComfyUIClient
    client = ComfyUIClient()
    available, reason = client.health_check()
    print(f"  [{'OK' if available else 'MISS'}] ComfyUI at {client.base_url}: "
          f"{'reachable' if available else reason}")


def check_packages():
    packages = ["trimesh", "rembg", "PIL", "requests", "yaml", "pydantic"]
    for pkg in packages:
        try:
            __import__(pkg if pkg != "PIL" else "PIL.Image")
            print(f"  [OK  ] {pkg}")
        except ImportError:
            print(f"  [MISS] {pkg}")


def main():
    print("\n=== AI 3D Studio — Environment Check ===\n")

    print("[ PyTorch / CUDA ]")
    check_torch()

    print("\n[ Python packages ]")
    check_packages()

    print("\n[ Model weights ]")
    check_models()

    print("\n[ Blender ]")
    check_blender()

    print("\n[ ComfyUI ]")
    check_comfyui()

    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
