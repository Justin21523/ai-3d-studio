"""BlenderBridge — launches Blender in headless mode and communicates via JSON spec file."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ai3d.core.logging import get_logger
from ai3d.core.models import BlenderRenderResult, BlenderSceneSpec
from ai3d.core.paths import blender_executable

_log = get_logger(__name__)

_RESULT_MARKER = "AI3D_RESULT:"


class BlenderBridge:
    """Launches Blender headless and runs a Python script with a JSON scene spec."""

    def __init__(self, blender_exe: Optional[Path] = None) -> None:
        self._blender_exe = blender_exe or blender_executable()

    def is_available(self) -> bool:
        if not self._blender_exe.exists():
            return False
        try:
            result = subprocess.run(
                [str(self._blender_exe), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_version(self) -> str:
        try:
            result = subprocess.run(
                [str(self._blender_exe), "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            first_line = result.stdout.strip().splitlines()[0] if result.stdout else ""
            return first_line or "unknown"
        except Exception as exc:
            return f"error: {exc}"

    def launch_headless(
        self,
        scene_spec: BlenderSceneSpec,
        script_path: Path,
        timeout: int = 600,
    ) -> BlenderRenderResult:
        if not self.is_available():
            return BlenderRenderResult(
                success=False,
                output_dir=scene_spec.output_dir,
                error=f"Blender not found at: {self._blender_exe}",
            )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as f:
            json.dump(scene_spec.model_dump(mode="json"), f, indent=2)
            spec_path = Path(f.name)

        try:
            cmd = [
                str(self._blender_exe),
                "--background",
                "--python", str(script_path),
                "--",
                "--scene-spec", str(spec_path),
            ]
            _log.info("Launching Blender: %s", " ".join(cmd[:4]))
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if proc.returncode != 0:
                _log.error("Blender exited %d:\n%s", proc.returncode, proc.stderr[-2000:])
                return BlenderRenderResult(
                    success=False,
                    output_dir=scene_spec.output_dir,
                    error=f"Blender exited with code {proc.returncode}",
                    warnings=_extract_warnings(proc.stdout + proc.stderr),
                )

            result = _parse_result_from_stdout(proc.stdout, scene_spec.output_dir)
            _log.info("Blender render complete: %s", scene_spec.output_dir)
            return result

        except subprocess.TimeoutExpired:
            return BlenderRenderResult(
                success=False,
                output_dir=scene_spec.output_dir,
                error=f"Blender timed out after {timeout}s",
            )
        except Exception as exc:
            return BlenderRenderResult(
                success=False,
                output_dir=scene_spec.output_dir,
                error=str(exc),
            )
        finally:
            spec_path.unlink(missing_ok=True)


def _parse_result_from_stdout(stdout: str, fallback_output_dir: str) -> BlenderRenderResult:
    """Extract AI3D_RESULT: JSON line from Blender script stdout."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith(_RESULT_MARKER):
            payload = line[len(_RESULT_MARKER):]
            try:
                data = json.loads(payload)
                return BlenderRenderResult.model_validate(data)
            except Exception:
                pass
    # No result marker found — treat as partial success if Blender exited 0
    return BlenderRenderResult(
        success=True,
        output_dir=fallback_output_dir,
        warnings=["Blender script did not emit AI3D_RESULT marker."],
    )


def _extract_warnings(output: str) -> list[str]:
    warnings = []
    for line in output.splitlines():
        if any(tag in line for tag in ("Warning:", "ERROR", "error:")):
            warnings.append(line.strip())
    return warnings[:20]
