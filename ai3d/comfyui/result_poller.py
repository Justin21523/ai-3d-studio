"""ComfyUI result poller and output downloader."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from ai3d.core.logging import get_logger
from ai3d.core.models import ArtifactRef, ComfyJobResult
from ai3d.core.storage import ensure_directory
from ai3d.comfyui.client import ComfyUIClient

_log = get_logger(__name__)


class ResultPoller:
    """Polls a queued ComfyUI job, downloads outputs, and returns ComfyJobResult."""

    def __init__(self, client: Optional[ComfyUIClient] = None) -> None:
        self._client = client or ComfyUIClient()

    def wait_and_download(
        self,
        prompt_id: str,
        output_dir: Path,
        timeout: int = 1800,
    ) -> ComfyJobResult:
        ensure_directory(output_dir)

        try:
            history_entry = self._client.wait_for_completion(prompt_id, timeout=timeout)
        except TimeoutError as exc:
            return ComfyJobResult(success=False, prompt_id=prompt_id, error=str(exc))
        except RuntimeError as exc:
            return ComfyJobResult(success=False, prompt_id=prompt_id, error=str(exc))
        except Exception as exc:
            return ComfyJobResult(success=False, prompt_id=prompt_id, error=f"Unexpected error: {exc}")

        file_infos = self._client.get_output_files(history_entry)
        artifacts: List[ArtifactRef] = []
        warnings: List[str] = []

        for info in file_infos:
            filename = info.get("filename", "")
            subfolder = info.get("subfolder", "")
            folder_type = info.get("type", "output")

            if not filename:
                continue

            try:
                data = self._client.download_output(filename, subfolder, folder_type)
                dest = output_dir / filename
                dest.write_bytes(data)
                artifacts.append(ArtifactRef(
                    path=str(dest),
                    kind="comfyui_output",
                    label=filename,
                    size_bytes=len(data),
                ))
                _log.info("Downloaded: %s (%d bytes)", filename, len(data))
            except Exception as exc:
                warnings.append(f"Failed to download {filename}: {exc}")

        return ComfyJobResult(
            success=bool(artifacts),
            prompt_id=prompt_id,
            output_files=artifacts,
            warnings=warnings,
        )
