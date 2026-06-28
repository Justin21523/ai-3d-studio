"""ComfyUI HTTP client — queue, poll, and download workflow outputs."""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional, Tuple

import requests

from ai3d.core.logging import get_logger

_log = get_logger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:8188"
_DEFAULT_TIMEOUT = 1800
_DEFAULT_POLL_INTERVAL = 2.0


class ComfyUIClient:
    """HTTP client for the ComfyUI server API."""

    def __init__(self, base_url: str = _DEFAULT_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._client_id = str(uuid.uuid4())

    # ── Health ────────────────────────────────────────────────────────────────

    def health_check(self) -> Tuple[bool, Optional[str]]:
        try:
            resp = requests.get(f"{self.base_url}/system_stats", timeout=5)
            if resp.status_code == 200:
                return True, None
            return False, f"HTTP {resp.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Connection refused — is ComfyUI running?"
        except Exception as exc:
            return False, str(exc)

    def get_server_stats(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/system_stats", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def get_object_info(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/object_info", timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ── Queue ─────────────────────────────────────────────────────────────────

    def queue_prompt(self, prompt_graph: Dict[str, Any]) -> str:
        """Submit a workflow graph to ComfyUI. Returns prompt_id."""
        payload = {
            "prompt": prompt_graph,
            "client_id": self._client_id,
        }
        resp = requests.post(
            f"{self.base_url}/prompt",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        prompt_id: str = data["prompt_id"]
        _log.info("ComfyUI: queued prompt %s", prompt_id)
        return prompt_id

    # ── History / Polling ─────────────────────────────────────────────────────

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def wait_for_completion(
        self,
        prompt_id: str,
        timeout: int = _DEFAULT_TIMEOUT,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
    ) -> Dict[str, Any]:
        """Poll until the prompt finishes or timeout is reached."""
        deadline = time.monotonic() + timeout
        _log.info("ComfyUI: waiting for prompt %s (timeout=%ds)", prompt_id, timeout)

        while time.monotonic() < deadline:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("completed") or status.get("status_str") == "success":
                    _log.info("ComfyUI: prompt %s completed", prompt_id)
                    return entry
                if status.get("status_str") in ("error", "failed"):
                    raise RuntimeError(
                        f"ComfyUI prompt {prompt_id} failed: {status}"
                    )
            time.sleep(poll_interval)

        raise TimeoutError(
            f"ComfyUI prompt {prompt_id} did not complete within {timeout}s"
        )

    # ── Output download ───────────────────────────────────────────────────────

    def download_output(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> bytes:
        params = {"filename": filename, "type": folder_type}
        if subfolder:
            params["subfolder"] = subfolder
        resp = requests.get(f"{self.base_url}/view", params=params, timeout=60)
        resp.raise_for_status()
        return resp.content

    def get_output_files(self, history_entry: Dict[str, Any]) -> list[Dict[str, Any]]:
        """Extract file info records from a completed history entry."""
        files = []
        outputs = history_entry.get("outputs", {})
        for _node_id, node_output in outputs.items():
            for container in (node_output, node_output.get("ui", {})):
                for key in ("images", "videos", "gifs", "files", "3d"):
                    for item in container.get(key, []):
                        files.append(item)
        return files
