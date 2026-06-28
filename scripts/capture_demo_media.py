"""Capture screenshots, cover, and a short demo recording for the static demo."""
from __future__ import annotations

import argparse
import http.server
import shutil
import socketserver
import sys
import threading
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT / "_site"
MEDIA_DIR = ROOT / "demo" / "media"

sys.path.insert(0, str(ROOT))
from scripts.build_static_demo import build  # noqa: E402


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return


def serve(directory: Path):
    handler = lambda *args, **kwargs: QuietHandler(*args, directory=str(directory), **kwargs)
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def capture() -> None:
    build(SITE_DIR)
    if MEDIA_DIR.exists():
        shutil.rmtree(MEDIA_DIR)
    (MEDIA_DIR / "screenshots").mkdir(parents=True)
    (MEDIA_DIR / "demo").mkdir(parents=True)

    server, base_url = serve(SITE_DIR)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                executable_path=shutil.which("chromium") or None,
            )

            page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
            page.goto(base_url, wait_until="networkidle")
            page.screenshot(path=MEDIA_DIR / "screenshots" / "01-demo-console-desktop.png", full_page=True)
            page.locator("#run-button").click()
            page.wait_for_timeout(2800)
            page.screenshot(path=MEDIA_DIR / "screenshots" / "02-demo-completed-desktop.png", full_page=True)

            mobile = browser.new_page(
                viewport={"width": 390, "height": 900},
                is_mobile=True,
                device_scale_factor=2,
            )
            mobile.goto(base_url, wait_until="networkidle")
            mobile.screenshot(path=MEDIA_DIR / "screenshots" / "03-demo-mobile.png", full_page=True)

            video_context = browser.new_context(
                viewport={"width": 1280, "height": 820},
                record_video_dir=str(MEDIA_DIR / "demo"),
                record_video_size={"width": 1280, "height": 820},
            )
            video_page = video_context.new_page()
            video_page.goto(base_url, wait_until="networkidle")
            video_page.wait_for_timeout(700)
            video_page.locator("#scenario-select").select_option("character")
            video_page.wait_for_timeout(450)
            video_page.locator("#run-button").click()
            video_page.wait_for_timeout(3200)
            video_page.locator("#scenario-select").select_option("prop")
            video_page.wait_for_timeout(650)
            video_path = video_page.video.path()
            video_context.close()
            browser.close()

        recorded = Path(video_path)
        final_video = MEDIA_DIR / "demo" / "demo-walkthrough.webm"
        recorded.replace(final_video)
        _make_cover(
            MEDIA_DIR / "screenshots" / "02-demo-completed-desktop.png",
            MEDIA_DIR / "cover.webp",
        )
    finally:
        server.shutdown()
        server.server_close()


def _make_cover(source: Path, target: Path) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size
        crop_height = int(width * 9 / 16)
        top = max((height - crop_height) // 4, 0)
        image = image.crop((0, top, width, min(top + crop_height, height)))
        image = image.resize((1600, 900))
        image.save(target, "WEBP", quality=86)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    capture()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
