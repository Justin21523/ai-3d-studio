"""Build the static portfolio demo site into a GitHub Pages-ready folder."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "demo"


def build(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for path in DEMO_DIR.iterdir():
        if path.is_file():
            shutil.copy2(path, out_dir / path.name)
        elif path.is_dir() and path.name == "media":
            shutil.copytree(path, out_dir / path.name)

    (out_dir / ".nojekyll").write_text("", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="_site", help="Output directory")
    args = parser.parse_args()
    build((ROOT / args.out).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
