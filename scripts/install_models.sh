#!/usr/bin/env bash
# Download model weights for enabled backends via huggingface-cli.
# Usage: bash scripts/install_models.sh [--backend triposr|sf3d|all]

set -euo pipefail

BACKEND="${1:-all}"
MODELS_ROOT="${AI_MODELS_ROOT:-/mnt/c/ai_models}"

download() {
    local name="$1"
    local repo="$2"
    local dest="$3"

    echo ""
    echo "=== Downloading $name from $repo ==="
    mkdir -p "$dest"
    huggingface-cli download "$repo" --local-dir "$dest"
    echo "=== Done: $name -> $dest ==="
}

if [[ "$BACKEND" == "all" || "$BACKEND" == "triposr" ]]; then
    download "TripoSR" "stabilityai/TripoSR" "${MODELS_ROOT}/vision/triposr"
fi

if [[ "$BACKEND" == "all" || "$BACKEND" == "sf3d" ]]; then
    download "Stable Fast 3D" "stabilityai/stable-fast-3d" "${MODELS_ROOT}/vision/sf3d"
fi

echo ""
echo "All downloads complete. Run: python scripts/check_environment.py"
