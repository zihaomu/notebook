#!/usr/bin/env bash
set -euo pipefail

workspace="${OPENCV_AMD_END2END_ROOT:-/workspace}"
port="${JUPYTER_PORT:-8888}"
token="${JUPYTER_TOKEN:-opencv-amd-end2end}"

if [[ ! -f "$workspace/scripts/notebook_env.py" ]]; then
    echo "Expected the opencv_amd_end2end package at $workspace." >&2
    echo "Mount it with: -v /path/to/opencv_amd_end2end:/workspace" >&2
    exit 1
fi

export OPENCV_AMD_END2END_ROOT="$workspace"
export OPENCV_AMD_END2END_MODEL_DIR="${OPENCV_AMD_END2END_MODEL_DIR:-$workspace/models}"
export OPENCV_AMD_END2END_OUTPUT_DIR="${OPENCV_AMD_END2END_OUTPUT_DIR:-$workspace/output}"

mkdir -p "$OPENCV_AMD_END2END_MODEL_DIR" "$OPENCV_AMD_END2END_OUTPUT_DIR"
cd "$workspace"

exec /opt/venv/bin/jupyter-lab \
    --allow-root \
    --ip=0.0.0.0 \
    --port="$port" \
    --no-browser \
    --ServerApp.root_dir="$workspace" \
    --ServerApp.preferred_dir="$workspace" \
    --ServerApp.default_url=/lab \
    --ServerApp.allow_remote_access=True \
    --IdentityProvider.token="$token"
