#!/usr/bin/env bash
set -euo pipefail

workspace="${ULTRALYTICS_YOLO26_ROOT:-${OPENCV_AMD_END2END_ROOT:-/workspace}}"
port="${JUPYTER_PORT:-8888}"
token="${JUPYTER_TOKEN:-ultralytics-yolo26}"

if [[ ! -f "$workspace/scripts/notebook_env.py" ]]; then
    echo "Expected the ultralytics_yolo26 workshop at $workspace." >&2
    echo "Mount it with: -v /path/to/ultralytics_yolo26:/workspace" >&2
    exit 1
fi

export ULTRALYTICS_YOLO26_ROOT="$workspace"
export ULTRALYTICS_YOLO26_MODEL_DIR="${ULTRALYTICS_YOLO26_MODEL_DIR:-$workspace/models}"
export ULTRALYTICS_YOLO26_OUTPUT_DIR="${ULTRALYTICS_YOLO26_OUTPUT_DIR:-$workspace/output}"
export ULTRALYTICS_MIGRAPHX_CACHE_ROOT="${ULTRALYTICS_MIGRAPHX_CACHE_ROOT:-$ULTRALYTICS_YOLO26_MODEL_DIR/ort-migraphx-cache}"

# Keep aliases while the copied OpenCV-first scripts are migrated.
export OPENCV_AMD_END2END_ROOT="$workspace"
export OPENCV_AMD_END2END_MODEL_DIR="$ULTRALYTICS_YOLO26_MODEL_DIR"
export OPENCV_AMD_END2END_OUTPUT_DIR="$ULTRALYTICS_YOLO26_OUTPUT_DIR"

mkdir -p "$ULTRALYTICS_YOLO26_MODEL_DIR" "$ULTRALYTICS_YOLO26_OUTPUT_DIR" \
    "$ULTRALYTICS_MIGRAPHX_CACHE_ROOT"
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
