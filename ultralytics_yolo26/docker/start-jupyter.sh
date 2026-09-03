#!/usr/bin/env bash
set -euo pipefail

workspace="${ULTRALYTICS_YOLO26_ROOT:-${OPENCV_AMD_END2END_ROOT:-/workspace}}"
port="${JUPYTER_PORT:-8888}"
token="${JUPYTER_TOKEN:-ultralytics-yolo26}"

seed_root="/opt/ultralytics-yolo26/seed"
if [[ ! -f "$workspace/scripts/notebook_env.py" ]]; then
    echo "Expected the baked ultralytics_yolo26 workshop at $workspace." >&2
    exit 1
fi
if [[ ! -f "$seed_root/scripts/notebook_env.py" ]]; then
    echo "Image is missing the immutable workshop seed at $seed_root." >&2
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

seed_file() {
    local relative="$1"
    local source="$seed_root/$relative"
    local target
    case "$relative" in
        models/*) target="$ULTRALYTICS_YOLO26_MODEL_DIR/${relative#models/}" ;;
        output/*) target="$ULTRALYTICS_YOLO26_OUTPUT_DIR/${relative#output/}" ;;
        *) echo "Unsupported seed path: $relative" >&2; return 2 ;;
    esac
    if [[ ! -f "$target" ]]; then
        install -D -m 0644 "$source" "$target"
    fi
}

seed_file models/yolo26x.pt
seed_file models/yolo26x.onnx
while IFS= read -r -d '' source; do
    seed_file "${source#"$seed_root/"}"
done < <(find "$seed_root/models/ort-migraphx-cache" -type f -print0)
while IFS= read -r -d '' source; do
    seed_file "${source#"$seed_root/"}"
done < <(find "$seed_root/output" -type f -print0)

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
