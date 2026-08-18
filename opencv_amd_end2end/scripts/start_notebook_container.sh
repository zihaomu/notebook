#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_DIR="${MODEL_DIR:-$PACKAGE_ROOT/models}"
PIPELINE_IMAGE="${PIPELINE_IMAGE:-zihao/opencv-amd-end2end:rocm7.2.1}"
LLAMA_IMAGE="${LLAMA_IMAGE:-zihao/llamacpp-q8:b9766-rocm}"
NETWORK="${NETWORK:-opencv_amd_end2end}"
PIPELINE_CONTAINER="${PIPELINE_CONTAINER:-opencv_amd_end2end_notebook}"
LLAMA_CONTAINER="${LLAMA_CONTAINER:-opencv_amd_end2end_llamacpp}"
JUPYTER_PORT="${JUPYTER_PORT:-8891}"
LLAMA_PORT="${LLAMA_PORT:-8201}"
TOKEN="${TOKEN:-opencv-amd-end2end}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

for file in yolo26x.onnx yolo26x_compiled.mxr; do
    [[ -f "$MODEL_DIR/$file" ]] || {
        echo "Missing packaged YOLO model: $MODEL_DIR/$file" >&2
        exit 1
    }
done
mkdir -p "$MODEL_DIR"

docker image inspect "$PIPELINE_IMAGE" >/dev/null 2>&1 || \
    PIPELINE_IMAGE="$PIPELINE_IMAGE" \
    bash "$PACKAGE_ROOT/scripts/build_notebook_image.sh"
pipeline_image_id=$(docker image inspect -f '{{.Id}}' "$PIPELINE_IMAGE")

docker network create "$NETWORK" >/dev/null 2>&1 || true
video_gid=$(getent group video | cut -d: -f3)
render_gid=$(getent group render | cut -d: -f3)
user_id=$(id -u)
group_id=$(id -g)

# Recreate containers when their model/package mounts point somewhere else.
if docker inspect "$LLAMA_CONTAINER" >/dev/null 2>&1; then
    current_model_dir=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/models"}}{{.Source}}{{end}}{{end}}' "$LLAMA_CONTAINER")
    if [[ "$(realpath "$current_model_dir")" != "$(realpath "$MODEL_DIR")" ]]; then
        docker rm -f "$LLAMA_CONTAINER" >/dev/null
    fi
fi
if docker inspect "$PIPELINE_CONTAINER" >/dev/null 2>&1; then
    current_package_root=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}' "$PIPELINE_CONTAINER")
    current_pipeline_image_id=$(docker inspect -f '{{.Image}}' "$PIPELINE_CONTAINER")
    if [[ -z "$current_package_root" || \
          "$(realpath "$current_package_root")" != "$(realpath "$PACKAGE_ROOT")" || \
          "$current_pipeline_image_id" != "$pipeline_image_id" ]]; then
        docker rm -f "$PIPELINE_CONTAINER" >/dev/null
    fi
fi

if docker inspect "$LLAMA_CONTAINER" >/dev/null 2>&1; then
    docker start "$LLAMA_CONTAINER" >/dev/null
    docker network connect "$NETWORK" "$LLAMA_CONTAINER" >/dev/null 2>&1 || true
else
    docker run -d \
        --name "$LLAMA_CONTAINER" --restart unless-stopped \
        --device=/dev/kfd --device=/dev/dri \
        --group-add "$video_gid" --group-add "$render_gid" \
        --ipc=host --network "$NETWORK" -p "$LLAMA_PORT:8199" \
        -v "$MODEL_DIR:/models:ro" \
        "$LLAMA_IMAGE" \
        sh -lc 'echo "Waiting for Qwen GGUF files in /models ..."; while [ ! -s /models/Qwen3-VL-8B-Instruct-Q8_0.gguf ] || [ ! -s /models/mmproj-F16.gguf ]; do sleep 2; done; exec /opt/llama.cpp/build/bin/llama-server --model /models/Qwen3-VL-8B-Instruct-Q8_0.gguf --mmproj /models/mmproj-F16.gguf --host 0.0.0.0 --port 8199 --device ROCm0 --n-gpu-layers 99 --ctx-size 12288 --parallel 3 --flash-attn auto --image-min-tokens 1024' >/dev/null
fi

if docker inspect "$PIPELINE_CONTAINER" >/dev/null 2>&1; then
    docker start "$PIPELINE_CONTAINER" >/dev/null
    docker network connect "$NETWORK" "$PIPELINE_CONTAINER" >/dev/null 2>&1 || true
else
    docker run -d \
        --name "$PIPELINE_CONTAINER" --restart unless-stopped \
        --user "$user_id:$group_id" -e HOME=/tmp \
        --device=/dev/kfd --device=/dev/dri \
        --group-add "$video_gid" --group-add "$render_gid" \
        --ipc=host --network "$NETWORK" -p "$JUPYTER_PORT:8888" \
        -e OPENCV_AMD_END2END_ROOT=/workspace \
        -e OPENCV_AMD_END2END_MODEL_DIR=/workspace/models \
        -e OPENCV_AMD_END2END_OUTPUT_DIR=/workspace/output \
        -e LLAMACPP_ROOT_URL="http://$LLAMA_CONTAINER:8199" \
        -e LLAMACPP_BASE_URL="http://$LLAMA_CONTAINER:8199/v1" \
        -e HF_ENDPOINT="$HF_ENDPOINT" \
        -e JUPYTER_TOKEN="$TOKEN" \
        -v "$PACKAGE_ROOT:/workspace" \
        "$PIPELINE_IMAGE" >/dev/null
fi

if ! curl --retry 60 --retry-delay 1 --retry-all-errors \
    -fsS --max-time 2 "http://127.0.0.1:$JUPYTER_PORT/api?token=$TOKEN" \
    >/dev/null 2>&1; then
    echo "Jupyter did not become ready" >&2
    docker logs --tail 100 "$PIPELINE_CONTAINER" >&2
    exit 1
fi

if ! validation_output=$(docker exec "$PIPELINE_CONTAINER" \
    /usr/local/bin/validate-opencv-amd-image 2>&1); then
    echo "$validation_output" >&2
    exit 1
fi

echo "Jupyter: http://127.0.0.1:$JUPYTER_PORT/?token=$TOKEN"
echo "Package: $PACKAGE_ROOT"
echo "Models: $MODEL_DIR"
if [[ -f "$MODEL_DIR/Qwen3-VL-8B-Instruct-Q8_0.gguf" && -f "$MODEL_DIR/mmproj-F16.gguf" ]]; then
    echo "Qwen models are present; llama.cpp is loading or ready on port $LLAMA_PORT."
else
    echo "Qwen models are missing. Run the first model-preparation cell in either notebook."
    echo "The notebook will download them from $HF_ENDPOINT with live progress."
fi
