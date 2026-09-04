#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_VOLUME="${MODEL_VOLUME:-ultralytics_yolo26_models}"
OUTPUT_DIR="${OUTPUT_DIR:-$PACKAGE_ROOT/output}"
PIPELINE_IMAGE="${PIPELINE_IMAGE:-zihao/ultralytics-yolo26-workshop:rocm7.2.1-full}"
LLAMA_IMAGE="${LLAMA_IMAGE:-zihao/llamacpp-q8:b9766-rocm}"
NETWORK="${NETWORK:-ultralytics_yolo26}"
PIPELINE_CONTAINER="${PIPELINE_CONTAINER:-ultralytics_yolo26_notebook}"
LLAMA_CONTAINER="${LLAMA_CONTAINER:-ultralytics_yolo26_llamacpp}"
JUPYTER_PORT="${JUPYTER_PORT:-8892}"
LLAMA_PORT="${LLAMA_PORT:-8201}"
TOKEN="${TOKEN:-ultralytics-yolo26}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PIPELINE_GPU="${PIPELINE_GPU:-0}"
LLAMA_GPU="${LLAMA_GPU:-$PIPELINE_GPU}"
VAAPI_DEVICE="${VAAPI_DEVICE:-/dev/dri/renderD128}"

mkdir -p "$OUTPUT_DIR"
# The full image contains all four model files and the matching gfx1100 cache.
# A verified named volume exposes the same immutable model set to llama.cpp.

docker image inspect "$PIPELINE_IMAGE" >/dev/null 2>&1 || \
    PIPELINE_IMAGE="$PIPELINE_IMAGE" \
    bash "$PACKAGE_ROOT/scripts/build_notebook_image.sh"
pipeline_image_id=$(docker image inspect -f '{{.Id}}' "$PIPELINE_IMAGE")

docker volume create "$MODEL_VOLUME" >/dev/null
docker run --rm \
    -v "$MODEL_VOLUME:/model-volume" \
    --entrypoint /usr/local/bin/init-ultralytics-yolo26-model-volume \
    "$PIPELINE_IMAGE" /model-volume >/dev/null
model_set_id=$(docker run --rm --entrypoint cat "$PIPELINE_IMAGE" \
    /opt/ultralytics-yolo26/models/MODEL_SET_ID)

docker network create "$NETWORK" >/dev/null 2>&1 || true
video_gid=$(getent group video | cut -d: -f3)
render_gid=$(getent group render | cut -d: -f3)
user_id=$(id -u)
group_id=$(id -g)

# Recreate containers when their data mounts or baked image identity differ.
if docker inspect "$LLAMA_CONTAINER" >/dev/null 2>&1; then
    current_model_name=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/models"}}{{.Name}}{{end}}{{end}}' "$LLAMA_CONTAINER")
    current_model_set_id=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$LLAMA_CONTAINER" | sed -n 's/^ULTRALYTICS_MODEL_SET_ID=//p')
    if [[ "$current_model_name" != "$MODEL_VOLUME" || \
          "$current_model_set_id" != "$model_set_id" ]]; then
        docker rm -f "$LLAMA_CONTAINER" >/dev/null
    fi
fi
if docker inspect "$PIPELINE_CONTAINER" >/dev/null 2>&1; then
    current_workspace_mount=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}' "$PIPELINE_CONTAINER")
    current_output_dir=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/workspace/output"}}{{.Source}}{{end}}{{end}}' "$PIPELINE_CONTAINER")
    current_pipeline_image_id=$(docker inspect -f '{{.Image}}' "$PIPELINE_CONTAINER")
    if [[ -n "$current_workspace_mount" || \
          -z "$current_output_dir" || \
          "$(realpath "$current_output_dir")" != "$(realpath "$OUTPUT_DIR")" || \
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
        -e ROCR_VISIBLE_DEVICES="$LLAMA_GPU" -e HIP_VISIBLE_DEVICES=0 \
        -e ULTRALYTICS_MODEL_SET_ID="$model_set_id" \
        -v "$MODEL_VOLUME:/models:ro" \
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
        -e ROCR_VISIBLE_DEVICES="$PIPELINE_GPU" -e HIP_VISIBLE_DEVICES=0 \
        -e VAAPI_DEVICE="$VAAPI_DEVICE" \
        -e YOLO_CONFIG_DIR=/workspace/output/.ultralytics \
        -e ULTRALYTICS_YOLO26_ROOT=/workspace \
        -e ULTRALYTICS_YOLO26_MODEL_DIR=/opt/ultralytics-yolo26/models \
        -e ULTRALYTICS_YOLO26_OUTPUT_DIR=/workspace/output \
        -e ULTRALYTICS_MIGRAPHX_CACHE_ROOT=/opt/ultralytics-yolo26/models/ort-migraphx-cache \
        -e LLAMACPP_ROOT_URL="http://$LLAMA_CONTAINER:8199" \
        -e LLAMACPP_BASE_URL="http://$LLAMA_CONTAINER:8199/v1" \
        -e HF_ENDPOINT="$HF_ENDPOINT" \
        -e JUPYTER_TOKEN="$TOKEN" \
        -v "$OUTPUT_DIR:/workspace/output" \
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
    /usr/local/bin/validate-ultralytics-yolo26-image 2>&1); then
    echo "$validation_output" >&2
    exit 1
fi

echo "Jupyter: http://127.0.0.1:$JUPYTER_PORT/?token=$TOKEN"
echo "Workspace: baked into $PIPELINE_IMAGE"
echo "Models: baked at /opt/ultralytics-yolo26/models; shared volume: $MODEL_VOLUME"
echo "Model set: $model_set_id"
echo "Output: $OUTPUT_DIR"
echo "Pipeline GPU: $PIPELINE_GPU; llama.cpp GPU: $LLAMA_GPU; VA-API: $VAAPI_DEVICE"
echo "All YOLO/Qwen models are embedded and checksum-verified."
