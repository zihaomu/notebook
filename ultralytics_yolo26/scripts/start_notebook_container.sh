#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RELEASE_LOCK="${RELEASE_LOCK:-$PACKAGE_ROOT/release/current.env}"

pipeline_override="${PIPELINE_IMAGE:-}"
llama_override="${LLAMA_IMAGE:-}"
release_override="${WORKSHOP_RELEASE_ID:-}"

python3 "$PACKAGE_ROOT/scripts/validate_release_lock.py" --lock "$RELEASE_LOCK" >/dev/null
# The lock contains a fixed, validated KEY=VALUE schema with no shell metacharacters.
# shellcheck disable=SC1090
source "$RELEASE_LOCK"

PIPELINE_IMAGE="${pipeline_override:-$PIPELINE_IMAGE_REF}"
LLAMA_IMAGE="${llama_override:-$LLAMA_IMAGE_REF}"
WORKSHOP_RELEASE_ID="${release_override:-$WORKSHOP_RELEASE_ID}"
MODEL_VOLUME="${MODEL_VOLUME:-ultralytics_yolo26_models}"
OUTPUT_DIR="${OUTPUT_DIR:-$PACKAGE_ROOT/output}"
NETWORK="${NETWORK:-ultralytics_yolo26}"
PIPELINE_CONTAINER="${PIPELINE_CONTAINER:-ultralytics_yolo26_notebook}"
LLAMA_CONTAINER="${LLAMA_CONTAINER:-ultralytics_yolo26_llamacpp}"
JUPYTER_PORT="${JUPYTER_PORT:-8892}"
LLAMA_PORT="${LLAMA_PORT:-8202}"
TOKEN="${TOKEN:-ultralytics-yolo26}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PIPELINE_GPU="${PIPELINE_GPU:-0}"
LLAMA_GPU="${LLAMA_GPU:-$PIPELINE_GPU}"
VAAPI_DEVICE="${VAAPI_DEVICE:-/dev/dri/renderD128}"
ALLOW_LOCAL_BUILD="${ALLOW_LOCAL_BUILD:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
LLAMA_RUNTIME_PROFILE="${LLAMA_RUNTIME_PROFILE:-slot3-cache-off-v1}"

development_mode=0
if [[ -n "$pipeline_override" || -n "$llama_override" || -n "$release_override" ]]; then
    development_mode=1
    echo "WARNING: image/release override enabled; this is development mode." >&2
fi

ensure_image() {
    local role="$1"
    local reference="$2"
    if docker image inspect "$reference" >/dev/null 2>&1; then
        return
    fi
    echo "Pulling $role image: $reference"
    if docker pull "$reference"; then
        return
    fi
    if [[ "$role" == "pipeline" && "$ALLOW_LOCAL_BUILD" == "1" && "$reference" != *@sha256:* ]]; then
        echo "WARNING: pull failed; building local development image $reference" >&2
        PIPELINE_IMAGE="$reference" bash "$PACKAGE_ROOT/scripts/build_notebook_image.sh"
        return
    fi
    echo "Unable to obtain $role image: $reference" >&2
    exit 1
}

image_id() {
    docker image inspect -f '{{.Id}}' "$1"
}

container_environment() {
    docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$1"
}

wait_for_url() {
    local name="$1"
    local url="$2"
    local container="$3"
    local attempts="${4:-120}"
    local delay="${5:-2}"
    local attempt
    for ((attempt = 1; attempt <= attempts; attempt++)); do
        if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
            return
        fi
        sleep "$delay"
    done
    echo "$name did not become ready: $url" >&2
    docker logs --tail 100 "$container" >&2 || true
    return 1
}

ensure_image pipeline "$PIPELINE_IMAGE"
ensure_image llama.cpp "$LLAMA_IMAGE"

pipeline_image_id=$(image_id "$PIPELINE_IMAGE")
llama_image_id=$(image_id "$LLAMA_IMAGE")
if [[ "$development_mode" == "0" ]]; then
    python3 "$PACKAGE_ROOT/scripts/validate_release_lock.py" \
        --lock "$RELEASE_LOCK" --check-local-images >/dev/null
else
    for reference in "$PIPELINE_IMAGE" "$LLAMA_IMAGE"; do
        platform=$(docker image inspect -f '{{.Os}}/{{.Architecture}}' "$reference")
        [[ "$platform" == "linux/amd64" ]] || {
            echo "Unsupported image platform for $reference: $platform" >&2
            exit 1
        }
    done
    docker run --rm --entrypoint test "$LLAMA_IMAGE" \
        -x /opt/llama.cpp/build/bin/llama-server
fi

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
    printf 'RELEASE_PREFLIGHT=PASS\n'
    printf 'release_id=%s\npipeline=%s\npipeline_id=%s\n' \
        "$WORKSHOP_RELEASE_ID" "$PIPELINE_IMAGE" "$pipeline_image_id"
    printf 'llama=%s\nllama_id=%s\n' "$LLAMA_IMAGE" "$llama_image_id"
    exit 0
fi

mkdir -p "$OUTPUT_DIR"
# The full image contains all four model files and the matching gfx1100 cache.
# A verified named volume exposes the same immutable model set to llama.cpp.
docker volume create "$MODEL_VOLUME" >/dev/null
docker run --rm \
    -v "$MODEL_VOLUME:/model-volume" \
    --entrypoint /usr/local/bin/init-ultralytics-yolo26-model-volume \
    "$PIPELINE_IMAGE" /model-volume >/dev/null
model_set_id=$(docker run --rm --entrypoint cat "$PIPELINE_IMAGE" \
    /opt/ultralytics-yolo26/models/MODEL_SET_ID)
[[ "$model_set_id" == "$WORKSHOP_MODEL_SET_SHA256" ]] || {
    echo "Release/model-set mismatch: $model_set_id != $WORKSHOP_MODEL_SET_SHA256" >&2
    exit 1
}

docker network create "$NETWORK" >/dev/null 2>&1 || true
video_gid=$(getent group video | cut -d: -f3)
render_gid=$(getent group render | cut -d: -f3)
user_id=$(id -u)
group_id=$(id -g)

# Recreate containers when their image, release identity, or data mounts differ.
if docker inspect "$LLAMA_CONTAINER" >/dev/null 2>&1; then
    current_model_name=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/models"}}{{.Name}}{{end}}{{end}}' "$LLAMA_CONTAINER")
    current_llama_image_id=$(docker inspect -f '{{.Image}}' "$LLAMA_CONTAINER")
    current_llama_env=$(container_environment "$LLAMA_CONTAINER")
    current_model_set_id=$(sed -n 's/^ULTRALYTICS_MODEL_SET_ID=//p' <<<"$current_llama_env")
    current_release_id=$(sed -n 's/^WORKSHOP_RELEASE_ID=//p' <<<"$current_llama_env")
    current_runtime_profile=$(sed -n 's/^ULTRALYTICS_LLAMA_RUNTIME_PROFILE=//p' <<<"$current_llama_env")
    if [[ "$current_model_name" != "$MODEL_VOLUME" || \
          "$current_model_set_id" != "$model_set_id" || \
          "$current_release_id" != "$WORKSHOP_RELEASE_ID" || \
          "$current_runtime_profile" != "$LLAMA_RUNTIME_PROFILE" || \
          "$current_llama_image_id" != "$llama_image_id" ]]; then
        docker rm -f "$LLAMA_CONTAINER" >/dev/null
    fi
fi
if docker inspect "$PIPELINE_CONTAINER" >/dev/null 2>&1; then
    current_workspace_mount=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}' "$PIPELINE_CONTAINER")
    current_output_dir=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/workspace/output"}}{{.Source}}{{end}}{{end}}' "$PIPELINE_CONTAINER")
    current_pipeline_image_id=$(docker inspect -f '{{.Image}}' "$PIPELINE_CONTAINER")
    current_pipeline_env=$(container_environment "$PIPELINE_CONTAINER")
    current_release_id=$(sed -n 's/^WORKSHOP_RELEASE_ID=//p' <<<"$current_pipeline_env")
    if [[ -n "$current_workspace_mount" || \
          -z "$current_output_dir" || \
          "$(realpath "$current_output_dir")" != "$(realpath "$OUTPUT_DIR")" || \
          "$current_pipeline_image_id" != "$pipeline_image_id" || \
          "$current_release_id" != "$WORKSHOP_RELEASE_ID" ]]; then
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
        -e WORKSHOP_RELEASE_ID="$WORKSHOP_RELEASE_ID" \
        -e ULTRALYTICS_MODEL_SET_ID="$model_set_id" \
        -e ULTRALYTICS_LLAMA_RUNTIME_PROFILE="$LLAMA_RUNTIME_PROFILE" \
        -v "$MODEL_VOLUME:/models:ro" \
        "$LLAMA_IMAGE" \
        sh -lc 'exec /opt/llama.cpp/build/bin/llama-server --model /models/Qwen3-VL-8B-Instruct-Q8_0.gguf --mmproj /models/mmproj-F16.gguf --host 0.0.0.0 --port 8199 --device ROCm0 --n-gpu-layers 99 --ctx-size 12288 --parallel 3 --cache-ram 0 --flash-attn auto --image-min-tokens 1024' >/dev/null
fi

wait_for_url "llama.cpp health endpoint" "http://127.0.0.1:$LLAMA_PORT/health" "$LLAMA_CONTAINER"
models_json=$(curl -fsS --max-time 10 "http://127.0.0.1:$LLAMA_PORT/v1/models")
python3 -c '
import json, sys
payload = json.load(sys.stdin)
models = payload.get("models") or []
data = payload.get("data") or []
assert any(item.get("id", "").endswith("Qwen3-VL-8B-Instruct-Q8_0.gguf") for item in data), payload
assert any("multimodal" in item.get("capabilities", []) for item in models), payload
' <<<"$models_json"

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
        -e WORKSHOP_RELEASE_ID="$WORKSHOP_RELEASE_ID" \
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

wait_for_url "Jupyter API" "http://127.0.0.1:$JUPYTER_PORT/api/status?token=$TOKEN" "$PIPELINE_CONTAINER" 60 1

if ! validation_output=$(docker exec "$PIPELINE_CONTAINER" \
    /usr/local/bin/validate-ultralytics-yolo26-image 2>&1); then
    echo "$validation_output" >&2
    exit 1
fi

printf 'WORKSHOP_READY=PASS\n'
printf 'Release: %s\n' "$WORKSHOP_RELEASE_ID"
printf 'Jupyter: http://127.0.0.1:%s/?token=%s\n' "$JUPYTER_PORT" "$TOKEN"
printf 'Pipeline image: %s (%s)\n' "$PIPELINE_IMAGE" "$pipeline_image_id"
printf 'llama.cpp image: %s (%s)\n' "$LLAMA_IMAGE" "$llama_image_id"
printf 'Models: baked at /opt/ultralytics-yolo26/models; shared volume: %s\n' "$MODEL_VOLUME"
printf 'Model set: %s\nOutput: %s\n' "$model_set_id" "$OUTPUT_DIR"
printf 'Pipeline GPU: %s; llama.cpp GPU: %s; VA-API: %s\n' "$PIPELINE_GPU" "$LLAMA_GPU" "$VAAPI_DEVICE"
