#!/usr/bin/env bash
set -euo pipefail

package_root="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
run_id="${GITHUB_RUN_ID:-local}-$(printf '%s' "${GITHUB_RUN_ATTEMPT:-1}" | tr -cd '[:alnum:]_.-')"
resource_suffix=$(printf '%s' "$run_id" | tr -cd '[:alnum:]_.-')
output_dir="${OUTPUT_DIR:-${RUNNER_TEMP:-/tmp}/ultralytics_yolo26_$resource_suffix}"
model_volume="${MODEL_VOLUME:-ultralytics_yolo26_models_$resource_suffix}"
network="${NETWORK:-ultralytics_yolo26_$resource_suffix}"
pipeline_container="${PIPELINE_CONTAINER:-ultralytics_yolo26_notebook_$resource_suffix}"
llama_container="${LLAMA_CONTAINER:-ultralytics_yolo26_llama_$resource_suffix}"
jupyter_port="${JUPYTER_PORT:-18992}"
llama_port="${LLAMA_PORT:-18202}"
gpu="${CI_GPU:-0}"
vaapi_device="${CI_VAAPI_DEVICE:-/dev/dri/renderD128}"
pipeline_override="${PIPELINE_IMAGE:-}"
llama_override="${LLAMA_IMAGE:-}"
release_override="${WORKSHOP_RELEASE_ID:-}"
if [[ -z "$pipeline_override" && ( -n "$llama_override" || -n "$release_override" ) ]]; then
    echo "LLAMA_IMAGE/WORKSHOP_RELEASE_ID overrides require PIPELINE_IMAGE" >&2
    exit 1
fi

cleanup() {
    if [[ -d "$output_dir" ]]; then
        mkdir -p "$output_dir/ci-logs"
        docker logs "$pipeline_container" >"$output_dir/ci-logs/pipeline.log" 2>&1 || true
        docker logs "$llama_container" >"$output_dir/ci-logs/llama.log" 2>&1 || true
    fi
    docker rm -f "$pipeline_container" "$llama_container" >/dev/null 2>&1 || true
    docker network rm "$network" >/dev/null 2>&1 || true
    docker volume rm "$model_volume" >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup
mkdir -p "$output_dir/ci-logs"

python3 "$package_root/scripts/validate_release_lock.py" \
    --lock "$package_root/release/current.env" >/dev/null
# shellcheck disable=SC1090
source "$package_root/release/current.env"
pipeline_image="${pipeline_override:-$PIPELINE_IMAGE_REF}"
llama_image="${llama_override:-$LLAMA_IMAGE_REF}"

PIPELINE_IMAGE="$pipeline_image" PIPELINE_GPU="$gpu" VAAPI_DEVICE="$vaapi_device" \
    bash "$package_root/scripts/test_notebook_image.sh"

if [[ -n "$pipeline_override" && -z "$release_override" ]]; then
    release_override=$(docker image inspect -f \
        '{{index .Config.Labels "io.ultralytics.release.id"}}' "$pipeline_image")
    [[ -n "$release_override" && "$release_override" != "<no value>" && "$release_override" != "unreleased" ]] || {
        echo "A candidate pipeline image must declare io.ultralytics.release.id" >&2
        exit 1
    }
fi
if [[ -n "$pipeline_override" ]]; then
    declared_companion=$(docker image inspect -f \
        '{{index .Config.Labels "io.ultralytics.companion.digest"}}' "$pipeline_image")
    [[ "$declared_companion" == "$llama_image" ]] || {
        echo "Candidate companion mismatch: $declared_companion != $llama_image" >&2
        exit 1
    }
fi

# Seed immutable reference outputs before the launcher mounts the writable output directory.
docker run --rm -v "$output_dir:/target" --entrypoint sh "$pipeline_image" \
    -lc 'cp -a /opt/ultralytics-yolo26/seed/output/. /target/ && chmod -R a+rwX /target'

launcher_environment=(
    "PACKAGE_ROOT=$package_root"
    "MODEL_VOLUME=$model_volume"
    "OUTPUT_DIR=$output_dir"
    "NETWORK=$network"
    "PIPELINE_CONTAINER=$pipeline_container"
    "LLAMA_CONTAINER=$llama_container"
    "JUPYTER_PORT=$jupyter_port"
    "LLAMA_PORT=$llama_port"
    "PIPELINE_GPU=$gpu"
    "LLAMA_GPU=$gpu"
    "VAAPI_DEVICE=$vaapi_device"
)
[[ -n "$pipeline_override" ]] && launcher_environment+=("PIPELINE_IMAGE=$pipeline_override")
[[ -n "$llama_override" ]] && launcher_environment+=("LLAMA_IMAGE=$llama_override")
[[ -n "$release_override" ]] && launcher_environment+=("WORKSHOP_RELEASE_ID=$release_override")
env "${launcher_environment[@]}" bash "$package_root/scripts/start_notebook_container.sh"

python3 - "$llama_port" <<'PY'
import json
import sys
import urllib.request

port = int(sys.argv[1])
payload = json.dumps({
    "model": "/models/Qwen3-VL-8B-Instruct-Q8_0.gguf",
    "messages": [{"role": "user", "content": "Reply with exactly READY."}],
    "temperature": 0,
    "max_tokens": 128,
}).encode()
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=120) as response:
    result = json.load(response)
choice = result["choices"][0]
assert choice["message"].get("content", "").strip(), result
assert choice.get("finish_reason") == "stop", result
print("QWEN_COMPLETION=PASS")
PY

docker exec "$pipeline_container" sh -lc '
    cd /workspace &&
    python3 scripts/validate_runtime.py --require-models --require-vlm &&
    python3 tests/test_async_vlm_client.py &&
    python3 tests/test_async_roi_workflow.py &&
    python3 tests/test_predict_production_parity.py \
        --model /opt/ultralytics-yolo26/models/yolo26x.onnx \
        --video data/sidewalk.mp4 --minimum-iou 0.90 &&
    python3 tests/test_gpu_overlay_encode.py \
        --video data/sidewalk.mp4 \
        --output /workspace/output/ci-gpu-overlay-smoke.mp4 \
        --render-node "${VAAPI_DEVICE}" --frames 12 &&
    python3 scripts/run_pipeline.py --validate-only
'

active_release_id="${release_override:-$WORKSHOP_RELEASE_ID}"
echo "GPU_RELEASE_SMOKE=PASS release=$active_release_id"
