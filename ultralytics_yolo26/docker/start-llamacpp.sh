#!/usr/bin/env bash
set -euo pipefail

binary="${LLAMACPP_BINARY:-/usr/local/bin/llama-server}"
model_dir="${ULTRALYTICS_YOLO26_MODEL_DIR:-/opt/ultralytics-yolo26/models}"
model="${LLAMACPP_MODEL:-$model_dir/Qwen3-VL-8B-Instruct-Q8_0.gguf}"
mmproj="${LLAMACPP_MMPROJ:-$model_dir/mmproj-F16.gguf}"
port="${LLAMACPP_PORT:-8199}"

[[ -x "$binary" ]] || {
    echo "[llama.cpp] executable is missing: $binary" >&2
    exit 1
}
[[ -s "$model" ]] || {
    echo "[llama.cpp] model is missing or empty: $model" >&2
    exit 1
}
[[ -s "$mmproj" ]] || {
    echo "[llama.cpp] vision projector is missing or empty: $mmproj" >&2
    exit 1
}
if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
    echo "[llama.cpp] invalid port: $port" >&2
    exit 1
fi

echo "[llama.cpp] starting b9766 on http://127.0.0.1:$port"
exec "$binary" \
    --model "$model" \
    --mmproj "$mmproj" \
    --host 127.0.0.1 \
    --port "$port" \
    --device "${LLAMACPP_DEVICE:-ROCm0}" \
    --n-gpu-layers "${LLAMACPP_GPU_LAYERS:-99}" \
    --ctx-size "${LLAMACPP_CTX_SIZE:-12288}" \
    --parallel "${LLAMACPP_PARALLEL:-3}" \
    --cache-ram "${LLAMACPP_CACHE_RAM:-0}" \
    --flash-attn "${LLAMACPP_FLASH_ATTN:-auto}" \
    --image-min-tokens "${LLAMACPP_IMAGE_MIN_TOKENS:-1024}"
