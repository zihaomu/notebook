#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${HUNYUANOCR_DEMO_ROOT:-/hunyuanOCR_workspace/demo}
VLLM_PORT=${VLLM_PORT:-18016}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-ROCM_ATTN}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1800}
RUNTIME_DIR=$ROOT/outputs/runtime
LOG_DIR=$ROOT/outputs/logs
PID_FILE=$RUNTIME_DIR/vllm.pid
LOG_FILE=$LOG_DIR/vllm.log
MODELS_FILE=$RUNTIME_DIR/models.json
BACKEND_FILE=$RUNTIME_DIR/attention_backend
mkdir -p "$RUNTIME_DIR" "$LOG_DIR"

if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:${VLLM_PORT}/v1/models" > "$MODELS_FILE" 2>/dev/null; then
    active_backend=$(cat "$BACKEND_FILE" 2>/dev/null || true)
    if [[ "$active_backend" != "$VLLM_ATTENTION_BACKEND" ]]; then
        echo "ERROR: vLLM uses ${active_backend:-an unknown backend}; stop it before selecting ${VLLM_ATTENTION_BACKEND}" >&2
        exit 1
    fi
    echo "READY: vLLM already serves on port ${VLLM_PORT} with ${active_backend}"
    exit 0
fi

if [[ -f "$PID_FILE" ]]; then
    pid=$(<"$PID_FILE")
    state=
    if [[ "$pid" =~ ^[0-9]+$ && -r "/proc/${pid}/stat" ]]; then
        state=$(awk '{print $3}' "/proc/${pid}/stat")
    fi
    if [[ -n "$state" && "$state" != Z ]]; then
        echo "ERROR: vLLM pid ${pid} is alive (state=${state}) but the API is not ready" >&2
        exit 1
    fi
    rm -f "$PID_FILE"
fi

[[ "${MIOPEN_FIND_MODE:-}" == 2 ]] || { echo 'ERROR: MIOPEN_FIND_MODE=2 is required' >&2; exit 64; }
case "$VLLM_ATTENTION_BACKEND" in
    ROCM_ATTN|TRITON_ATTN) ;;
    *)
        echo 'ERROR: VLLM_ATTENTION_BACKEND must be ROCM_ATTN or TRITON_ATTN' >&2
        exit 64
        ;;
esac
test -f /hunyuanOCR_workspace/models/HunyuanOCR/model.safetensors
if [[ -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
    unset HIP_VISIBLE_DEVICES
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    unset CUDA_VISIBLE_DEVICES
fi
mkdir -p "$HOME" "$VLLM_CACHE_ROOT" "$MIOPEN_USER_DB_PATH" \
    "$MIOPEN_CUSTOM_CACHE_DIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR"
: > "$LOG_FILE"
/opt/venv/bin/python3 /hunyuanOCR_workspace/bin/verify_model.py \
    --model-dir "$MODEL_DIR" >> "$LOG_FILE" 2>&1
/opt/venv/bin/python3 - <<'PYGPU' >> "$LOG_FILE" 2>&1
import json
import torch

facts = {
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": [],
}
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    facts["devices"].append({
        "index": index,
        "name": properties.name,
        "arch": getattr(properties, "gcnArchName", None),
    })
assert facts["torch"] == "2.10.0+rocm7.2.4.git3d3aa833", facts
assert facts["hip"] == "7.2.53211", facts
assert facts["cuda_available"], facts
assert facts["device_count"] == 1, facts
assert str(facts["devices"][0]["arch"]).startswith("gfx1100"), facts
print(json.dumps(facts, sort_keys=True))
PYGPU
started=$(date +%s)
printf '%s\n' "$VLLM_ATTENTION_BACKEND" > "$BACKEND_FILE"
nohup setsid env -u VLLM_ATTENTION_BACKEND MIOPEN_FIND_MODE=2 \
    /opt/venv/bin/vllm serve "$MODEL_DIR" \
    --served-model-name tencent/HunyuanOCR \
    -tp 1 \
    --limit-mm-per-prompt '{"image":4,"video":0}' \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 131072 \
    --max-num-batched-tokens 131072 \
    --skip-mm-profiling \
    --attention-backend "$VLLM_ATTENTION_BACKEND" \
    </dev/null >> "$LOG_FILE" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"

if ! curl --noproxy '*' --fail --silent --show-error \
    --retry 360 --retry-all-errors --retry-connrefused --retry-delay 5 \
    --retry-max-time "$STARTUP_TIMEOUT" --connect-timeout 2 --max-time 5 \
    "http://127.0.0.1:${VLLM_PORT}/v1/models" > "$MODELS_FILE"; then
    tail -n 120 "$LOG_FILE" >&2 || true
    exit 1
fi

jq -e '.data[0].id=="tencent/HunyuanOCR" and .data[0].root=="/hunyuanOCR_workspace/models/HunyuanOCR" and .data[0].max_model_len==131072' "$MODELS_FILE" >/dev/null
elapsed=$(( $(date +%s) - started ))
printf 'READY: pid=%s port=%s backend=%s elapsed_seconds=%s\n' \
    "$pid" "$VLLM_PORT" "$VLLM_ATTENTION_BACKEND" "$elapsed"
