#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${HUNYUANOCR_DEMO_ROOT:-/hunyuanOCR_workspace/demo}
VLLM_PORT=${VLLM_PORT:-18016}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-1800}
RUNTIME_DIR=$ROOT/outputs/runtime
LOG_DIR=$ROOT/outputs/logs
PID_FILE=$RUNTIME_DIR/vllm.pid
LOG_FILE=$LOG_DIR/vllm.log
MODELS_FILE=$RUNTIME_DIR/models.json
mkdir -p "$RUNTIME_DIR" "$LOG_DIR"

if curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:${VLLM_PORT}/v1/models" > "$MODELS_FILE" 2>/dev/null; then
    echo "READY: vLLM already serves on port ${VLLM_PORT}"
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
test -f /hunyuanOCR_workspace/models/HunyuanOCR/model.safetensors
test ! -e /workspace
started=$(date +%s)
nohup setsid env PORT="$VLLM_PORT" MIOPEN_FIND_MODE=2 \
    /hunyuanOCR_workspace/bin/entrypoint.sh serve \
    </dev/null > "$LOG_FILE" 2>&1 &
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
printf 'READY: pid=%s port=%s elapsed_seconds=%s\n' "$pid" "$VLLM_PORT" "$elapsed"
