#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${HUNYUANOCR_DEMO_ROOT:-/hunyuanOCR_workspace/demo}
PID_FILE=$ROOT/outputs/runtime/vllm.pid
BACKEND_FILE=$ROOT/outputs/runtime/attention_backend
if [[ ! -f "$PID_FILE" ]]; then
    rm -f "$BACKEND_FILE"
    echo 'ABSENT: no vLLM pid file'
    exit 0
fi
pid=$(<"$PID_FILE")
state=
if [[ "$pid" =~ ^[0-9]+$ && -r "/proc/${pid}/stat" ]]; then
    state=$(awk '{print $3}' "/proc/${pid}/stat")
fi
if [[ -n "$state" && "$state" != Z ]]; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in {1..90}; do
        state=
        if [[ -r "/proc/${pid}/stat" ]]; then
            state=$(awk '{print $3}' "/proc/${pid}/stat")
        fi
        [[ -z "$state" || "$state" == Z ]] && break
        sleep 1
    done
    if [[ -n "$state" && "$state" != Z ]]; then
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
fi
rm -f "$PID_FILE" "$BACKEND_FILE"
echo "STOPPED: pid=${pid}"
