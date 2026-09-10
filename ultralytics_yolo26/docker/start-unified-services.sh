#!/usr/bin/env bash
set -euo pipefail

llamacpp_launcher="${LLAMACPP_LAUNCHER:-/usr/local/bin/start-ultralytics-yolo26-llamacpp}"
sshd_launcher="${SSHD_LAUNCHER:-/usr/local/bin/start-ultralytics-yolo26-sshd}"
jupyter_launcher="${JUPYTER_LAUNCHER:-/usr/local/bin/start-ultralytics-yolo26-jupyter}"
python_binary="${PYTHON_BINARY:-/opt/venv/bin/python}"
children=()

export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"

cleanup() {
    local pid
    trap - EXIT INT TERM
    for pid in "${children[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${children[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}

wait_for_llamacpp() {
    local pid="$1"
    local port="${LLAMACPP_PORT:-8199}"
    local timeout="${LLAMACPP_STARTUP_TIMEOUT:-180}"
    local deadline

    if [[ ! "$timeout" =~ ^[0-9]+$ ]] || (( timeout < 1 )); then
        echo "[services] invalid LLAMACPP_STARTUP_TIMEOUT: $timeout" >&2
        return 2
    fi
    deadline=$((SECONDS + timeout))
    while (( SECONDS < deadline )); do
        if "$python_binary" - "http://127.0.0.1:$port/health" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(sys.argv[1], timeout=2) as response:
    payload = json.loads(response.read())
    if response.status != 200 or payload.get("status") != "ok":
        raise SystemExit(1)
PY
        then
            echo "[services] llama.cpp is ready on http://127.0.0.1:$port"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" || true
            echo "[services] llama.cpp exited before readiness" >&2
            return 1
        fi
        sleep 1
    done
    echo "[services] llama.cpp readiness timed out after ${timeout}s" >&2
    return 1
}

run_all() {
    local llama_pid
    local status

    "$llamacpp_launcher" &
    llama_pid=$!
    children+=("$llama_pid")
    wait_for_llamacpp "$llama_pid"

    FOREGROUND=1 "$sshd_launcher" "${SSH_PORT:-22}" &
    children+=("$!")
    "$jupyter_launcher" &
    children+=("$!")

    set +e
    wait -n "${children[@]}"
    status=$?
    set -e
    if (( status == 0 )); then
        status=1
    fi
    echo "[services] a managed service exited; stopping the container" >&2
    return "$status"
}

case "${SERVICE_MODE:-all}" in
    all)
        trap cleanup EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM
        run_all
        ;;
    jupyter)
        exec "$jupyter_launcher"
        ;;
    sshd)
        export FOREGROUND=1
        exec "$sshd_launcher" "${SSH_PORT:-22}"
        ;;
    llamacpp)
        exec "$llamacpp_launcher"
        ;;
    *)
        echo "SERVICE_MODE must be one of: all, jupyter, sshd, llamacpp" >&2
        exit 2
        ;;
esac
