#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=${HUNYUANOCR_DEMO_ROOT:-/hunyuanOCR_workspace/demo}
: "${JUPYTER_TOKEN:?JUPYTER_TOKEN must be set}"
: "${JUPYTER_PORT:=8892}"
: "${SSH_PORT:=2222}"
: "${SSH_AUTHORIZED_KEYS_FILE:=/hunyuanOCR_workspace/ssh/authorized_keys}"
: "${SSH_STATE_DIR:=/hunyuanOCR_workspace/cache/ssh}"
export SSH_PORT SSH_AUTHORIZED_KEYS_FILE SSH_STATE_DIR
[[ "${MIOPEN_FIND_MODE:-}" == 2 ]] || { echo 'ERROR: MIOPEN_FIND_MODE=2 is required' >&2; exit 64; }
test ! -e /workspace
mkdir -p "$ROOT/outputs/json" "$ROOT/outputs/visualizations" "$ROOT/outputs/logs" "$ROOT/outputs/runtime" "$HOME"
test -w "$ROOT/outputs/runtime"
rm -f "$ROOT/outputs/runtime/vllm.pid" "$ROOT/outputs/runtime/models.json" "$SSH_STATE_DIR/sshd.pid"
/hunyuanOCR_workspace/jupyter-venv/bin/python "$ROOT/scripts/generate_assets.py"
"$ROOT/scripts/start_sshd.sh"
exec /hunyuanOCR_workspace/jupyter-venv/bin/jupyter lab \
    --allow-root \
    --no-browser \
    --ip=0.0.0.0 \
    --port="$JUPYTER_PORT" \
    --ServerApp.port_retries=0 \
    --ServerApp.root_dir="$ROOT" \
    --ServerApp.allow_remote_access=True \
    --IdentityProvider.token="$JUPYTER_TOKEN"
