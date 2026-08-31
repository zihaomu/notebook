#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONTAINER=${CONTAINER:-hunyuanocr-notebook}
JUPYTER_PORT=${JUPYTER_PORT:-8892}
VLLM_PORT=${VLLM_PORT:-18016}
SSH_PORT=${SSH_PORT:-2222}
if ! docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "ABSENT container=$CONTAINER"
    exit 1
fi
docker inspect "$CONTAINER" --format 'container={{.Name}} image={{.Image}} status={{.State.Status}}'
backend=$(docker inspect "$CONTAINER" --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | awk -F= '$1 == "VLLM_ATTENTION_BACKEND" {print $2}')
printf 'attention_backend=%s\n' "${backend:-ROCM_ATTN}"
docker top "$CONTAINER" -eo pid,ppid,etime,stat,%cpu,%mem,args
if [[ -f "$ROOT/.runtime/jupyter-token" ]]; then
    token=$(<"$ROOT/.runtime/jupyter-token")
    if curl --noproxy '*' -fsS --max-time 3 \
        "http://127.0.0.1:${JUPYTER_PORT}/api/status?token=${token}" >/dev/null; then
        printf 'jupyter=READY\nLogin page: http://127.0.0.1:%s/login\nToken: %s\nNotebook path: hunyuan_ocr_demo.ipynb\n' \
            "$JUPYTER_PORT" "$token"
    else
        echo 'jupyter=NOT_READY'
    fi
fi
if ssh-keyscan -T 3 -p "$SSH_PORT" 127.0.0.1 >/dev/null 2>&1; then
    printf 'ssh=READY user=hunyuanocr port=%s\nSSH: ssh -p %s hunyuanocr@<server-address>\n' \
        "$SSH_PORT" "$SSH_PORT"
else
    echo 'ssh=NOT_READY'
fi
curl --noproxy '*' -fsS --max-time 3 "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null \
    | jq -c '{vllm:"READY",model:.data[0].id,root:.data[0].root,max_model_len:.data[0].max_model_len}' \
    || echo 'vllm=NOT_STARTED'
