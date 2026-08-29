#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${IMAGE:-hunyuanocr-notebook:20260828}
CONTAINER=${CONTAINER:-hunyuanocr-notebook}
GPU=${GPU:-1}
JUPYTER_PORT=${JUPYTER_PORT:-8892}
VLLM_PORT=${VLLM_PORT:-18016}
SSH_PORT=${SSH_PORT:-2222}
SSH_AUTHORIZED_KEYS_FILE=${SSH_AUTHORIZED_KEYS_FILE:-$HOME/.ssh/authorized_keys}
CONTAINER_AUTHORIZED_KEYS_FILE=/hunyuanOCR_workspace/ssh/authorized_keys
CACHE_VOLUME=${CACHE_VOLUME:-hunyuanocr-notebook-cache}
CONTAINER_UID=1001
CONTAINER_GID=1001
TOKEN=${TOKEN:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(18))')}

if [[ "$(id -u):$(id -g)" != "$CONTAINER_UID:$CONTAINER_GID" ]]; then
    echo "ERROR: this demo requires host UID:GID $CONTAINER_UID:$CONTAINER_GID for bind mounts" >&2
    exit 1
fi
if [[ ! -r "$SSH_AUTHORIZED_KEYS_FILE" || ! -s "$SSH_AUTHORIZED_KEYS_FILE" ]]; then
    echo "ERROR: SSH authorized keys file is missing or unreadable: $SSH_AUTHORIZED_KEYS_FILE" >&2
    exit 1
fi
if ! ssh-keygen -l -f "$SSH_AUTHORIZED_KEYS_FILE" >/dev/null 2>&1; then
    echo "ERROR: SSH authorized keys file contains no valid public key: $SSH_AUTHORIZED_KEYS_FILE" >&2
    exit 1
fi
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "ERROR: container already exists: $CONTAINER" >&2
    exit 1
fi
for port in "$JUPYTER_PORT" "$VLLM_PORT" "$SSH_PORT"; do
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
        echo "ERROR: host port is already in use: $port" >&2
        exit 1
    fi
done
mkdir -p "$ROOT/.runtime/home" "$ROOT/outputs" "$ROOT/assets/images"
printf '%s\n' "$TOKEN" > "$ROOT/.runtime/jupyter-token"
chmod 0600 "$ROOT/.runtime/jupyter-token"

docker run --rm --user root --entrypoint chown \
    -v "$ROOT:/hunyuanOCR_workspace/demo" "$IMAGE" \
    -R "$CONTAINER_UID:$CONTAINER_GID" \
    /hunyuanOCR_workspace/demo/assets \
    /hunyuanOCR_workspace/demo/outputs \
    /hunyuanOCR_workspace/demo/.runtime

docker volume create "$CACHE_VOLUME" >/dev/null
docker run --rm --user root --entrypoint chown \
    -v "$CACHE_VOLUME:/hunyuanOCR_workspace/cache" "$IMAGE" \
    -R "$CONTAINER_UID:$CONTAINER_GID" /hunyuanOCR_workspace/cache
docker run --rm --user "$CONTAINER_UID:$CONTAINER_GID" \
    --entrypoint /hunyuanOCR_workspace/jupyter-venv/bin/python \
    -v "$ROOT:/hunyuanOCR_workspace/demo" \
    "$IMAGE" /hunyuanOCR_workspace/demo/scripts/generate_assets.py

docker run -d \
    --name "$CONTAINER" \
    --network host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add 44 \
    --group-add 993 \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --shm-size=16g \
    --user "$CONTAINER_UID:$CONTAINER_GID" \
    -e "HIP_VISIBLE_DEVICES=$GPU" \
    -e MIOPEN_FIND_MODE=2 \
    -e "JUPYTER_PORT=$JUPYTER_PORT" \
    -e "JUPYTER_TOKEN=$TOKEN" \
    -e "VLLM_PORT=$VLLM_PORT" \
    -e "SSH_PORT=$SSH_PORT" \
    -e "SSH_AUTHORIZED_KEYS_FILE=$CONTAINER_AUTHORIZED_KEYS_FILE" \
    -v "$ROOT:/hunyuanOCR_workspace/demo" \
    -v "$SSH_AUTHORIZED_KEYS_FILE:$CONTAINER_AUTHORIZED_KEYS_FILE:ro" \
    -v "$ROOT/.runtime/home:/hunyuanOCR_workspace/home" \
    -v "$CACHE_VOLUME:/hunyuanOCR_workspace/cache" \
    "$IMAGE" >/dev/null

for _ in {1..60}; do
    if curl --noproxy '*' -fsS --max-time 3 \
        "http://127.0.0.1:${JUPYTER_PORT}/api/status?token=${TOKEN}" >/dev/null 2>&1 \
        && ssh-keyscan -T 3 -p "$SSH_PORT" 127.0.0.1 >/dev/null 2>&1; then
        printf 'READY container=%s\nLogin page: http://127.0.0.1:%s/login\nToken: %s\nNotebook path: hunyuan_ocr_demo.ipynb\nSSH: ssh -p %s hunyuanocr@<server-address>\nAPI after notebook startup cell: http://127.0.0.1:%s/v1\n' \
            "$CONTAINER" "$JUPYTER_PORT" "$TOKEN" "$SSH_PORT" "$VLLM_PORT"
        exit 0
    fi
    sleep 2
done
docker logs --tail 120 "$CONTAINER" >&2
exit 1
