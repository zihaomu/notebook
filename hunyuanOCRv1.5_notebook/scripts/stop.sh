#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONTAINER=${CONTAINER:-hunyuanocr-notebook}
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    docker rm -f "$CONTAINER" >/dev/null
    echo "STOPPED container=$CONTAINER"
else
    echo "ABSENT container=$CONTAINER"
fi
rm -f "$ROOT/outputs/runtime/vllm.pid" \
    "$ROOT/outputs/runtime/models.json" \
    "$ROOT/outputs/runtime/attention_backend"
