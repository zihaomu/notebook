#!/usr/bin/env bash
set -Eeuo pipefail
CONTAINER=${CONTAINER:-hunyuanocr-notebook}
if ! docker container inspect "$CONTAINER" >/dev/null 2>&1; then
    echo "ABSENT container=$CONTAINER"
    exit 0
fi
docker rm -f "$CONTAINER" >/dev/null
echo "STOPPED container=$CONTAINER"
