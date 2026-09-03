#!/usr/bin/env bash
set -euo pipefail

PIPELINE_CONTAINER="${PIPELINE_CONTAINER:-ultralytics_yolo26_notebook}"
LLAMA_CONTAINER="${LLAMA_CONTAINER:-ultralytics_yolo26_llamacpp}"

for container in "$PIPELINE_CONTAINER" "$LLAMA_CONTAINER"; do
    if docker inspect "$container" >/dev/null 2>&1; then
        docker stop "$container" >/dev/null
        echo "Stopped $container"
    fi
done
