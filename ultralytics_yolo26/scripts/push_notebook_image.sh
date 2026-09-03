#!/usr/bin/env bash
set -euo pipefail

source_image="${PIPELINE_IMAGE:-zihao/ultralytics-yolo26-workshop:rocm7.2.1-baked}"
target_image="${REGISTRY_IMAGE:-crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/muzihao2/work:ultralytics-yolo26-workshop_2026_09_03}"

docker image inspect "$source_image" >/dev/null
source_revision=$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$source_image")
source_bundle=$(docker image inspect -f '{{index .Config.Labels "io.ultralytics.workshop.bundle.sha256"}}' "$source_image")
if [[ -z "$source_revision" || "$source_revision" == "unknown" ]]; then
    echo "Image is missing org.opencontainers.image.revision" >&2
    exit 1
fi
if [[ -z "$source_bundle" || "$source_bundle" == "unknown" ]]; then
    echo "Image is missing the workshop bundle identity" >&2
    exit 1
fi

printf 'Publishing %s -> %s\n' "$source_image" "$target_image"
printf '  revision: %s\n  bundle:   %s\n' "$source_revision" "$source_bundle"
docker tag "$source_image" "$target_image"
docker push "$target_image"
docker buildx imagetools inspect "$target_image"
