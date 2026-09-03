#!/usr/bin/env bash
set -euo pipefail

package_root="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
image="${PIPELINE_IMAGE:-zihao/ultralytics-yolo26-workshop:rocm7.2.1}"
gpu="${PIPELINE_GPU:-0}"
vaapi_device="${VAAPI_DEVICE:-/dev/dri/renderD128}"

for path in \
    "$package_root/scripts/notebook_env.py" \
    "$package_root/scripts/model_setup.py" \
    "$package_root/data/sidewalk.mp4"; do
    [[ -e "$path" ]] || {
        echo "Missing smoke-test input: $path" >&2
        exit 1
    }
done

docker image inspect "$image" >/dev/null 2>&1 || {
    echo "Image is unavailable: $image" >&2
    echo "Build it with: bash scripts/build_notebook_image.sh" >&2
    exit 1
}

workdir=$(docker image inspect -f '{{.Config.WorkingDir}}' "$image")
[[ "$workdir" == "/workspace" ]] || {
    echo "Unexpected image workdir: $workdir" >&2
    exit 1
}

video_gid=$(getent group video | cut -d: -f3)
render_gid=$(getent group render | cut -d: -f3)

docker run --rm \
    --user "$(id -u):$(id -g)" \
    --device=/dev/kfd --device=/dev/dri \
    --group-add "$video_gid" --group-add "$render_gid" \
    --ipc=host \
    -e HOME=/tmp \
    -e ROCR_VISIBLE_DEVICES="$gpu" -e HIP_VISIBLE_DEVICES=0 \
    -e VAAPI_DEVICE="$vaapi_device" \
    -e YOLO_CONFIG_DIR=/workspace/output/.ultralytics \
    -v "$package_root:/workspace" \
    "$image" \
    /usr/local/bin/validate-ultralytics-yolo26-image
