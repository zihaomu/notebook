#!/usr/bin/env bash
set -euo pipefail

package_root="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
image="${PIPELINE_IMAGE:-zihao/opencv-amd-end2end:rocm7.2.1}"

for path in \
    "$package_root/scripts/notebook_env.py" \
    "$package_root/data/sidewalk.mp4" \
    "$package_root/models/yolo26x.onnx" \
    "$package_root/models/yolo26x_compiled.mxr"; do
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
    -v "$package_root:/workspace" \
    "$image" \
    /usr/local/bin/validate-opencv-amd-image
