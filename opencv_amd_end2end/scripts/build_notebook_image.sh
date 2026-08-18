#!/usr/bin/env bash
set -euo pipefail

package_root="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
legacy_workspace="${LEGACY_WORKSPACE:-$package_root/../../opencv_workspace/opencv_end2end}"
opencv_source="$(realpath "${OPENCV_SOURCE:-$legacy_workspace/third_party/opencv}")"
opencv_contrib_source="$(realpath "${OPENCV_CONTRIB_SOURCE:-$legacy_workspace/third_party/opencv_contrib}")"
image="${PIPELINE_IMAGE:-zihao/opencv-amd-end2end:rocm7.2.1}"
base_image="${BASE_IMAGE:-zihao/opencv-llamacpp-q8:rocm7.2.1}"
jupyterlab_version="${JUPYTERLAB_VERSION:-4.6.0}"

for path in "$package_root/docker/Dockerfile" "$opencv_source/CMakeLists.txt" "$opencv_contrib_source/modules"; do
    [[ -e "$path" ]] || {
        echo "Missing image build input: $path" >&2
        exit 1
    }
done

docker image inspect "$base_image" >/dev/null 2>&1 || {
    echo "Base image is unavailable: $base_image" >&2
    echo "Set BASE_IMAGE to a pullable image containing the validated /opt runtime." >&2
    exit 1
}

opencv_commit="$(git -C "$opencv_source" rev-parse HEAD)"
opencv_contrib_commit="$(git -C "$opencv_contrib_source" rev-parse HEAD)"
base_image_id="$(docker image inspect -f '{{.Id}}' "$base_image")"

if [[ -n "$(git -C "$opencv_source" status --short)" ]]; then
    echo "OpenCV source has uncommitted changes: $opencv_source" >&2
    exit 1
fi
if [[ -n "$(git -C "$opencv_contrib_source" status --short)" ]]; then
    echo "OpenCV contrib source has uncommitted changes: $opencv_contrib_source" >&2
    exit 1
fi

echo "Building $image"
echo "  base:            $base_image ($base_image_id)"
echo "  OpenCV:          $opencv_source ($opencv_commit)"
echo "  OpenCV contrib:  $opencv_contrib_source ($opencv_contrib_commit)"
echo "  JupyterLab:      $jupyterlab_version"
echo "  user workspace:  mounted at /workspace at runtime"

docker buildx build \
    --load \
    --file "$package_root/docker/Dockerfile" \
    --tag "$image" \
    --build-arg "BASE_IMAGE=$base_image" \
    --build-arg "BASE_IMAGE_ID=$base_image_id" \
    --build-arg "OPENCV_COMMIT=$opencv_commit" \
    --build-arg "OPENCV_CONTRIB_COMMIT=$opencv_contrib_commit" \
    --build-arg "JUPYTERLAB_VERSION=$jupyterlab_version" \
    --build-context "opencv_source=$opencv_source" \
    --build-context "opencv_contrib_source=$opencv_contrib_source" \
    "$package_root"

docker image inspect "$image" --format \
    'Built {{index .RepoTags 0}} ({{.Id}}), workdir={{json .Config.WorkingDir}}'
