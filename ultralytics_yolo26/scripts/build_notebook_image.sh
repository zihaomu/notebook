#!/usr/bin/env bash
set -euo pipefail

package_root="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
image="${PIPELINE_IMAGE:-zihao/ultralytics-yolo26-workshop:rocm7.2.1-baked}"
base_image="${BASE_IMAGE:-crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/muzihao2/work:opencv_end2end_2026_08_12}"
ultralytics_repository="${ULTRALYTICS_REPOSITORY:-https://gh-test.anruicloud.com/zihaomu/ultralytics.git}"
ultralytics_branch="${ULTRALYTICS_BRANCH:-add-onnx-migraphx-backend}"
ultralytics_commit="${ULTRALYTICS_COMMIT:-34e213ca3ece4c18962f5bb922ec74da0c474d24}"
ultralytics_source="${ULTRALYTICS_SOURCE:-$package_root/.build/ultralytics}"
ort_version="${ONNXRUNTIME_MIGRAPHX_VERSION:-1.24.2}"
wheel_dir="${ORT_WHEEL_DIR:-$package_root/.build/wheels}"
patch_path="$package_root/docker/patches/ultralytics-migraphx-iobinding.patch"
cache_dir="$package_root/models/ort-migraphx-cache/735f1583e99dfeb733da"
cache_file="$cache_dir/20e00-58de11c69ae52cf2-9880cf1608079e0d-36a8840bfe2de0d1.mxr"

for path in \
    "$package_root/docker/Dockerfile" \
    "$package_root/docker/start-jupyter.sh" \
    "$package_root/docker/validate-image.py" \
    "$package_root/native/hip_vaapi_bridge.cpp" \
    "$package_root/native/vaapi_hip_encode_probe.cpp" \
    "$package_root/scripts/build_native_bridge.sh" \
    "$package_root/scripts/build_vaapi_hip_probe.sh" \
    "$package_root/scripts/workshop_bundle_identity.py" \
    "$package_root/models/yolo26x.pt" \
    "$package_root/models/yolo26x.onnx" \
    "$cache_dir/identity.json" \
    "$cache_file" \
    "$patch_path"; do
    [[ -f "$path" ]] || {
        echo "Missing image build input: $path" >&2
        exit 1
    }
done

docker image inspect "$base_image" >/dev/null 2>&1 || docker pull "$base_image"
base_image_id=$(docker image inspect -f '{{.Id}}' "$base_image")
patch_sha256=$(sha256sum "$patch_path" | cut -d' ' -f1)
bridge_sha256=$(sha256sum "$package_root/native/hip_vaapi_bridge.cpp" | cut -d' ' -f1)
checkpoint_sha256=$(sha256sum "$package_root/models/yolo26x.pt" | cut -d' ' -f1)
onnx_sha256=$(sha256sum "$package_root/models/yolo26x.onnx" | cut -d' ' -f1)
cache_sha256=$(sha256sum "$cache_file" | cut -d' ' -f1)
bundle_sha256=$(python3 "$package_root/scripts/workshop_bundle_identity.py" "$package_root")
workshop_git_commit=$(git -C "$package_root" rev-parse HEAD)

if [[ ! -d "$ultralytics_source/.git" ]] || \
        [[ "$(git -C "$ultralytics_source" rev-parse HEAD 2>/dev/null || true)" != "$ultralytics_commit" ]]; then
    rm -rf "$ultralytics_source"
    mkdir -p "$(dirname "$ultralytics_source")"
    git -c http.version=HTTP/1.1 clone --depth 1 --single-branch \
        --branch "$ultralytics_branch" "$ultralytics_repository" "$ultralytics_source"
fi
[[ "$(git -C "$ultralytics_source" rev-parse HEAD)" == "$ultralytics_commit" ]] || {
    echo "Ultralytics source is not at $ultralytics_commit" >&2
    exit 1
}
[[ -z "$(git -C "$ultralytics_source" status --short)" ]] || {
    echo "Ultralytics source must be clean: $ultralytics_source" >&2
    exit 1
}
git -C "$ultralytics_source" apply --check "$patch_path"

mkdir -p "$wheel_dir"
if ! compgen -G "$wheel_dir/onnxruntime_migraphx-${ort_version}-cp310-*.whl" >/dev/null; then
    docker run --rm --network host \
        -v "$wheel_dir:/wheels" \
        --entrypoint /opt/venv/bin/pip \
        "$base_image" \
        download --no-deps --dest /wheels "onnxruntime-migraphx==$ort_version"
fi

wheel_path=$(compgen -G "$wheel_dir/onnxruntime_migraphx-${ort_version}-cp310-*.whl" | head -1)
echo "Building $image"
echo "  base:              $base_image ($base_image_id)"
echo "  Ultralytics:       $ultralytics_repository@$ultralytics_commit"
echo "  ORT MIGraphX:      $wheel_path"
echo "  I/O binding patch: $patch_sha256"
echo "  HIP/VAAPI bridge:  $bridge_sha256"
echo "  YOLO checkpoint:   $checkpoint_sha256"
echo "  YOLO ONNX:         $onnx_sha256"
echo "  gfx1100 MXR cache: $cache_sha256"
echo "  Workshop bundle:  $bundle_sha256"
echo "  Workshop commit:  $workshop_git_commit"

docker buildx build \
    --load \
    --file "$package_root/docker/Dockerfile" \
    --tag "$image" \
    --build-arg "BASE_IMAGE=$base_image" \
    --build-arg "BASE_IMAGE_ID=$base_image_id" \
    --build-arg "ULTRALYTICS_COMMIT=$ultralytics_commit" \
    --build-arg "ONNXRUNTIME_MIGRAPHX_VERSION=$ort_version" \
    --build-arg "ULTRALYTICS_PATCH_SHA256=$patch_sha256" \
    --build-arg "HIP_VAAPI_BRIDGE_SHA256=$bridge_sha256" \
    --build-arg "WORKSHOP_BUNDLE_SHA256=$bundle_sha256" \
    --build-arg "YOLO_CHECKPOINT_SHA256=$checkpoint_sha256" \
    --build-arg "YOLO_ONNX_SHA256=$onnx_sha256" \
    --build-arg "MIGRAPHX_CACHE_SHA256=$cache_sha256" \
    --build-arg "WORKSHOP_GIT_COMMIT=$workshop_git_commit" \
    --build-context "ultralytics_source=$ultralytics_source" \
    --build-context "ort_wheel=$wheel_dir" \
    "$package_root"

docker image inspect "$image" --format \
    'Built {{index .RepoTags 0}} ({{.Id}}), workdir={{json .Config.WorkingDir}}'
