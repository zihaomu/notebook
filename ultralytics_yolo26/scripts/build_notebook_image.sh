#!/usr/bin/env bash
set -euo pipefail

package_root="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
image="${PIPELINE_IMAGE:-zihao/ultralytics-yolo26-workshop:rocm7.2.1}"
base_image="${BASE_IMAGE:-crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/muzihao2/work:opencv_end2end_2026_08_12}"
ultralytics_repository="${ULTRALYTICS_REPOSITORY:-https://gh-test.anruicloud.com/zihaomu/ultralytics.git}"
ultralytics_branch="${ULTRALYTICS_BRANCH:-add-onnx-migraphx-backend}"
ultralytics_commit="${ULTRALYTICS_COMMIT:-34e213ca3ece4c18962f5bb922ec74da0c474d24}"
ultralytics_source="${ULTRALYTICS_SOURCE:-$package_root/.build/ultralytics}"
ort_version="${ONNXRUNTIME_MIGRAPHX_VERSION:-1.24.2}"
wheel_dir="${ORT_WHEEL_DIR:-$package_root/.build/wheels}"
patch_path="$package_root/docker/patches/ultralytics-migraphx-iobinding.patch"

for path in \
    "$package_root/docker/Dockerfile" \
    "$package_root/docker/start-jupyter.sh" \
    "$package_root/docker/validate-image.py" \
    "$package_root/native/hip_vaapi_bridge.cpp" \
    "$package_root/scripts/build_native_bridge.sh" \
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
    --build-context "ultralytics_source=$ultralytics_source" \
    --build-context "ort_wheel=$wheel_dir" \
    "$package_root"

docker image inspect "$image" --format \
    'Built {{index .RepoTags 0}} ({{.Id}}), workdir={{json .Config.WorkingDir}}'
