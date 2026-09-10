#!/usr/bin/env bash
set -euo pipefail

package_root="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
base_image="${BASE_IMAGE:-crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/muzihao2/work@sha256:ec4984ee98c2d490b14c4389f80acbcdc0f777a1f756c747b1591757ac4eba29}"
image="${UNIFIED_IMAGE:-zihao/ultralytics-yolo26-workshop:2026-09-11-unified-local}"
runtime_dir="${LLAMACPP_RUNTIME_DIR:-$package_root/.build/llama-b9766/runtime}"
llamacpp_commit="${LLAMACPP_SOURCE_COMMIT:-035cd8f9a6dda9cd0224b07d3df2dc95f7a3a31e}"
llamacpp_sha256="${LLAMACPP_BINARY_SHA256:-659fe4ac4a226c0c215f820708bc59eaaa660878c5421a759adbbe5ba7a9a679}"
repository_root=$(git -C "$package_root" rev-parse --show-toplevel)
package_relative=$(realpath --relative-to="$repository_root" "$package_root")
dirty=$(git -C "$repository_root" status --porcelain --untracked-files=all -- "$package_relative")

[[ -z "$dirty" ]] || {
    echo "Refusing to build unified image from a dirty workshop tree:" >&2
    echo "$dirty" >&2
    exit 1
}
[[ -x "$runtime_dir/llama-server" ]] || {
    echo "Missing llama-server runtime; run scripts/build_llamacpp_runtime.sh first." >&2
    exit 1
}
[[ -s "$runtime_dir/LLAMA_CPP_LICENSE" && -s "$runtime_dir/LLAMA_CPP_COMMIT" ]] || {
    echo "Incomplete llama.cpp runtime metadata in $runtime_dir" >&2
    exit 1
}
[[ "$(cat "$runtime_dir/LLAMA_CPP_COMMIT")" == "$llamacpp_commit" ]]
echo "$llamacpp_sha256  $runtime_dir/llama-server" | sha256sum -c -

source_commit=$(git -C "$repository_root" rev-parse HEAD)
unified_layer_sha256=$(
    cat \
        "$package_root/docker/Dockerfile.unified" \
        "$package_root/docker/start-llamacpp.sh" \
        "$package_root/docker/start-unified-services.sh" \
        "$package_root/docker/start-sshd.sh" \
        | sha256sum | cut -d' ' -f1
)

docker image inspect "$base_image" >/dev/null 2>&1 || docker pull "$base_image"
docker buildx build \
    --load \
    --file "$package_root/docker/Dockerfile.unified" \
    --build-context "llamacpp_runtime=$runtime_dir" \
    --build-arg "BASE_IMAGE=$base_image" \
    --build-arg "UNIFIED_LAYER_SHA256=$unified_layer_sha256" \
    --build-arg "UNIFIED_SOURCE_COMMIT=$source_commit" \
    --build-arg "LLAMACPP_BINARY_SHA256=$llamacpp_sha256" \
    --build-arg "LLAMACPP_SOURCE_COMMIT=$llamacpp_commit" \
    --tag "$image" \
    "$package_root/docker"

[[ "$(docker image inspect "$image" --format '{{.Config.User}}')" == root ]]
[[ "$(docker image inspect "$image" --format '{{.Config.WorkingDir}}')" == /app ]]
[[ "$(docker image inspect "$image" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')" == "$source_commit" ]]
[[ "$(docker image inspect "$image" --format '{{index .Config.Labels "io.ultralytics.unified.enabled"}}')" == true ]]
[[ "$(docker image inspect "$image" --format '{{index .Config.Labels "io.ultralytics.unified.layer.sha256"}}')" == "$unified_layer_sha256" ]]
[[ "$(docker image inspect "$image" --format '{{index .Config.Labels "io.ultralytics.llamacpp.binary.sha256"}}')" == "$llamacpp_sha256" ]]

docker run --rm --entrypoint /bin/bash "$image" -lc '
    set -e
    test -x /usr/local/bin/llama-server
    test -x /usr/local/bin/start-ultralytics-yolo26-llamacpp
    test -x /usr/local/bin/start-ultralytics-yolo26-services
    test -x /usr/local/bin/start-ultralytics-yolo26-sshd
    test -x /usr/local/bin/start-ultralytics-yolo26-jupyter
    test -s /opt/ultralytics-yolo26/models/Qwen3-VL-8B-Instruct-Q8_0.gguf
    test -s /opt/ultralytics-yolo26/models/mmproj-F16.gguf
    test "$LLAMACPP_BASE_URL" = http://127.0.0.1:8199/v1
'

echo "UNIFIED_IMAGE_BUILD=PASS"
echo "image=$image"
echo "source_commit=$source_commit"
echo "layer_sha256=$unified_layer_sha256"
