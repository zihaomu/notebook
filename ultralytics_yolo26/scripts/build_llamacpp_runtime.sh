#!/usr/bin/env bash
set -euo pipefail

package_root="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
base_image="${BASE_IMAGE:-crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/muzihao2/work:ultralytics-yolo26-workshop-20260911-sshd}"
expected_base_id="${EXPECTED_BASE_ID:-sha256:ec4984ee98c2d490b14c4389f80acbcdc0f777a1f756c747b1591757ac4eba29}"
source_url="${LLAMACPP_SOURCE_URL:-https://gh-test.anruicloud.com/ggml-org/llama.cpp.git}"
source_commit="${LLAMACPP_SOURCE_COMMIT:-035cd8f9a6dda9cd0224b07d3df2dc95f7a3a31e}"
build_number="${LLAMACPP_BUILD_NUMBER:-9766}"
expected_binary_sha256="${EXPECTED_LLAMACPP_SHA256:-659fe4ac4a226c0c215f820708bc59eaaa660878c5421a759adbbe5ba7a9a679}"
build_jobs="${LLAMACPP_BUILD_JOBS:-16}"
work_dir="${LLAMACPP_WORK_DIR:-$package_root/.build/llama-b9766}"
host_uid=$(id -u)
host_gid=$(id -g)

docker image inspect "$base_image" >/dev/null 2>&1 || docker pull "$base_image"
actual_base_id=$(docker image inspect "$base_image" --format '{{.Id}}')
if [[ "$actual_base_id" != "$expected_base_id" ]]; then
    echo "Unexpected SSHD base image ID: $actual_base_id != $expected_base_id" >&2
    exit 1
fi

mkdir -p "$work_dir"
docker run --rm \
    -v "$work_dir:/work" \
    --entrypoint /bin/bash \
    "$base_image" \
    -lc "rm -rf /work/src /work/build /work/runtime; chown -R $host_uid:$host_gid /work"
mkdir -p "$work_dir/src" "$work_dir/build" "$work_dir/runtime"

docker run --rm --network host \
    --user "$host_uid:$host_gid" \
    -e HOME=/tmp \
    -e LLAMACPP_SOURCE_URL="$source_url" \
    -e LLAMACPP_SOURCE_COMMIT="$source_commit" \
    -e LLAMACPP_BUILD_NUMBER="$build_number" \
    -e EXPECTED_LLAMACPP_SHA256="$expected_binary_sha256" \
    -e LLAMACPP_BUILD_JOBS="$build_jobs" \
    -v "$work_dir:/work" \
    --entrypoint /bin/bash \
    "$base_image" \
    -lc '
        set -euo pipefail
        git init -q /work/src
        git -C /work/src remote add origin "$LLAMACPP_SOURCE_URL"
        git -C /work/src -c http.version=HTTP/1.1 fetch --depth 1 origin "$LLAMACPP_SOURCE_COMMIT"
        git -C /work/src checkout -q --detach FETCH_HEAD
        test "$(git -C /work/src rev-parse HEAD)" = "$LLAMACPP_SOURCE_COMMIT"

        cmake -S /work/src -B /work/build -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
            -DBUILD_SHARED_LIBS=OFF \
            -DGGML_HIP=ON \
            -DAMDGPU_TARGETS=gfx1100 \
            -DGGML_NATIVE=OFF \
            -DGGML_CCACHE=OFF \
            -DLLAMA_BUILD_NUMBER="$LLAMACPP_BUILD_NUMBER" \
            -DLLAMA_BUILD_COMMIT="${LLAMACPP_SOURCE_COMMIT:0:9}" \
            -DLLAMA_BUILD_SERVER=ON \
            -DLLAMA_BUILD_TESTS=OFF \
            -DLLAMA_BUILD_TOOLS=ON \
            -DLLAMA_BUILD_EXAMPLES=OFF \
            -DLLAMA_BUILD_APP=OFF \
            -DLLAMA_BUILD_UI=OFF \
            -DLLAMA_USE_PREBUILT_UI=OFF \
            -DLLAMA_OPENSSL=OFF
        cmake --build /work/build --target llama-server -j"$LLAMACPP_BUILD_JOBS"

        install -m 0755 /work/build/bin/llama-server /work/runtime/llama-server
        install -m 0644 /work/src/LICENSE /work/runtime/LLAMA_CPP_LICENSE
        printf "%s\n" "$LLAMACPP_SOURCE_COMMIT" > /work/runtime/LLAMA_CPP_COMMIT
        /work/runtime/llama-server --version 2>&1 | tee /work/runtime/version.txt
        grep -F "version: $LLAMACPP_BUILD_NUMBER (${LLAMACPP_SOURCE_COMMIT:0:9})" /work/runtime/version.txt
        ldd /work/runtime/llama-server | tee /work/runtime/ldd.txt
        test -z "$(grep -E "not found|lib(llama|ggml|mtmd|server)" /work/runtime/ldd.txt || true)"
        sha256sum /work/runtime/llama-server | tee /work/runtime/SHA256SUMS
        test "$(sha256sum /work/runtime/llama-server | cut -d" " -f1)" = "$EXPECTED_LLAMACPP_SHA256"
    '

echo "LLAMACPP_RUNTIME_BUILD=PASS"
echo "binary=$work_dir/runtime/llama-server"
echo "sha256=$expected_binary_sha256"
