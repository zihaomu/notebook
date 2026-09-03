#!/usr/bin/env bash
set -euo pipefail

root="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
hipcc="${HIPCC:-/opt/rocm/bin/hipcc}"
arch="${AMDGPU_TARGET:-gfx1100}"
output="${OUTPUT:-$root/native/build/vaapi-hip-encode-probe}"
mkdir -p "$(dirname "$output")"

"$hipcc" \
    -std=c++17 -O3 --offload-arch="$arch" \
    "$root/native/vaapi_hip_encode_probe.cpp" \
    -o "$output" \
    $(pkg-config --cflags --libs libavcodec libavformat libavutil libva libva-drm libdrm)

chmod 0755 "$output"
printf '%s\n' "$output"
