#!/usr/bin/env bash
set -euo pipefail

root="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python="${PYTHON:-/opt/venv/bin/python3}"
hipcc="${HIPCC:-/opt/rocm/bin/hipcc}"
arch="${AMDGPU_TARGET:-gfx1100}"
output_dir="${OUTPUT_DIR:-$root/native/build}"
extension_suffix="$($python-config --extension-suffix 2>/dev/null || "$python" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
mkdir -p "$output_dir"

"$hipcc" \
    -std=c++17 -O3 -shared -fPIC --offload-arch="$arch" \
    $($python -m pybind11 --includes) \
    "$root/native/hip_vaapi_bridge.cpp" \
    -o "$output_dir/hip_vaapi_bridge$extension_suffix" \
    $(pkg-config --cflags --libs libavcodec libavformat libavutil libva libva-drm libdrm)

printf '%s\n' "$output_dir/hip_vaapi_bridge$extension_suffix"
