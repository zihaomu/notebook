#!/usr/bin/env bash
set -euo pipefail

source_dir="${ULTRALYTICS_BAKED_MODEL_DIR:-/opt/ultralytics-yolo26/models}"
target_dir="${1:-/model-volume}"
manifest="$source_dir/SHA256SUMS"
model_set_id="$source_dir/MODEL_SET_ID"

[[ -f "$manifest" && -f "$model_set_id" ]] || {
    echo "Missing baked model manifest: $manifest" >&2
    exit 1
}
mkdir -p "$target_dir"
expected_model_set_id=$(cat "$model_set_id")
if [[ -f "$target_dir/MODEL_SET_ID" ]] && \
        [[ "$(cat "$target_dir/MODEL_SET_ID")" == "$expected_model_set_id" ]]; then
    complete=true
    while read -r _ relative; do
        if [[ ! -s "$target_dir/$relative" ]]; then
            complete=false
            break
        fi
    done < "$manifest"
    if [[ "$complete" == true ]]; then
        echo "MODEL_VOLUME_READY=$target_dir (cached)"
        exit 0
    fi
fi

while read -r expected relative; do
    [[ -n "$expected" && -n "$relative" ]] || continue
    source="$source_dir/$relative"
    target="$target_dir/$relative"
    [[ -f "$source" ]] || {
        echo "Missing baked model: $source" >&2
        exit 1
    }
    if [[ -f "$target" ]] && [[ "$(sha256sum "$target" | cut -d' ' -f1)" == "$expected" ]]; then
        continue
    fi
    mkdir -p "$(dirname "$target")"
    temporary="$target.part.$$"
    rm -f "$temporary"
    cp "$source" "$temporary"
    actual=$(sha256sum "$temporary" | cut -d' ' -f1)
    [[ "$actual" == "$expected" ]] || {
        rm -f "$temporary"
        echo "Model checksum mismatch: $relative" >&2
        exit 1
    }
    chmod 0644 "$temporary"
    mv -f "$temporary" "$target"
done < "$manifest"

cp "$manifest" "$target_dir/SHA256SUMS.part.$$"
mv -f "$target_dir/SHA256SUMS.part.$$" "$target_dir/SHA256SUMS"
cp "$model_set_id" "$target_dir/MODEL_SET_ID.part.$$"
mv -f "$target_dir/MODEL_SET_ID.part.$$" "$target_dir/MODEL_SET_ID"
echo "MODEL_VOLUME_READY=$target_dir"
