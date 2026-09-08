#!/usr/bin/env bash
set -euo pipefail

package_root="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source_image="${PIPELINE_IMAGE:-zihao/ultralytics-yolo26-workshop:rocm7.2.1-full}"
target_image="${REGISTRY_IMAGE:-}"

[[ -n "$target_image" ]] || {
    echo "REGISTRY_IMAGE must name an explicitly approved release tag" >&2
    exit 1
}
target_name="${target_image##*/}"
[[ "$target_name" == *:* ]] || {
    echo "REGISTRY_IMAGE must include an explicit tag: $target_image" >&2
    exit 1
}
[[ "$target_image" != *@sha256:* && "$target_image" != *:latest ]] || {
    echo "REGISTRY_IMAGE must be a new immutable human-readable tag: $target_image" >&2
    exit 1
}
[[ "${APPROVED_REGISTRY_IMAGE:-}" == "$target_image" ]] || {
    echo "APPROVED_REGISTRY_IMAGE must exactly match REGISTRY_IMAGE" >&2
    exit 1
}

repository_root=$(git -C "$package_root" rev-parse --show-toplevel)
package_relative=$(realpath --relative-to="$repository_root" "$package_root")
dirty=$(git -C "$repository_root" status --porcelain --untracked-files=all -- \
    "$package_relative" .github/workflows/ultralytics-yolo26-smoke.yml)
[[ -z "$dirty" ]] || {
    echo "Refusing to publish from a dirty workshop tree:" >&2
    echo "$dirty" >&2
    exit 1
}

docker image inspect "$source_image" >/dev/null
source_revision=$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$source_image")
source_bundle=$(docker image inspect -f '{{index .Config.Labels "io.ultralytics.workshop.bundle.sha256"}}' "$source_image")
source_release=$(docker image inspect -f '{{index .Config.Labels "io.ultralytics.release.id"}}' "$source_image")
source_companion=$(docker image inspect -f '{{index .Config.Labels "io.ultralytics.companion.digest"}}' "$source_image")
head_revision=$(git -C "$package_root" rev-parse HEAD)
bundle_sha256=$(python3 "$package_root/scripts/workshop_bundle_identity.py" "$package_root")

[[ "$source_revision" == "$head_revision" ]] || {
    echo "Image/source revision mismatch: $source_revision != $head_revision" >&2
    exit 1
}
[[ "$source_bundle" == "$bundle_sha256" ]] || {
    echo "Image/source bundle mismatch: $source_bundle != $bundle_sha256" >&2
    exit 1
}
[[ -n "$source_release" && "$source_release" != "unreleased" && "$source_release" != "<no value>" ]] || {
    echo "Image is missing a release ID" >&2
    exit 1
}
[[ "$source_companion" == *@sha256:* ]] || {
    echo "Image is missing a digest-pinned companion reference" >&2
    exit 1
}
set +e
remote_check=$(docker buildx imagetools inspect "$target_image" 2>&1)
remote_rc=$?
set -e
if [[ "$remote_rc" == "0" ]]; then
    echo "Refusing to overwrite existing remote tag: $target_image" >&2
    exit 1
fi
if ! grep -Eqi 'not found|manifest unknown|name unknown' <<<"$remote_check"; then
    echo "Unable to prove that the remote tag is unused:" >&2
    echo "$remote_check" >&2
    exit 1
fi

docker image inspect "$source_companion" >/dev/null 2>&1 || docker pull "$source_companion"
docker run --rm --entrypoint test "$source_companion" \
    -x /opt/llama.cpp/build/bin/llama-server

PIPELINE_IMAGE="$source_image" bash "$package_root/scripts/test_notebook_image.sh"
printf 'Publishing %s -> %s\n' "$source_image" "$target_image"
printf '  release:   %s\n  revision:  %s\n  bundle:    %s\n  companion: %s\n' \
    "$source_release" "$source_revision" "$source_bundle" "$source_companion"
docker tag "$source_image" "$target_image"
docker push "$target_image"
docker buildx imagetools inspect "$target_image"
