#!/usr/bin/env bash
set -euo pipefail

package_root="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source_image="${SSHD_IMAGE:-zihao/ultralytics-yolo26-workshop:2026-09-11-sshd-candidate}"
target_image="${REGISTRY_IMAGE:-}"

[[ -n "$target_image" ]] || { echo "REGISTRY_IMAGE is required" >&2; exit 1; }
[[ "${APPROVED_REGISTRY_IMAGE:-}" == "$target_image" ]] || {
    echo "APPROVED_REGISTRY_IMAGE must exactly match REGISTRY_IMAGE" >&2
    exit 1
}
[[ "$target_image" != *:latest && "$target_image" != *@sha256:* ]] || {
    echo "REGISTRY_IMAGE must use a new immutable human-readable tag" >&2
    exit 1
}

repository_root=$(git -C "$package_root" rev-parse --show-toplevel)
package_relative=$(realpath --relative-to="$repository_root" "$package_root")
dirty=$(git -C "$repository_root" status --porcelain --untracked-files=all -- "$package_relative")
[[ -z "$dirty" ]] || {
    echo "Refusing to publish SSHD image from a dirty workshop tree:" >&2
    echo "$dirty" >&2
    exit 1
}

docker image inspect "$source_image" >/dev/null
source_commit=$(docker image inspect "$source_image" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')
head_commit=$(git -C "$repository_root" rev-parse HEAD)
[[ "$source_commit" == "$head_commit" ]] || {
    echo "SSHD image/source revision mismatch: $source_commit != $head_commit" >&2
    exit 1
}
[[ "$(docker image inspect "$source_image" --format '{{index .Config.Labels "io.ultralytics.sshd.enabled"}}')" == "true" ]]
[[ "$(docker image inspect "$source_image" --format '{{.Config.User}}')" == "root" ]]
[[ "$(docker image inspect "$source_image" --format '{{.Config.WorkingDir}}')" == "/app" ]]

set +e
remote_check=$(docker buildx imagetools inspect "$target_image" 2>&1)
remote_rc=$?
set -e
if [[ "$remote_rc" == 0 ]]; then
    echo "Refusing to overwrite existing remote tag: $target_image" >&2
    exit 1
fi
if ! grep -Eqi 'not found|manifest unknown|name unknown' <<<"$remote_check"; then
    echo "Unable to prove that the remote tag is unused:" >&2
    echo "$remote_check" >&2
    exit 1
fi

docker tag "$source_image" "$target_image"
docker push "$target_image"
docker buildx imagetools inspect "$target_image"
