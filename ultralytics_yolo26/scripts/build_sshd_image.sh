#!/usr/bin/env bash
set -euo pipefail

package_root="${PACKAGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
base_image="${BASE_IMAGE:-crpi-a7t9nblyxh55vyd2.cn-shanghai.personal.cr.aliyuncs.com/muzihao2/work@sha256:cdf4c926066cc75bdcf54371b675fc4aeb61283b8315e8a0fe6aae394f37748e}"
image="${SSHD_IMAGE:-zihao/ultralytics-yolo26-workshop:2026-09-11-sshd-candidate}"
repository_root=$(git -C "$package_root" rev-parse --show-toplevel)
package_relative=$(realpath --relative-to="$repository_root" "$package_root")
dirty=$(git -C "$repository_root" status --porcelain --untracked-files=all -- "$package_relative")
[[ -z "$dirty" ]] || {
    echo "Refusing to build SSHD image from a dirty workshop tree:" >&2
    echo "$dirty" >&2
    exit 1
}
source_commit=$(git -C "$repository_root" rev-parse HEAD)

docker image inspect "$base_image" >/dev/null 2>&1 || docker pull "$base_image"
sshd_layer_sha256=$(
    cat \
        "$package_root/docker/Dockerfile.sshd" \
        "$package_root/docker/start-sshd.sh" \
        "$package_root/docker/start-services.sh" \
        | sha256sum | cut -d' ' -f1
)
docker buildx build \
    --load \
    --file "$package_root/docker/Dockerfile.sshd" \
    --build-arg "BASE_IMAGE=$base_image" \
    --build-arg "SSHD_LAYER_SHA256=$sshd_layer_sha256" \
    --build-arg "SSHD_SOURCE_COMMIT=$source_commit" \
    --tag "$image" \
    "$package_root/docker"

docker image inspect "$image" --format \
    'Built {{index .RepoTags 0}} ({{.Id}}), sshd={{index .Config.Labels "io.ultralytics.sshd.enabled"}}'

test "$(docker image inspect "$image" --format '{{.Config.User}}')" = "root"
test "$(docker image inspect "$image" --format '{{.Config.WorkingDir}}')" = "/app"
test "$(docker image inspect "$image" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}')" = "$source_commit"
docker run --rm --entrypoint /bin/bash "$image" -lc '
    set -e
    test "$(id -u)" = 0
    test -d /app && test -w /app
    test -x /bin/bash
    test -x /opt/venv/bin/jupyter
    test -x /opt/venv/bin/jupyter-lab
    command -v jupyter >/dev/null
    command -v jupyter-lab >/dev/null
'
echo "SSHD_PLATFORM_CONTRACT=PASS"
