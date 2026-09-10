#!/usr/bin/env bash
set -euo pipefail

port="${1:-${SSH_PORT:-22}}"
state_dir="${SSH_STATE_DIR:-/run/ultralytics-yolo26-sshd}"
authorized_keys="$state_dir/authorized_keys"
host_key="$state_dir/ssh_host_ed25519_key"

if [[ "$(id -u)" != "0" ]]; then
    echo "[sshd] root is required to start sshd" >&2
    exit 1
fi
if [[ ! "$port" =~ ^[0-9]+$ ]] || (( port < 1 || port > 65535 )); then
    echo "[sshd] invalid port: $port" >&2
    exit 1
fi

install -d -m 0755 /run/sshd
install -d -m 0700 "$state_dir"
if [[ ! -s "$host_key" ]]; then
    ssh-keygen -q -t ed25519 -N '' -f "$host_key"
fi

: > "$authorized_keys"
if [[ -n "${SSH_AUTHORIZED_KEYS_FILE:-}" ]]; then
    [[ -s "$SSH_AUTHORIZED_KEYS_FILE" ]] || {
        echo "[sshd] authorized keys file is missing or empty: $SSH_AUTHORIZED_KEYS_FILE" >&2
        exit 1
    }
    cat "$SSH_AUTHORIZED_KEYS_FILE" >> "$authorized_keys"
fi
if [[ -n "${SSH_PUBLIC_KEY:-${SSH_PUBKEY:-}}" ]]; then
    printf '%s\n' "${SSH_PUBLIC_KEY:-${SSH_PUBKEY}}" >> "$authorized_keys"
fi
chmod 0600 "$authorized_keys"

password_auth=no
if [[ -n "${ROOT_PASSWORD:-}" ]]; then
    printf 'root:%s\n' "$ROOT_PASSWORD" | chpasswd
    password_auth=yes
fi

if [[ -s "$authorized_keys" ]]; then
    ssh-keygen -l -f "$authorized_keys" >/dev/null
    echo "[sshd] public-key authentication enabled"
elif [[ "$password_auth" == "no" ]]; then
    echo "[sshd] warning: no SSH_PUBLIC_KEY, SSH_AUTHORIZED_KEYS_FILE, or ROOT_PASSWORD; login is disabled" >&2
fi

session_env=(
    "PATH=${PATH:-/opt/rocm/llvm/bin:/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
    "PYTHONPATH=${PYTHONPATH:-/opt/opencv5/lib/python3.10/site-packages:/opt/rocm/lib:/workspace:/workspace/src:/workspace/scripts}"
    "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/opt/opencv5/lib:/opt/rocm/lib:/usr/local/lib}"
    "ROCM_PATH=${ROCM_PATH:-/opt/rocm}"
    "ULTRALYTICS_YOLO26_ROOT=${ULTRALYTICS_YOLO26_ROOT:-/workspace}"
    "ULTRALYTICS_YOLO26_MODEL_DIR=${ULTRALYTICS_YOLO26_MODEL_DIR:-/opt/ultralytics-yolo26/models}"
    "ULTRALYTICS_YOLO26_OUTPUT_DIR=${ULTRALYTICS_YOLO26_OUTPUT_DIR:-/workspace/output}"
    "ULTRALYTICS_MIGRAPHX_CACHE_ROOT=${ULTRALYTICS_MIGRAPHX_CACHE_ROOT:-/opt/ultralytics-yolo26/models/ort-migraphx-cache}"
    "LLAMACPP_ROOT_URL=${LLAMACPP_ROOT_URL:-http://127.0.0.1:8199}"
    "LLAMACPP_BASE_URL=${LLAMACPP_BASE_URL:-http://127.0.0.1:8199/v1}"
    "NO_PROXY=127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
    "no_proxy=127.0.0.1,localhost${no_proxy:+,$no_proxy}"
)
for variable in ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES VAAPI_DEVICE HF_ENDPOINT; do
    if [[ -n "${!variable:-}" ]]; then
        session_env+=("$variable=${!variable}")
    fi
done

options=(
    -p "$port"
    -h "$host_key"
    -o "SetEnv=${session_env[*]}"
    -o "PidFile=$state_dir/sshd.pid"
    -o "AuthorizedKeysFile=$authorized_keys"
    -o "PasswordAuthentication=$password_auth"
    -o "KbdInteractiveAuthentication=no"
    -o "ChallengeResponseAuthentication=no"
    -o "PubkeyAuthentication=yes"
    -o "PermitRootLogin=yes"
    -o "PermitEmptyPasswords=no"
    -o "UsePAM=no"
    -o "X11Forwarding=no"
)

/usr/sbin/sshd -t "${options[@]}"
if [[ "${FOREGROUND:-0}" == "1" ]]; then
    echo "[sshd] listening on port $port (foreground)"
    exec /usr/sbin/sshd -D -e "${options[@]}"
fi

if [[ -s "$state_dir/sshd.pid" ]]; then
    kill "$(cat "$state_dir/sshd.pid")" 2>/dev/null || true
    rm -f "$state_dir/sshd.pid"
fi
/usr/sbin/sshd "${options[@]}"
echo "[sshd] listening on port $port"
