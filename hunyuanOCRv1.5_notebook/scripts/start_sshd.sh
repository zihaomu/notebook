#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${HUNYUANOCR_DEMO_ROOT:-/hunyuanOCR_workspace/demo}
SSH_PORT=${SSH_PORT:-2222}
AUTHORIZED_KEYS_SOURCE=${SSH_AUTHORIZED_KEYS_FILE:-$HOME/.ssh/authorized_keys}
SSH_STATE_DIR=${SSH_STATE_DIR:-/hunyuanOCR_workspace/cache/ssh}
PID_FILE=$SSH_STATE_DIR/sshd.pid
HOST_KEY=$SSH_STATE_DIR/ssh_host_ed25519_key
AUTHORIZED_KEYS=$SSH_STATE_DIR/authorized_keys
LOG_FILE=$ROOT/outputs/logs/sshd.log

if [[ ! "$SSH_PORT" =~ ^[0-9]+$ ]] || (( SSH_PORT < 1024 || SSH_PORT > 65535 )); then
    echo "ERROR: SSH_PORT must be between 1024 and 65535 for non-root sshd" >&2
    exit 64
fi
[[ -s "$AUTHORIZED_KEYS_SOURCE" ]] || {
    echo "ERROR: SSH authorized keys file is missing or empty: $AUTHORIZED_KEYS_SOURCE" >&2
    exit 66
}

mkdir -p "$SSH_STATE_DIR" "$ROOT/outputs/logs"
chmod 0700 "$SSH_STATE_DIR"
install -m 0600 "$AUTHORIZED_KEYS_SOURCE" "$AUTHORIZED_KEYS"
ssh-keygen -l -f "$AUTHORIZED_KEYS" >/dev/null
if [[ ! -f "$HOST_KEY" ]]; then
    ssh-keygen -q -t ed25519 -N "" -f "$HOST_KEY"
fi
chmod 0600 "$HOST_KEY"

if [[ -f "$PID_FILE" ]]; then
    pid=$(<"$PID_FILE")
    state=
    if [[ "$pid" =~ ^[0-9]+$ && -r "/proc/${pid}/stat" ]]; then
        state=$(awk '{print $3}' "/proc/${pid}/stat")
    fi
    if [[ -n "$state" && "$state" != Z ]]; then
        if ssh-keyscan -T 2 -p "$SSH_PORT" 127.0.0.1 >/dev/null 2>&1; then
            echo "READY: sshd already serves hunyuanocr on port $SSH_PORT"
            exit 0
        fi
        echo "ERROR: sshd pid $pid is alive (state=$state) but port $SSH_PORT is not ready" >&2
        exit 1
    fi
    rm -f "$PID_FILE"
fi

nohup /usr/sbin/sshd -D -e -p "$SSH_PORT" -h "$HOST_KEY" \
    -o "PidFile=$PID_FILE" \
    -o "AuthorizedKeysFile=$AUTHORIZED_KEYS" \
    -o StrictModes=yes \
    -o AuthenticationMethods=publickey \
    -o PubkeyAuthentication=yes \
    -o PasswordAuthentication=no \
    -o KbdInteractiveAuthentication=no \
    -o UsePAM=no \
    -o PermitRootLogin=no \
    -o PermitEmptyPasswords=no \
    -o PermitUserEnvironment=no \
    -o X11Forwarding=no \
    -o AllowUsers=hunyuanocr \
    -o "SetEnv=HUNYUANOCR_WORKSPACE=/hunyuanOCR_workspace MODEL_DIR=/hunyuanOCR_workspace/models/HunyuanOCR MIOPEN_FIND_MODE=2 VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-ROCM_ATTN} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0} PYTHONPATH=/hunyuanOCR_workspace/demo PATH=/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin XDG_CACHE_HOME=/hunyuanOCR_workspace/cache/xdg VLLM_CACHE_ROOT=/hunyuanOCR_workspace/cache/vllm MIOPEN_USER_DB_PATH=/hunyuanOCR_workspace/cache/miopen-db MIOPEN_CUSTOM_CACHE_DIR=/hunyuanOCR_workspace/cache/miopen-kernels TORCHINDUCTOR_CACHE_DIR=/hunyuanOCR_workspace/cache/torchinductor" \
    </dev/null >> "$LOG_FILE" 2>&1 &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"

for _ in {1..40}; do
    if ssh-keyscan -T 2 -p "$SSH_PORT" 127.0.0.1 >/dev/null 2>&1; then
        printf 'READY: sshd pid=%s user=hunyuanocr port=%s\n' "$pid" "$SSH_PORT"
        exit 0
    fi
    state=
    if [[ -r "/proc/${pid}/stat" ]]; then
        state=$(awk '{print $3}' "/proc/${pid}/stat")
    fi
    if [[ -z "$state" || "$state" == Z ]]; then
        tail -n 80 "$LOG_FILE" >&2 || true
        exit 1
    fi
    sleep 0.25
done
tail -n 80 "$LOG_FILE" >&2 || true
exit 1
