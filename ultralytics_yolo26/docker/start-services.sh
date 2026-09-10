#!/usr/bin/env bash
set -euo pipefail

case "${SERVICE_MODE:-all}" in
    all)
        /usr/local/bin/start-ultralytics-yolo26-sshd "${SSH_PORT:-22}"
        exec /usr/local/bin/start-ultralytics-yolo26-jupyter
        ;;
    jupyter)
        exec /usr/local/bin/start-ultralytics-yolo26-jupyter
        ;;
    sshd)
        export FOREGROUND=1
        exec /usr/local/bin/start-ultralytics-yolo26-sshd "${SSH_PORT:-22}"
        ;;
    *)
        echo "SERVICE_MODE must be one of: all, jupyter, sshd" >&2
        exit 2
        ;;
esac
