#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

test -x envs/analysis/bin/python || {
    echo "Create envs/analysis on this machine first"
    exit 1
}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

exec envs/analysis/bin/python \
    musetalk_ft/scripts/stt_server.py "$@"
