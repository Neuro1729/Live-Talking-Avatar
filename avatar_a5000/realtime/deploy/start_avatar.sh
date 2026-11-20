#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

test -x envs/musetalk/bin/python || {
    echo "Create envs/musetalk on this machine first"
    exit 1
}

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export AVATAR_CFG_MUSETALK=musetalk_shubh
export GPU="${GPU:-0}"

exec bash avatar_system/scripts/run_conversation.sh \
    --port "${PORT:-8782}" \
    --graph motion_graph_silent_7f1cea162602.json \
    --llm sarvam \
    --tts sarvam-ws \
    --stt "${STT_MODE:-local}" \
    --stt-url "${STT_URL:-http://127.0.0.1:8310}" \
    --batch 2 \
    --buffer 2 \
    --ack-delay 0.5 \
    "$@"
