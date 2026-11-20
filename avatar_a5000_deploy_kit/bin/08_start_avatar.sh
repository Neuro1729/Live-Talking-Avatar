#!/usr/bin/env bash
# Run on physical GPU 0. Does not kill any other server or tmux process.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_project
KEY_FILE="$AVATAR_ROOT/.env"
if [[ -f "$KEY_FILE" ]]; then
  # Never print the API key. The file should contain simple KEY=value lines.
  set -a
  source "$KEY_FILE"
  set +a
fi
[[ -n "${SARVAM_API_KEY:-}" && "${SARVAM_API_KEY}" != REPLACE_WITH_YOUR_SARVAM_API_KEY ]] || {
  echo "ERROR: Configure SARVAM_API_KEY in $KEY_FILE" >&2; exit 1;
}
[[ -x "$AVATAR_ROOT/envs/musetalk/bin/python" ]] || {
  echo "ERROR: MuseTalk environment missing. See README step 4." >&2; exit 1;
}
export GPU="${GPU:-0}"
export AVATAR_CFG_MUSETALK=musetalk_shubh
export PYTHONPATH="$AVATAR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$AVATAR_ROOT"
PORT="${PORT:-8782}"
if command -v ss >/dev/null 2>&1 && ss -ltn | grep -Eq ":${PORT}[[:space:]]"; then
  echo "ERROR: Port $PORT is already in use. Set PORT to a free port; no process was killed." >&2
  exit 1
fi

echo "Starting graph-aware MuseTalk + Sarvam LLM/TTS + ${STT_MODE:-local} STT on GPU $GPU"
exec bash avatar_system/scripts/run_conversation.sh \
  --port "$PORT" \
  --graph motion_graph_silent_7f1cea162602.json \
  --llm sarvam \
  --tts sarvam-ws \
  --stt "${STT_MODE:-local}" \
  --stt-url "${STT_URL:-http://127.0.0.1:8310}" \
  --batch 2 \
  --buffer 2 \
  --ack-delay 0.5 \
  "$@"
