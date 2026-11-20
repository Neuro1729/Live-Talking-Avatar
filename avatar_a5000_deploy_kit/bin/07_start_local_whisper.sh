#!/usr/bin/env bash
# Local Whisper STT runs on physical GPU 1; API binds loopback port 8310.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_project
PY="$AVATAR_ROOT/envs/stt/bin/python"
[[ -x "$PY" ]] || {
  echo "ERROR: Missing STT env. Run: bash '$KIT_DIR/bin/04_install_stt_env.sh'" >&2
  exit 1
}
export CUDA_VISIBLE_DEVICES="${STT_GPU:-1}"
export HF_HOME="${HF_HOME:-$AVATAR_ROOT/.cache/huggingface}"
export PYTHONPATH="$AVATAR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_HOME"
cd "$AVATAR_ROOT/musetalk_ft"
echo "Starting local Hindi/English Whisper on GPU ${STT_GPU:-1}, port ${STT_PORT:-8310}"
echo 'First start downloads Hindi Vaani and English Whisper large-v3-turbo models; requires internet.'
exec "$PY" scripts/stt_server.py --port "${STT_PORT:-8310}"
