#!/usr/bin/env bash
# Optional: creates a separate local Whisper STT environment for GPU 1.
# Keeps contemporary transformers/torch isolated from the old MuseTalk environment.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_project
PY="${PYTHON_310:-python3.10}"
require_cmd "$PY"
ENV="$AVATAR_ROOT/envs/stt"
if [[ ! -x "$ENV/bin/python" ]]; then
  "$PY" -m venv "$ENV"
fi
"$ENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV/bin/python" -m pip install \
  'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu121
"$ENV/bin/python" -m pip install \
  'transformers>=4.46,<5' 'accelerate>=0.34,<2' \
  'fastapi>=0.110,<1' 'uvicorn>=0.29,<1' \
  'soundfile>=0.12,<1' 'safetensors>=0.4,<1' \
  'sentencepiece>=0.2,<1'
"$ENV/bin/python" - <<'PY'
import torch, transformers, fastapi, soundfile
print('STT environment OK: torch', torch.__version__, 'transformers', transformers.__version__)
print('CUDA available:', torch.cuda.is_available())
PY
