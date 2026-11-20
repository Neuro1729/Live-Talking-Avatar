#!/usr/bin/env bash
# Best-effort bootstrap from the working gpu20 pip snapshots.
# The original MuseTalk uses torch 2.0.1 + cu118 and possibly compiled MMCV; compatibility must be validated.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_project
PY="${PYTHON_310:-python3.10}"
require_cmd "$PY"
for n in musetalk analysis; do
  freeze="$AVATAR_ROOT/requirements_${n}_installed.txt"
  [[ -s "$freeze" ]] || { echo "ERROR: no snapshot: $freeze" >&2; exit 1; }
  if grep -nE '(^-e |@ file:|@ /|git\+|/var/autofs/)' "$freeze"; then
    echo "ERROR: $freeze contains nonportable requirements. Review/edit a copy before installing." >&2
    exit 1
  fi
  env="$AVATAR_ROOT/envs/$n"
  [[ -x "$env/bin/python" ]] || "$PY" -m venv "$env"
  "$env/bin/python" -m pip install --upgrade pip wheel setuptools
  echo "Installing the captured $n environment; failure may require MMCV/CUDA wheel adjustments."
  "$env/bin/python" -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r "$freeze"
done
echo 'Captured environments installed. Test GPU inference before launching the website.'
