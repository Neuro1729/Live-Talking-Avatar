#!/usr/bin/env bash
# Patch machine-specific paths in the downloaded copy (not the original gpu20 source).
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_project
require_cmd ffmpeg
FFMPEG="$(command -v ffmpeg)"
export FFMPEG AVATAR_ROOT
python3 - <<'PY'
import os, re
from pathlib import Path
root = Path(os.environ['AVATAR_ROOT'])
files = (
    (root/'avatar_system/configs/avatar.yaml', r'(?m)^(\s*ffmpeg:\s*).+$', lambda m: m.group(1)+os.environ['FFMPEG']),
    (root/'avatar_system/configs/musetalk_shubh.yaml', r'(?m)^(\s*gpu:\s*)\d+[^\n]*$', lambda m: m.group(1)+'0'),
)
for p, regex, repl in files:
    before = p.read_text()
    after, n = re.subn(regex, repl, before)
    if n != 1: raise SystemExit(f'Expected one matching setting in {p}, got {n}')
    backup = p.with_suffix(p.suffix+'.before_a5000')
    if not backup.exists(): backup.write_text(before)
    if before != after: p.write_text(after)
    print('Configured', p)
PY
SRC_ENV="$KIT_DIR/templates/_env_portable.sh"
DST_ENV="$AVATAR_ROOT/avatar_system/scripts/_env.sh"
if [[ -f "$DST_ENV" && ! -f "$DST_ENV.before_a5000" ]]; then
  cp -p "$DST_ENV" "$DST_ENV.before_a5000"
fi
cp "$SRC_ENV" "$DST_ENV"
chmod +x "$DST_ENV"
mkdir -p "$AVATAR_ROOT/avatar_cache" "$AVATAR_ROOT/envs"
KEY_FILE="$AVATAR_ROOT/../.env"
if [[ ! -f "$KEY_FILE" ]]; then
  umask 077
  cp "$KIT_DIR/templates/.env.example" "$KEY_FILE"
  chmod 600 "$KEY_FILE"
  echo "Created KEY TEMPLATE: $KEY_FILE"
  echo "Edit it with your real SARVAM_API_KEY (never paste the key into chat)."
else
  echo "Existing .env preserved: $KEY_FILE"
fi
echo "Patched GPU 0, FFmpeg ($FFMPEG), and portable env for: $AVATAR_ROOT"
