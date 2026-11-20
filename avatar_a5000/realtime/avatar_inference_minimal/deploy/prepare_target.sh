#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

F="$(command -v ffmpeg)" || {
    echo "Install ffmpeg first"
    exit 1
}

export F

python3 - <<'PY'
import os
import re
from pathlib import Path

p = Path("avatar_system/configs/avatar.yaml")
s = p.read_text()

s, n = re.subn(
    r"(?m)^(\s*ffmpeg:\s*).+$",
    lambda m: m.group(1) + os.environ["F"],
    s
)
assert n == 1
p.write_text(s)

p = Path("avatar_system/configs/musetalk_shubh.yaml")
s = p.read_text()

s, n = re.subn(
    r"(?m)^(gpu:\s*)\d+",
    r"\g<1>0",
    s
)
assert n == 1
p.write_text(s)

print("Configured FFmpeg and MuseTalk GPU 0")
PY
