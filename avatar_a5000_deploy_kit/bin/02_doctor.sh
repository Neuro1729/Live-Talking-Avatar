#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
show_paths
python3 "$KIT_DIR/bin/doctor.py" --root "$AVATAR_ROOT" "$@"
