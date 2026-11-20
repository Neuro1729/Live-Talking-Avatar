#!/usr/bin/env bash
set -Eeuo pipefail
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AVATAR_ROOT="${AVATAR_ROOT:-$HOME/a5000_avatar/realtime}"
export KIT_DIR AVATAR_ROOT

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing command: $1" >&2; exit 1; }
}

require_project() {
  [[ -f "$AVATAR_ROOT/avatar_system/streaming/server.py" ]] || {
    echo "ERROR: avatar code missing at $AVATAR_ROOT" >&2
    echo "Run: bash '$KIT_DIR/bin/01_download_minimal.sh'" >&2
    exit 1
  }
}

show_paths() {
  echo "Kit: $KIT_DIR"
  echo "Project: $AVATAR_ROOT"
}
