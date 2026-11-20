#!/usr/bin/env bash
# Run on A5000 server with rclone configured for the user's Drive.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_cmd rclone
REMOTE="${AVATAR_REMOTE:-gdrive:avatar_inference_minimal}"
mkdir -p "$AVATAR_ROOT"
show_paths
echo "Copying private inference package from $REMOTE"
rclone copy "$REMOTE" "$AVATAR_ROOT" \
  --progress --transfers 2 --checkers 2 \
  --tpslimit 2 --tpslimit-burst 2 \
  --retries 10 --low-level-retries 20
require_project
echo "Download succeeded. Run bin/02_doctor.sh (or python3 bin/doctor.py)."
