#!/usr/bin/env bash
# Run on ORIGINAL gpu20; uploads the genuine custom preprocessor, not training data.
set -Eeuo pipefail
SRC="${AVATAR_SOURCE:-/var/autofs/nfs10g/nest10/usrs/vishal/Aaryan/avatar/realtime}"
REMOTE="${AVATAR_REMOTE:-gdrive:avatar_inference_minimal}"
command -v rclone >/dev/null || { echo 'ERROR: rclone not found' >&2; exit 1; }
for f in scripts/prep_avatar.py scripts/__init__.py; do
  if [[ -f "$SRC/$f" ]]; then
    echo "Uploading $f"
    rclone copyto "$SRC/$f" "$REMOTE/$f" \
      --progress --transfers 1 --checkers 1 \
      --tpslimit 1 --tpslimit-burst 1 --retries 10
  fi
done
[[ -s "$SRC/scripts/prep_avatar.py" ]] || {
  echo "ERROR: $SRC/scripts/prep_avatar.py missing" >&2; exit 1;
}
echo "=== Local import references (check if any extra local modules are needed) ==="
python3 - "$SRC/scripts/prep_avatar.py" <<'PY'
import ast, sys
from pathlib import Path
p = Path(sys.argv[1]); tree = ast.parse(p.read_text())
for n in ast.walk(tree):
    if isinstance(n, ast.ImportFrom):
        print('from', '.' * n.level + (n.module or ''), 'import', ','.join(a.name for a in n.names))
    elif isinstance(n, ast.Import):
        for a in n.names: print('import', a.name)
PY
echo "Done. Preprocessing script is now on Drive; it was absent from the 3.358 GiB upload."
echo "If it imports further project-local root scripts, upload those as well."
