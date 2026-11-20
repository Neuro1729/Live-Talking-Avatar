#!/usr/bin/env bash
# Regenerate only the 26 runtime caches, preserving the exact shipped motion graph.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_project
PY="$AVATAR_ROOT/envs/musetalk/bin/python"
PREP="$AVATAR_ROOT/scripts/prep_avatar.py"
[[ -x "$PY" ]] || { echo "ERROR: MuseTalk Python env missing: $PY" >&2; exit 1; }
[[ -f "$PREP" ]] || {
  echo "ERROR: original custom preprocessor missing: $PREP" >&2
  echo "Upload it from gpu20 using bin/00_upload_missing_preprocessor_gpu20.sh, then redownload." >&2
  exit 1
}
for f in \
   "$AVATAR_ROOT/MuseTalk/models/sd-vae/diffusion_pytorch_model.bin" \
   "$AVATAR_ROOT/MuseTalk/models/whisper/pytorch_model.bin" \
   "$AVATAR_ROOT/MuseTalk/models/musetalkV15_shubh/unet.pth"; do
  [[ -s "$f" ]] || { echo "ERROR: missing weight: $f" >&2; exit 1; }
done
GRAPH="$AVATAR_ROOT/avatar_system/data/motion_graph_silent_7f1cea162602.json"
[[ -f "$GRAPH" ]] || { echo 'ERROR: original motion graph is missing' >&2; exit 1; }
LIST="$AVATAR_ROOT/SELECTED_26_CLIPS.txt"
[[ -s "$LIST" ]] || { echo 'ERROR: runtime clip manifest missing' >&2; exit 1; }
# Do not run the graph-builder or feature extractor on the new server.
export CUDA_VISIBLE_DEVICES="${CACHE_GPU:-0}"
export PYTHONPATH="$AVATAR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$AVATAR_ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$AVATAR_ROOT/.cache/torch}"
mkdir -p "$AVATAR_ROOT/avatar_cache" "$HF_HOME" "$TORCH_HOME"

avail_kb="$(df -Pk "$AVATAR_ROOT" | awk 'NR==2 {print $4}')"
if [[ "${avail_kb:-0}" =~ ^[0-9]+$ ]] && (( avail_kb < 24 * 1024 * 1024 )); then
  echo 'WARNING: <24 GiB disk free; full 26-clip caches previously used ~18 GiB.' >&2
fi

limit="${CACHE_LIMIT:-0}"
[[ "$limit" =~ ^[0-9]+$ ]] || { echo 'ERROR: CACHE_LIMIT must be a nonnegative integer' >&2; exit 1; }
processed=0
while IFS= read -r name || [[ -n "$name" ]]; do
  [[ -n "$name" ]] || continue
  [[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "ERROR: unsafe name: $name" >&2; exit 1; }
  [[ -s "$AVATAR_ROOT/base_clips/$name.mp4" ]] || { echo "ERROR: missing clip $name.mp4" >&2; exit 1; }
  cache="$AVATAR_ROOT/avatar_cache/$name"
  if [[ -f "$cache/DONE" ]]; then
    valid=1
    for f in frames.npy masks.pkl latents.pt boxes.npy info.json; do
      [[ -s "$cache/$f" ]] || valid=0
    done
    if (( valid == 1 )); then
      echo "SKIP cache exists: $name"
      continue
    fi
    echo "ERROR: cache $name contains DONE but is incomplete; inspect it manually before retrying." >&2
    exit 1
  fi
  echo "=== Preprocessing $name (GPU $CUDA_VISIBLE_DEVICES) ==="
  (cd "$AVATAR_ROOT/MuseTalk" && "$PY" ../scripts/prep_avatar.py "$name")
  for f in frames.npy masks.pkl latents.pt boxes.npy info.json DONE; do
    [[ -s "$cache/$f" || ( "$f" == DONE && -f "$cache/$f" ) ]] || {
      echo "ERROR: $name missing cache file: $f" >&2; exit 1;
    }
  done
  echo "OK: $name"
  processed=$((processed + 1))
  if (( limit > 0 && processed >= limit )); then
    echo "Stopped after $processed newly generated caches (CACHE_LIMIT=$limit)."
    break
  fi
done < "$LIST"

if (( limit > 0 )); then
  echo 'Partial test finished; rerun without CACHE_LIMIT to build all missing caches.'
else
  echo '=== Validating all 26 caches ==='
  python3 "$KIT_DIR/bin/doctor.py" --root "$AVATAR_ROOT" --require-cache
fi
