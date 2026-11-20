#!/usr/bin/env bash
# Follow the talking-bank generation: every new base_clips/<PATTERN>.mp4 gets indexed (features + talk_index, CPU)
# and a MuseTalk cache (GPU, batched per round so models load once per round). Stops after DONE_COUNT clips are ready.
#   GPU=3 PATTERN='tb_*' DONE_COUNT=60 bash avatar_system/scripts/bank_followup.sh
source "$(dirname "$0")/_env.sh"
PATTERN=${PATTERN:-tb_*}; DONE_COUNT=${DONE_COUNT:-60}
export CUDA_VISIBLE_DEVICES=${GPU:-3}
while true; do
  new=()
  for f in $R/base_clips/$PATTERN.mp4; do
    [ -e "$f" ] || continue
    n=$(basename "$f" .mp4)
    [ -f "$R/avatar_cache/$n/DONE" ] && continue
    [ -f "$R/avatar_system/data/talk_bank/${n}_parts.json" ] || continue
    new+=("$n")
  done
  ready=$(ls -d $R/avatar_cache/$PATTERN 2>/dev/null | while read d; do [ -f "$d/DONE" ] && echo x; done | wc -l)
  echo "[$(date +%H:%M:%S)] ready $ready/$DONE_COUNT, new: ${new[*]:-none}"
  (( ready >= DONE_COUNT )) && { echo BANK_READY; exit 0; }
  if (( ${#new[@]} )); then
    $AN -m avatar_system.generation.talk_index "${new[@]}" || true
    (cd $R/MuseTalk && $MT ../scripts/prep_avatar.py "${new[@]}") 2>&1 | grep -E "frames, |Error" || true
  fi
  sleep 60
done
