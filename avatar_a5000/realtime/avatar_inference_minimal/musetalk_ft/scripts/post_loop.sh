#!/bin/bash
# Phase 3+4 follow-up: every 10 min realign newly scored clips (build_aligned.py) and run MuseTalk preprocessing
# on them (preprocess_shubh.py, GPU 3). Exits after a pass in which every clip of clips.json has a sync json.
# Waits for any running preprocess (tmux mtft_prep) so two never overlap.
FT=$(cd "$(dirname "$0")/.." && pwd); R=$(dirname "$FT")
while true; do
  while tmux has-session -t =mtft_prep 2>/dev/null; do sleep 30; done
  n_clips=$(python3 -c "import json;print(len(json.load(open('$FT/data/clips.json'))))")
  n_sync=$(ls "$FT"/data/sync/mtft_*.json | wc -l)
  echo "== $(date '+%F %T') sync $n_sync/$n_clips"
  (cd "$FT" && ../envs/analysis/bin/python scripts/build_aligned.py)
  (cd "$R/MuseTalk" && CUDA_VISIBLE_DEVICES=3 ../envs/musetalk_train/bin/python -u ../musetalk_ft/scripts/preprocess_shubh.py 2>&1 \
     | grep -vi warn | grep -v "Processing videos" | tail -3)
  [ "$n_sync" -ge "$n_clips" ] && { echo POST_DONE; break; }
  sleep 600
done
