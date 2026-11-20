#!/bin/bash
# Waits until every clip has a video and the post loop (realign + preprocess) has finished, then stops the two
# ComfyUI servers (GPU 1 :8191, GPU 3 :8192), waits for their GPU memory to be released and starts stage 1.
FT=$(cd "$(dirname "$0")/.." && pwd)
n_clips=$(python3 -c "import json;print(len(json.load(open('$FT/data/clips.json'))))")
until [ "$(ls "$FT"/data/videos/mtft_*.mp4 | wc -l)" -ge "$n_clips" ] && ! tmux has-session -t =mtft_post 2>/dev/null \
      && ! tmux has-session -t =mtft_prep 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') dataset complete; stopping ComfyUI"
for port in 8191 8192; do
  pid=$(pgrep -u "$USER" -f "main.py --listen 127.0.0.1 --port $port ")
  [ -n "$pid" ] && kill "$pid"
done
tmux kill-session -t =mtft_vidA 2>/dev/null; tmux kill-session -t =mtft_vidB 2>/dev/null
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s}')" -lt 2000 ]; do sleep 10; done
free -g | sed -n 2p
echo "$(date '+%F %T') starting stage 1"
exec "$FT/scripts/train.sh" shubh_stage1
