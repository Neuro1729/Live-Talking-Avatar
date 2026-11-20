#!/bin/bash
# Restart the fine-tuned live server (tmux shubh_ft_server, :8782, GPU 3) with extra server args, wait until ready.
#   musetalk_ft/scripts/restart_live.sh --tts sarvam-stream
R=$(cd "$(dirname "$0")/../.." && pwd); cd "$R"
P=$(ps -u "$USER" -o pid=,args= | awk '/[s]erver.py --port 8782/ {print $1}' | head -1)
[ -n "$P" ] && kill "$P" && sleep 5
tmux kill-session -t =shubh_ft_server 2>/dev/null
L=musetalk_ft/logs/live_server_8782.log; : > $L
tmux new-session -d -s shubh_ft_server "cd $R && AVATAR_CFG_MUSETALK=musetalk_shubh GPU=3 avatar_system/scripts/run_conversation.sh \
  --port 8782 --graph motion_graph_silent_7f1cea162602.json $* 2>&1 | tee -a $L"
timeout 300 bash -c "until curl -s -m 2 http://127.0.0.1:8782/api/state >/dev/null 2>&1 || grep -q Traceback $L; do sleep 2; done"
grep -E "MuseTalk loaded|Traceback" $L; free -g | sed -n 2p
