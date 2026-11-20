#!/usr/bin/env bash
# Full talking bank (configs/talk_scripts_full.yaml, 60 tb_* clips): Sarvam TTS audio for every clip, then LTX-2.5
# audio-to-video on the running ComfyUI (GPU 1, port 8191) with the steady prompt + faint home-pose mid guides
# (the _stmid settings, lowest head drift in the pilot). Existing clips are skipped, so it can simply be re-run.
#   nohup setsid bash avatar_system/scripts/gen_full_bank.sh > logs/full_bank.log 2>&1 &
source "$(dirname "$0")/_env.sh"
need_sarvam
curl -sf -m5 127.0.0.1:8191/queue >/dev/null || { echo "ERROR: ComfyUI (GPU 1, port 8191) not running - scripts/start_comfy.sh"; exit 1; }
$AN -m avatar_system.generation.talk_bank audio --scripts talk_scripts_full
$AN -m avatar_system.generation.talk_bank run --scripts talk_scripts_full --steady --mid 0.25
echo FULL_BANK_DONE
