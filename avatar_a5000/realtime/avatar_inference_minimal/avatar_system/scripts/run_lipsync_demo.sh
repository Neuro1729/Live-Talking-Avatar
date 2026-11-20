#!/usr/bin/env bash
# MuseTalk lip-sync demo on prerecorded speech (GPU). Use --mock to test the timeline without MuseTalk.
source "$(dirname "$0")/_env.sh"
if [[ " $* " != *" --mock "* ]]; then need_gpu 12; need_weights; fi
[ -f avatar_system/data/test_audio/hi_1.wav ] || { echo "ERROR: test audio missing (avatar_system/data/test_audio)"; exit 1; }
exec $MT avatar_system/scripts/run_lipsync_demo.py "$@"
