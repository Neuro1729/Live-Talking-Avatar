#!/usr/bin/env bash
# Continuous silent motion-graph playback over a simulated conversation (CPU). Args pass through (e.g. --minutes 10).
source "$(dirname "$0")/_env.sh"
exec $AN avatar_system/scripts/run_silent_demo.py "$@"
