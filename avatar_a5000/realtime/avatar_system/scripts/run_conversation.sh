#!/usr/bin/env bash
# Conversational avatar server on :8780 (ssh -L 8780:127.0.0.1:8780 vishal@<gpu28>, open http://localhost:8780).
# Mock mode: --mock --tts mock --llm mock (no GPU, no API key; proves nothing about lip-sync quality).
source "$(dirname "$0")/_env.sh"
if [[ " $* " != *" --mock "* ]]; then need_gpu 12; need_weights; fi
if [[ " $* " != *" --tts mock "* || " $* " != *" --llm mock "* ]]; then need_sarvam; fi
exec $MT avatar_system/streaming/server.py "$@"
