#!/bin/bash
# Local LLM for the live avatar: Sarvam-30B FP8 (MoE, 2.4B active) on vLLM 0.32 (native SarvamMoE support since
# vllm PR #33942), OpenAI-compatible on 127.0.0.1:${PORT:-8300}, one GPU (${GPU:-2}); tmux session llm_server.
# Flags follow the sarvam-30b-fp8 model card (modelopt FP8, fp8 KV cache); --moe-backend triton because FlashInfer
# JIT-compiles its fused MoE with the old default nvcc and fails (the card env VLLM_USE_FLASHINFER_MOE_FP8 is gone; the
# card's "server hangs in compilation" warning). Thinking is disabled per request (chat_template_kwargs.enable_thinking=false).
#   musetalk_ft/scripts/serve_llm.sh            -> server --llm local --llm-url http://127.0.0.1:8300
R=$(cd "$(dirname "$0")/../.." && pwd); cd "$R"
L=musetalk_ft/logs/llm_server.log; : > $L
tmux kill-session -t =llm_server 2>/dev/null
tmux new-session -d -s llm_server "cd $R && PATH=$R/envs/vllm/bin:\$PATH CUDA_VISIBLE_DEVICES=${GPU:-2} CUDA_HOME=/usr/local/cuda-12.9 \
  HF_HOME=/var/autofs/nfs10g/disk013/usrs/vishal/video-gen/.cache/huggingface \
  envs/vllm/bin/vllm serve models/llm/sarvam-30b-fp8 --served-model-name sarvam-30b --trust-remote-code \
  --quantization modelopt --kv-cache-dtype fp8 --moe-backend triton --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --host 127.0.0.1 --port ${PORT:-8300} 2>&1 | tee -a $L"
timeout 1200 bash -c "until curl -s -m 2 http://127.0.0.1:${PORT:-8300}/v1/models >/dev/null 2>&1 || grep -qE 'Traceback|Error' $L; do sleep 5; done"
curl -s http://127.0.0.1:${PORT:-8300}/v1/models | head -c 300; echo; grep -E "Traceback|Error" $L | tail -5; free -g | sed -n 2p
