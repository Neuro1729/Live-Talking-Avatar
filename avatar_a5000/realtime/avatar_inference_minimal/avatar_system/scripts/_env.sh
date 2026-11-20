#!/usr/bin/env bash
# Shared environment + safety checks for avatar_system scripts. Source it.
set -euo pipefail
R=/var/autofs/nfs10g/nest10/usrs/vishal/Aaryan/avatar/realtime
AN=$R/envs/analysis/bin/python      # CPU env: mediapipe, graph, tests, evaluation
MT=$R/envs/musetalk/bin/python      # GPU env: MuseTalk, server
export PATH=/var/autofs/nfs10g/disk013/usrs/vishal/video-gen/bin:$PATH
export HF_HOME=/var/autofs/nfs10g/disk013/usrs/vishal/video-gen/.cache/huggingface
export TORCH_HOME=/var/autofs/nfs10g/disk013/usrs/vishal/video-gen/.cache/torch
export CUDA_DEVICE_ORDER=PCI_BUS_ID TF_CPP_MIN_LOG_LEVEL=3
unset PYTORCH_CUDA_ALLOC_CONF       # MuseTalk env (torch 2.0.1) breaks with it
# onnxruntime-gpu (face restore) uses the CUDA/cuDNN libraries shipped with torch / nvidia wheels
_SP=$R/envs/musetalk/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$(ls -d $R/models/cuda11_libs/nvidia/*/lib 2>/dev/null | tr '\n' ':')$_SP/torch/lib:$(ls -d $_SP/nvidia/*/lib 2>/dev/null | tr '\n' ':')${LD_LIBRARY_PATH:-}
cd $R

need_gpu() {  # need_gpu <min_free_GB>: pick $GPU (default from configs/musetalk.yaml) and verify it has room
  local min=${1:-12}
  GPU=${GPU:-$($AN -c "from avatar_system.common import load_config; print(load_config('musetalk')['gpu'])")}
  command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found (no GPU)"; exit 1; }
  local free; free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null) \
    || { echo "ERROR: GPU $GPU not available"; exit 1; }
  (( free / 1024 >= min )) || { echo "ERROR: GPU $GPU has only $((free/1024)) GB free (need $min). Set GPU=<idx>."; exit 1; }
  local ram; ram=$(free -g | awk '/Mem:/ {print $7}')
  (( ram >= 30 )) || { echo "ERROR: only ${ram} GB host RAM available"; exit 1; }
  export CUDA_VISIBLE_DEVICES=$GPU
  echo "[env] GPU $GPU (${free} MiB free), host RAM available ${ram} GB"
}
need_weights() {
  for f in MuseTalk/models/musetalkV15/unet.pth MuseTalk/models/sd-vae MuseTalk/models/whisper; do
    [ -e "$R/$f" ] || { echo "ERROR: missing MuseTalk weights $R/$f (scripts/dl_musetalk_weights.sh)"; exit 1; }
  done
}
need_sarvam() {
  grep -qi '^sarvam_api_key' "$R/../.env" 2>/dev/null || { echo "ERROR: SARVAM_API_KEY missing in Aaryan/avatar/.env (or use --tts mock --llm mock)"; exit 1; }
}
