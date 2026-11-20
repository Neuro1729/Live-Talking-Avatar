#!/usr/bin/env bash
# Portable substitute for avatar_system/scripts/_env.sh; installed on the target only.
set -Eeuo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AN="$R/envs/analysis/bin/python"
MT="$R/envs/musetalk/bin/python"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TF_CPP_MIN_LOG_LEVEL=3
export HF_HOME="${HF_HOME:-$R/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$R/.cache/torch}"
unset PYTORCH_CUDA_ALLOC_CONF
mkdir -p "$HF_HOME" "$TORCH_HOME"
# CUDA wheels normally carry their own runtime libs; make torch's libs discoverable.
_SP="$R/envs/musetalk/lib/python3.10/site-packages"
if [[ -d "$_SP/torch/lib" ]]; then
  _LIBS="$_SP/torch/lib"
  while IFS= read -r -d '' dir; do _LIBS="$_LIBS:$dir"; done < <(find "$_SP/nvidia" -maxdepth 3 -type d -name lib -print0 2>/dev/null || true)
  export LD_LIBRARY_PATH="$_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
cd "$R"

need_gpu() {
  local min_gb="${1:-10}"
  GPU="${GPU:-0}"
  command -v nvidia-smi >/dev/null || { echo 'ERROR: NVIDIA driver/nvidia-smi missing' >&2; exit 1; }
  local free_mb
  free_mb="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')" || exit 1
  if (( free_mb < min_gb * 1024 )); then
    echo "ERROR: GPU $GPU has ${free_mb} MiB free; need at least ${min_gb} GiB" >&2
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="$GPU"
  echo "[env] GPU $GPU, ${free_mb} MiB free"
}

need_weights() {
  local missing=0
  local f
  for f in \
      MuseTalk/models/musetalkV15/musetalk.json \
      MuseTalk/models/musetalkV15_shubh/unet.pth \
      MuseTalk/models/sd-vae/config.json \
      MuseTalk/models/sd-vae/diffusion_pytorch_model.bin \
      MuseTalk/models/whisper/config.json \
      MuseTalk/models/whisper/pytorch_model.bin \
      MuseTalk/models/whisper/preprocessor_config.json; do
    if [[ ! -s "$R/$f" ]]; then
      echo "ERROR: missing inference weight/config: $f" >&2
      missing=1
    fi
  done
  (( missing == 0 )) || exit 1
}

need_sarvam() {
  if [[ -z "${SARVAM_API_KEY:-}" ]]; then
    echo "ERROR: SARVAM_API_KEY is missing. Configure $R/../.env" >&2
    exit 1
  fi
}
