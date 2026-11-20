#!/usr/bin/env bash
# Fetch only MuseTalk inference/preprocessing weights. Never overwrites the trained UNet.
set -Eeuo pipefail
source "$(dirname "$0")/_common.sh"
require_project
require_cmd python3
M="$AVATAR_ROOT/MuseTalk/models"
mkdir -p "$M"
DL_ENV="$KIT_DIR/.download_env"
if [[ ! -x "$DL_ENV/bin/hf" ]]; then
  python3 -m venv "$DL_ENV"
  "$DL_ENV/bin/python" -m pip install --upgrade pip 'huggingface_hub[cli]' gdown
fi
HF="$DL_ENV/bin/hf"

# Required by the live lip-sync adapter; do NOT download stock unet.pth.
"$HF" download TMElyralab/MuseTalk musetalkV15/musetalk.json --local-dir "$M"
"$HF" download stabilityai/sd-vae-ft-mse config.json diffusion_pytorch_model.bin --local-dir "$M/sd-vae"
"$HF" download openai/whisper-tiny config.json pytorch_model.bin preprocessor_config.json --local-dir "$M/whisper"

# Model dependencies used by original face detection/face parsing prep routines.
"$HF" download yzd-v/DWPose dw-ll_ucoco_384.pth --local-dir "$M/dwpose"
mkdir -p "$M/face-parse-bisent"
FP="$M/face-parse-bisent/79999_iter.pth"
if [[ ! -s "$FP" ]]; then
  "$DL_ENV/bin/gdown" --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O "$FP"
fi
RN="$M/face-parse-bisent/resnet18-5c106cde.pth"
if [[ ! -s "$RN" ]]; then
  require_cmd curl
  curl -fL --retry 4 --retry-delay 3 \
    https://download.pytorch.org/models/resnet18-5c106cde.pth -o "$RN"
fi

# The only UNet used at runtime is the person's fine-tuned checkpoint.
FT="$M/musetalkV15_shubh/unet.pth"
if [[ ! -f "$FT" ]]; then
  echo "IMPORTANT: Place fine-tuned UNet here: $FT"
fi
printf '\nPretrained model downloads complete. The original stock 3.2 GB UNet and SyncNet were deliberately skipped.\n'
