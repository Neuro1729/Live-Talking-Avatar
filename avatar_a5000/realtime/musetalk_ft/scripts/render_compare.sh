#!/bin/bash
# Render held-out test clips with given stage-2 UNet checkpoints (plain MuseTalk inference, one model at a time on
# one GPU) and upload real | stock | step A | step B ... side-by-sides (face crop, driving audio) to Drive.
#   GPU=0 scripts/render_compare.sh "7000 10000" mtft_test_000 mtft_test_003
# Renders are cached in eval/s2_<step>_one/render/v15/, stock ones in eval/stock/render/v15/.
set -e
STEPS=$1; shift; CLIPS="$*"
FT=$(cd "$(dirname "$0")/.." && pwd); R=$(dirname "$FT"); MT=$R/MuseTalk; CL=dataset/shubh/clips
FF=$R/envs/musetalk/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
DRIVE="gdrive:Realtime_Avatar_2026-10-30/musetalk_ft_eval/stage2_step5000/sbs"
export PATH=$FT/eval/bin:$PATH
cd "$MT"
for st in $STEPS; do
  E=$FT/eval/s2_${st}_one; mkdir -p "$E"; : > "$E/tasks.yaml"; i=0; need=0
  for c in $CLIPS; do
    [ -e "$E/render/v15/${c}_${c}.mp4" ] && continue
    printf "task_$i:\n video_path: \"$CL/$c.mp4\"\n audio_path: \"$CL/$c.wav\"\n" >> "$E/tasks.yaml"; i=$((i+1)); need=1
  done
  [ $need = 1 ] && CUDA_VISIBLE_DEVICES=${GPU:-0} "$R/envs/musetalk/bin/python" -m scripts.inference \
      --inference_config "$E/tasks.yaml" --result_dir "$E/render" --unet_model_path exp_out/shubh/shubh_s2/unet-$st.pth \
      --unet_config ./models/musetalkV15/musetalk.json --version v15 --gpu_id 0 >> "$E/render.log" 2>&1
  echo "step $st rendered"
done
K="crop=400:460:318:50"
for c in $CLIPS; do
  ins=(-i "$CL/$c.mp4" -i "$FT/eval/stock/render/v15/${c}_${c}.mp4"); name="${c}__real_stock"
  for st in $STEPS; do ins+=(-i "$FT/eval/s2_${st}_one/render/v15/${c}_${c}.mp4"); name+="_s$st"; done
  n=$(( ${#ins[@]} / 2 )); f=""; s=""
  for j in $(seq 0 $((n-1))); do f+="[$j:v]$K[v$j];"; s+="[v$j]"; done
  O=$FT/eval/preview/$name.mp4
  "$FF" -y -loglevel error "${ins[@]}" -i "$CL/$c.wav" -filter_complex "${f}${s}hstack=inputs=$n[v]" -map "[v]" -map $n:a \
    -c:v libx264 -crf 16 -pix_fmt yuv420p -c:a aac -shortest "$O"
  rclone copyto "$O" "$DRIVE/$(basename "$O")" 2>&1 | grep -v NOTICE || true
  echo "uploaded $(basename "$O")"
done
