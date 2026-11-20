#!/bin/bash
# Phase 5: MuseTalk 1.5 fine-tune on the shubh dataset, DDP on GPUs 1+3 (models loaded one rank at a time, host-RAM rule) (configs/training/gpu_shubh.yaml),
# training-only env envs/musetalk_trainer (torch 2.4.1 cu121; torch 2.0.1 crashes in backward on these H100s).
# W&B project musetalk-shubh on CoreWeave forge (credentials in ~/.netrc).
#
#   musetalk_ft/scripts/train.sh shubh_stage1|shubh_stage2      (log: musetalk_ft/logs/train_<cfg>.log)
#
# Resume: train.py always continues from the newest MuseTalk/exp_out/shubh/<exp>/checkpoint-N (optimizer, LR,
# discriminators, RNG, step, same W&B run), so after a crash / GPU reset just run this script again.
# It also relaunches by itself after a crash (MAX_RESTARTS, default 5, 2 min apart).
# Graceful stop: touch MuseTalk/exp_out/shubh/<exp>/STOP  -> saves full state + unet-N.pth at the next step, exits.
CFG=${1:?config name, e.g. shubh_stage1}
FT=$(cd "$(dirname "$0")/.." && pwd); R=$(dirname "$FT")
cd "$R/MuseTalk"
EXP=$(python3 -c "import yaml;c=yaml.safe_load(open('configs/training/$CFG.yaml'));print(c['output_dir'].rstrip('/')+'/'+c['exp_name'])")
LOG="$FT/logs/train_$CFG.log"
export WANDB_BASE_URL=https://api.forge.coreweave.com
rm -f "$EXP/STOP"  # launching = we want it to run
for attempt in $(seq 0 "${MAX_RESTARTS:-5}"); do
  echo "=== $(date '+%F %T') launch $CFG (attempt $attempt), resume from: $(ls -d "$EXP"/checkpoint-[0-9]* 2>/dev/null | grep -v tmp | sort -t- -k2 -n | tail -1)" | tee -a "$LOG"
  "$R/envs/musetalk_trainer/bin/accelerate" launch --config_file ./configs/training/gpu_shubh.yaml \
    --main_process_port 29517 train.py --config "./configs/training/$CFG.yaml" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] && { echo "=== $(date '+%F %T') $CFG exited cleanly" | tee -a "$LOG"; break; }
  [ -e "$EXP/STOP" ] && break
  if tail -n 300 "$LOG" | grep -q "OutOfMemoryError"; then  # deterministic: a relaunch would just crash again
    echo "=== $(date '+%F %T') $CFG crashed with CUDA OOM; NOT relaunching (reduce batch / frames)" | tee -a "$LOG"; break
  fi
  echo "=== $(date '+%F %T') $CFG crashed (rc $rc); relaunching from the latest checkpoint in 2 min" | tee -a "$LOG"
  sleep 120
done
