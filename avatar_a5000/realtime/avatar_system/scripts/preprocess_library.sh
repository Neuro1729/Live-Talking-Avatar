#!/usr/bin/env bash
# 1) features for every library clip (CPU, cached by checksum)  2) graph + validation
# 3) MuseTalk caches for runtime clips that lack one (GPU, ~65 s/clip)
source "$(dirname "$0")/_env.sh"
[ -f "$R/comfy/input/home.png" ] || { echo "ERROR: avatar image comfy/input/home.png missing"; exit 1; }
ls $R/base_clips/*.mp4 >/dev/null 2>&1 || { echo "ERROR: no clips in base_clips/"; exit 1; }
$AN -m avatar_system.motion.feature_extractor --workers ${WORKERS:-12}
$AN -m avatar_system.motion.graph_builder
$AN -m avatar_system.motion.graph_validator || { echo "graph validation FAILED"; exit 1; }
missing=$($AN -c "
import json;from pathlib import Path
g=json.load(open('avatar_system/data/motion_graph.json'))
print(' '.join(c for c in sorted({s['clip'] for s in g['segments']}) if not Path('avatar_cache',c,'DONE').exists()))")
if [ -n "$missing" ]; then
  need_gpu 12; need_weights
  (cd MuseTalk && $MT ../scripts/prep_avatar.py $missing)
fi
echo "library ready"
