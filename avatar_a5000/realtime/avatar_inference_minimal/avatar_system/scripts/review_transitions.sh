#!/usr/bin/env bash
# Render evidence for flagged/worst/sampled joins, then serve the review UI on :8770
# (ssh -L 8770:127.0.0.1:8770 ...). Decisions persist in data/reviews.json; press "rebuild graph" to apply.
source "$(dirname "$0")/_env.sh"
$AN -m avatar_system.motion.transition_review render --status review --worst ${WORST:-20} --sample ${SAMPLE:-20}
exec $AN -m avatar_system.motion.transition_review serve --port ${PORT:-8770}
