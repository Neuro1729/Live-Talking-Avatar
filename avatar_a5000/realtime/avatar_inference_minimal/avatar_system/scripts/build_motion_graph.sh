#!/usr/bin/env bash
# Rebuild graph from cached features (+ manual reviews in data/reviews.json) and validate it. CPU only.
source "$(dirname "$0")/_env.sh"
$AN -m avatar_system.motion.feature_extractor --workers ${WORKERS:-12}
$AN -m avatar_system.motion.graph_builder
$AN -m avatar_system.motion.graph_validator
