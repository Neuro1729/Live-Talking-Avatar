#!/usr/bin/env bash
# Tests + evaluation report (docs/benchmark_report.md). CPU only (uses an already rendered lip-sync demo if present).
source "$(dirname "$0")/_env.sh"
$AN -m pytest -q avatar_system/tests
$AN -m avatar_system.evaluation.benchmark --seeds ${SEEDS:-3} --minutes ${MINUTES:-10}
