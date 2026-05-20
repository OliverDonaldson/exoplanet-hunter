#!/usr/bin/env bash
# Run TESS then Kepler scoring chunks back-to-back. Designed for unattended
# overnight execution. Wrap with caffeinate to prevent system sleep.
#
# Usage:
#   caffeinate -dis bash scripts/overnight_score.sh
#
# Pre-flight requirements:
#   - SANDISK mounted at /Volumes/SANDISK/exoplanet_kepler  (Kepler cache)
#   - Laptop lid OPEN  (caffeinate -dis does NOT prevent clamshell sleep)
#   - Laptop on AC power

set -uo pipefail

cd "$(dirname "$0")/.."

KEPLER_CACHE=/Volumes/SANDISK/exoplanet_kepler
if [ ! -d "$KEPLER_CACHE" ]; then
  echo "ERROR: SANDISK not mounted at $KEPLER_CACHE"
  echo "       Mount the drive (Finder -> SANDISK), then re-run."
  exit 1
fi
export KEPLER_RAW_DIR="$KEPLER_CACHE"

mkdir -p logs
LOG="logs/overnight_$(date +%Y%m%d_%H%M%S).log"
PY=/opt/anaconda3/envs/exoplanet-hunter/bin/python

echo "logging to $LOG"
echo "KEPLER_RAW_DIR=$KEPLER_RAW_DIR"

{
  echo "=== TESS chunk starting @ $(date) ==="
  "$PY" scripts/score_candidates.py limit_mission=TESS \
    || echo "WARN: TESS exited non-zero (will continue to Kepler)"

  echo "=== Kepler chunk starting @ $(date) ==="
  "$PY" scripts/score_candidates.py limit_mission=Kepler \
    || echo "WARN: Kepler exited non-zero"

  echo "=== Overnight run complete @ $(date) ==="
  "$PY" -c "
import pandas as pd
df = pd.read_parquet('results/candidates_scored.parquet')
print(f'final rows: {len(df):,}, ok: {(df.status==\"ok\").sum():,}, err: {(df.status!=\"ok\").sum():,}')
print(f'missions: {df.mission.value_counts().to_dict()}')
print(f'>0.9: {(df.prob_mean>0.9).sum()}, >0.95: {(df.prob_mean>0.95).sum()}')
"
} 2>&1 | tee -a "$LOG"

osascript -e 'display notification "Bulk scoring complete" with title "Exoplanet Hunter"' \
  2>/dev/null || true
