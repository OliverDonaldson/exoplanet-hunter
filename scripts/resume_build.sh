#!/usr/bin/env bash
# Resume the Kepler+TESS bulk download after a network transition.
#
# Usage (from repo root, with conda env activated):
#     ./scripts/resume_build.sh
#
# What it does, in order:
#   1. Refuses to run if a build is already running (avoids accidental double-spawn).
#   2. Clears any "download error" entries from data/raw/manifest.json so they
#      get retried on the new network. Keeps "no pipeline data" / "search error"
#      entries (those won't change with a new network).
#   3. Re-exports KEPLER_RAW_DIR and resumes the build with caffeinate + tee
#      (appending to build.log, not overwriting).

set -euo pipefail

# --- Pre-flight --------------------------------------------------------------

if pgrep -f "scripts/build_dataset.py" > /dev/null; then
  echo "ERROR: a build is already running (pgrep found scripts/build_dataset.py)."
  echo "       Ctrl-C the existing one before resuming."
  exit 1
fi

if [[ ! -d /Volumes/SANDISK/exoplanet_kepler ]]; then
  echo "ERROR: /Volumes/SANDISK/exoplanet_kepler not mounted."
  echo "       Plug in the SANDISK before resuming."
  exit 1
fi

# --- Manifest cleanup --------------------------------------------------------

python - <<'PY'
import json
from pathlib import Path

manifest_path = Path("data/raw/manifest.json")
m = json.loads(manifest_path.read_text())

before_total = len(m)
before_success = sum(1 for v in m.values() if v.get("success"))

# Only clear "download error" — keep permanent failures (no pipeline data,
# search error) so we don't keep retrying targets that genuinely have no SPOC
# data on the archive.
to_clear = [
    k for k, v in m.items()
    if not v.get("success")
    and "download error" in str(v.get("reason", "")).lower()
]
for k in to_clear:
    m.pop(k)

manifest_path.write_text(json.dumps(m, indent=2))

print(f"manifest before: {before_total} entries ({before_success} successful)")
print(f"cleared {len(to_clear)} 'download error' entries (will be retried)")
print(f"manifest after:  {len(m)} entries")
PY

# --- Resume ------------------------------------------------------------------

export KEPLER_RAW_DIR=/Volumes/SANDISK/exoplanet_kepler
echo
echo "Resuming build on $(networksetup -getairportnetwork en0 2>/dev/null | sed 's/.*: //' || echo 'unknown network')"
echo "Logs appending to build.log — Ctrl-C cleanly when you need to move."
echo

caffeinate -is python scripts/build_dataset.py data=large 2>&1 | tee -a build.log
