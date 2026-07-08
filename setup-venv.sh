#!/bin/bash
# Build the native macOS venv for whisperx-asr-service (Apple Silicon).
# Reproduces the verified environment from requirements-mac.lock, including
# the sealambda whisperX fork (pyannote.audio 4 compat) pinned by commit.
#
# venvs hardcode absolute paths — always REBUILD in place (run this script),
# never mv/cp a venv from another directory.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PYTHON="${PYTHON:-/opt/homebrew/bin/python3.11}"
"$PYTHON" -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements-mac.lock

# Upstream's Dockerfile sed-patches whisperx/diarize.py for pyannote.audio 4
# (use_token= -> token=). Locate diarize.py dynamically and do the same;
# idempotent if the pinned commit is already patched.
DIARIZE=$(venv/bin/python -c "import whisperx.diarize, inspect; print(inspect.getsourcefile(whisperx.diarize))")
sed -i '' 's/use_token=/token=/g' "$DIARIZE"

# Pre-download NLTK punkt_tab into the repo-local cache (run.sh points
# NLTK_DATA here; /.cache is read-only on macOS root).
mkdir -p .cache/nltk_data
venv/bin/python -c "import nltk; nltk.download('punkt_tab', download_dir='.cache/nltk_data')"

echo "venv ready. Launch with: HF_TOKEN=hf_xxx ./run.sh"
