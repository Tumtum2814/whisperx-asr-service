#!/bin/bash
DIR="$HOME/Documents/claude/whisperxmac/whisperx-asr-service"
cd "$DIR"
source venv/bin/activate
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN (pyannote gated models)}"
export CACHE_DIR="$DIR/.cache"          # container default is /.cache — read-only on macOS
export HF_HOME="$DIR/.cache"            # pyannote/HF downloads land here too
export NLTK_DATA="$DIR/.cache/nltk_data"
export PYTHONWARNINGS="ignore::UserWarning"   # silences the cosmetic torchcodec wall
export PRELOAD_MODEL="${PRELOAD_MODEL:-medium}"
# Metal for the plain-PyTorch stages (align + diarize). Transcription stays on
# DEVICE=cpu (CTranslate2 has no MPS). Diarization ~203s -> ~18s on the M4 Pro.
# Override with TORCH_DEVICE=cpu to force the old all-CPU path.
export TORCH_DEVICE="${TORCH_DEVICE:-mps}"
export PYTORCH_ENABLE_MPS_FALLBACK=1          # CPU-fallback for any pyannote op lacking an MPS kernel
mkdir -p "$CACHE_DIR"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-9002}"
