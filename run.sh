#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # location-independent: dev copy and ~/aistack/whisperx both work
cd "$DIR"
source venv/bin/activate
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN (pyannote gated models)}"
export CACHE_DIR="$DIR/.cache"          # container default is /.cache — read-only on macOS
export HF_HOME="$DIR/.cache"            # pyannote/HF downloads land here too
export NLTK_DATA="$DIR/.cache/nltk_data"
export PYTHONWARNINGS="ignore::UserWarning"   # silences the cosmetic torchcodec wall
# Transcription backend: ct2 (default, CPU) or mlx (Apple-Silicon Metal).
# MLX pairs with large-v3-turbo (distil-large-v3.5 has no good MLX
# conversion). Milestone 2 bench 2026-07-15: transcription 81s -> 9s,
# total pipeline ~96s -> ~25s on the 5-min clip, text parity 0.991.
export WHISPER_BACKEND="${WHISPER_BACKEND:-ct2}"
#export WHISPER_BACKEND="${WHISPER_BACKEND:-mlx}"
if [ "$WHISPER_BACKEND" = "mlx" ]; then
  export PRELOAD_MODEL="${PRELOAD_MODEL:-large-v3-turbo}"
else
  export PRELOAD_MODEL="${PRELOAD_MODEL:-distil-large-v3.5}"   # fastest + large-class quality on CT2 (2026-07-06 bench)
fi
#export PRELOAD_MODEL="${PRELOAD_MODEL:-medium}"
# Metal for the plain-PyTorch stages (align + diarize). Transcription stays on
# DEVICE=cpu (CTranslate2 has no MPS). Diarization ~203s -> ~18s on the M4 Pro.
# Override with TORCH_DEVICE=cpu to force the old all-CPU path.
export TORCH_DEVICE="${TORCH_DEVICE:-mps}"
export PYTORCH_ENABLE_MPS_FALLBACK=1          # CPU-fallback for any pyannote op lacking an MPS kernel
mkdir -p "$CACHE_DIR"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-9002}"
