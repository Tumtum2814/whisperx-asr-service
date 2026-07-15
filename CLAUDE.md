# CLAUDE.md — whisperx-asr-service fork (Tumtum2814)

## What this project is
Fork of murtaza-nasir/whisperx-asr-service (alpha, CUDA-first) with the goal of
running it **natively on an Apple M4 Pro Mac Mini (64GB)** — first CPU-only,
then Metal-accelerated. Upstream publishes amd64-only Docker images (no arm64
manifest) and Docker-on-Mac has no GPU access at all, so this runs as a native
Python service, not a container.

It serves Speakr (meeting transcription app) as its diarization/voice-profile
ASR backend. API contract must stay compatible with what Speakr's asr_endpoint
connector expects, including speaker embeddings (ASR_RETURN_SPEAKER_EMBEDDINGS).

## Current state (milestone 0: native CPU — WORKING, benchmarked)
- venv-based install (Python 3.11 via Homebrew), launched by `run.sh`,
  uvicorn simple mode on **port 9002**, host 0.0.0.0.
  (9001 is TAKEN by supervisord's web UI — an HTML 401 from "our" port means
  supervisord answered, not this service. FastAPI always answers JSON.)
- End-to-end verified 2026-07-05: /asr with diarize=true on a ~5-min 2-speaker
  clip (Hard Fork podcast) returns speaker-labeled segments.
- **Benchmark (5-min clip, medium, CPU int8): ~4:20 total ≈ 1x realtime.**
  Stage split: transcription ~1:15, diarization ~3:05 — **diarization is ~70%
  of wall clock.** This measurement reordered the roadmap (see below).
- **Milestone 1 DONE 2026-07-06 — MPS for align+diarize (TORCH_DEVICE=mps).**
  Same clip, same medium/int8 CPU transcription: **total ~4:55 → ~1:51**;
  diarization **~3:23 → ~18s (~11x)**; alignment ~6s→~7s (noise). Transcription
  unchanged (~85s, still CPU). Output numerically identical to the CPU run:
  57 segments, 5 speakers, per-speaker embedding cosine-sim = 1.000000, 100%
  segment speaker agreement. (This run's re-measured CPU baseline was
  ~4:55/3:23, a bit above the original ~4:20/3:05 — machine-load variance.)
- DEVICE=cpu, COMPUTE_TYPE=int8, BATCH_SIZE=2 (pipeline.py auto-detects; these
  are the correct CPU values per upstream's env example).
- PRELOAD_MODEL=distil-large-v3.5 (NOTE: this env doubles as the per-request
  DEFAULT model, not just a warmup hint — see pipeline.py DEFAULT_MODEL).
  Chosen by the 2026-07-06 five-model benchmark; run.sh has commented
  one-line switches for medium and large-v3-turbo.
- **Model benchmark 2026-07-06 (5-min clip, CPU int8 batch 2, MPS diarize —
  transcription-stage seconds from server log, model load excluded):**
  distil-large-v3.5 **80.9s** · distil-large-v3 82.5s · medium 88.0s ·
  large-v3-turbo 91.9s · large-v3 132.3s. All five: same 5 speakers, same
  diarization. Text similarity vs medium: v3.5/turbo/large-v3 ≈ 0.986-0.988
  (agree with each other), distil-v3 0.971 (paraphrases). **Winner:
  distil-large-v3.5 — fastest AND large-class fidelity** (3.5 fixed distil's
  paraphrase drift). turbo is a hair better on brand names ("Pure Leaf") but
  12% slower. large-v3 not worth +50%. NOBODY spells "Kokotajlo" — hard names
  need Speakr's hotwords. Advertised distil/turbo speedups (2x/6x) do NOT
  materialize on CTranslate2 int8 M4 — all land within ±10% of medium.
- **PROMOTED TO PROD 2026-07-08:** runs under supervisord (`[program:whisperx]`,
  port 9002) from ~/aistack/whisperx (clone of the GitHub fork, mps-support
  branch; venv rebuilt in place via setup-venv.sh, model caches copied from
  dev). Logs: ~/aistack/logs/whisperx.log. HF_TOKEN + PATH (needs
  /opt/homebrew/bin for ffmpeg) live in the supervisord environment= line.
  Verified end-to-end post-promotion: 5 speakers + 5 embeddings on the test
  clip. Dev copy stays in ~/Documents/claude/whisperxmac/ — push there,
  `git pull` in ~/aistack/whisperx, `supervisorctl restart whisperx`.

## Hard-won environment facts (do not rediscover these)
- **/.cache is the container default and is READ-ONLY on macOS root.** run.sh
  must export CACHE_DIR and HF_HOME to a writable path inside the repo dir,
  and mkdir it. Symptom if missed: "Errno 30 Read-only file system" on model
  preload.
- **The torchcodec warning wall at startup is COSMETIC.** pyannote 4 warns its
  built-in decoder can't load; this service never uses it (WhisperX decodes via
  ffmpeg CLI and passes preloaded waveform dicts). Suppressed via
  PYTHONWARNINGS=ignore::UserWarning (upstream's own compose does the same).
  Root cause if ever needed: pip resolves newest torchcodec, mismatched with
  torch 2.8 (missing _aoti_torch_aten_full symbol). Optional fix:
  torchcodec==0.7.0.
- **WhisperX must come from sealambda's fork**, branch feat/pyannote-audio-4
  (pyannote.audio 4 compatibility). Upstream Dockerfile then sed-patches
  whisperx/diarize.py: `use_token=` → `token=`. The venv setup script does the
  same, locating diarize.py dynamically.
- **ray[serve] is NOT installed and NOT needed.** Only app/serve_app.py and
  app/serve_deployments.py import ray (Ray Serve mode). Simple mode
  (uvicorn, app/main.py) never touches it. Don't add it back. (Grep gotcha:
  "ray" matches np.ndarray.)
- Upstream Dockerfile's torch re-pin to 2.7.1 is Pascal-GPU appeasement —
  irrelevant here. Plain `pip install torch torchaudio` on macOS arm64 gives
  CPU+MPS wheels; torch>=2.8 (sealambda requirement) is fine and preferred.
- HF_TOKEN required at runtime; all THREE pyannote gates must be accepted on
  HF: speaker-diarization-community-1, speaker-diarization-3.1,
  segmentation-3.0. Missing gate = transcription works, speaker labels
  silently absent.
- NLTK punkt_tab is pre-downloaded to <repo>/.cache/nltk_data; NLTK_DATA env
  points there.
- **MPS device split: transcription vs torch-stages are separate knobs.**
  `DEVICE` (=cpu) drives transcription only (CTranslate2, no MPS ever).
  `TORCH_DEVICE` (new, defaults to DEVICE) drives align + diarize, the plain
  PyTorch stages. pipeline.py routes them independently; run.sh defaults
  TORCH_DEVICE=mps on this Mac. Set TORCH_DEVICE=cpu to force the old path.
- **The MPS whack-a-mole never happened.** pyannote community-1 + wav2vec2
  alignment ran fully on MPS under torch 2.8 with ZERO op fallbacks or
  dtype errors — no "not implemented" logs, no .cpu()/.float() patches needed.
  We still export PYTORCH_ENABLE_MPS_FALLBACK=1 as a safety net (other
  languages' align models / other pyannote models may hit an uncovered op),
  but the community-1 diarize path did not need it. Don't go hunting for
  fallback fixes that aren't there.
- **The speaker-embedding MPS-serialization bug is pre-neutralized upstream.**
  sealambda's whisperx/diarize.py already does `embeddings[s].tolist()` before
  returning them (tolist() copies off MPS implicitly), so the embedding payload
  is CPU-side floats by the time it reaches the JSON response. Verified: emb
  cosine-sim CPU-vs-MPS = 1.0, no non-finite values. No manual .cpu() needed.
- clear_gpu_memory() now also calls torch.mps.empty_cache() when
  TORCH_DEVICE=mps (was cuda-only).

## Test commands
```bash
HF_TOKEN=hf_xxx ./run.sh
curl http://localhost:9002/health
curl -F "audio_file=@test.m4a" "http://localhost:9002/asr?diarize=true&output=json"
# timed benchmark run:
time curl -F "audio_file=@clip.mp3" "http://localhost:9002/asr?diarize=true&output=json" -o result.json
./test-api.sh   # upstream's own smoke test
```

## Roadmap (reordered 2026-07-05 by benchmark — diarization dominates)
1. ~~Milestone 0: native CPU baseline~~ **FULLY DONE 2026-07-08.** Speakr wired
   (ASR_BASE_URL=http://host.docker.internal:9002, ASR_DIARIZE=true,
   ASR_RETURN_SPEAKER_EMBEDDINGS=true), state committed + pushed to the
   GitHub fork, and the service promoted to prod under supervisord (see
   Current state).
2. ~~Milestone 1 — MPS for diarization (+alignment)~~ **DONE 2026-07-06.**
   Added TORCH_DEVICE env (defaults to DEVICE) routing align+diarize onto MPS
   while transcription stays on DEVICE=cpu. Diarization ~3:23 → ~18s (~11x),
   well past the "under half of 3:05" success metric; output numerically
   identical to CPU. No op fallbacks or dtype fixes were needed (see env-facts).
   Commits on branch mps-support. Remaining upside is now Milestone 2
   (transcription on Metal via mlx-whisper) — see item 4.
3. ~~Optional quick win: distil-large-v3~~ TESTED 2026-07-05 (rejected: slower
   than medium pre-MPS), **RE-TESTED 2026-07-06 post-MPS in the five-model
   benchmark (see Current state): superseded by distil-large-v3.5, now the
   default.** The 07-05 "distil slower than medium" result flipped to ~6%
   faster on re-run — it was within machine-load noise all along; the real
   finding is that NO model's advertised speedup materializes on CT2/int8/M4.
4. ~~Milestone 2 — transcription on Metal~~ **DONE 2026-07-15 on branch
   mlx-backend (dev only — prod still runs ct2 pending flip).** WHISPER_BACKEND
   env (default ct2) routes transcribe() to mlx-whisper; _MLX_MODEL_MAP maps
   canonical names to MLX HF repos ("/" passes through). Bench (5-min clip,
   MLX large-v3-turbo): transcription 81s → **9.1s**, total pipeline ~96s →
   **~25s** (~13x realtime); text sim vs CT2 turbo 0.991, same 5 speakers +
   256-dim embeddings, clean timestamps. Turbo's advertised speedup that
   never materialized on CT2/int8 DOES on Metal. **distil-large-v3.5 has NO
   usable MLX conversion** (wbell7's drops ~35% of words, broken timestamps —
   rejected in resolve_mlx_repo with a clear error). MLX quirks: no hotwords
   param (folded into initial_prompt), no batching (BATCH_SIZE ignored),
   model cached inside mlx_whisper keyed by repo. To flip prod: set
   WHISPER_BACKEND=mlx in run.sh/supervisord env (PRELOAD_MODEL auto-switches
   to large-v3-turbo).
5. If MPS lands cleanly: PR DEVICE=mps support upstream (repo is alpha,
   15 forks — receptive stage for device patches).
6. **Dockerize the fork (CPU, multi-arch)** for distribution/Unraid/possibly
   the Mac Docker tier: swap base nvidia/cuda → python:3.11-slim, plain PyPI
   torch (no CUDA index, no Pascal re-pin), drop LD_LIBRARY_PATH cuDNN line,
   keep sealambda install + sed patch + NLTK pre-download, build with
   `docker buildx --platform linux/amd64,linux/arm64` (fixes upstream's
   arm64 manifest gap). Note: container = CPU forever (no Metal in Docker);
   native install stays the performance artifact if Milestone 1 pays off.

## Port map (Mac) — check before binding anything
8080 llama router · 8081 sd-server · 8082 whisper.cpp · 4000 LiteLLM ·
5679 Open WebUI · 8000 mcpo · 9000 (old container ASR plan, retired) ·
**9001 supervisord web UI (RESERVED)** · **9002 this service** ·
2376 hawser · 7007 dozzle-agent · 8899 speakr

## Surrounding stack (context, not managed from this repo)
- LiteLLM gateway :4000 (Docker) fronts all models; llama.cpp router :8080,
  stable-diffusion.cpp :8081, whisper.cpp+CoreML :8082 run natively under
  supervisord (config: /opt/homebrew/etc/supervisord.conf, logs:
  ~/aistack/logs/). This service joined them 2026-07-08 at :9002 from
  ~/aistack/whisperx (venv REBUILT there via setup-venv.sh, not mv'd; venvs
  hardcode absolute paths).
- Speakr + the rest of the Docker tier deploy via Dockhand (git-based) from
  the aimac stack; supervisord conf quirks: inline `;` comments need a
  preceding space, command= must be one physical line, PATH must include
  /opt/homebrew/bin.
