# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo has **two distinct layers**:

1. **Upstream library** — a fork of SYSTRAN/faster-whisper (CTranslate2-based Whisper inference). Lives in `faster_whisper/` with its pytest suite in `tests/`. Relatively untouched.
2. **Production Chinese ASR service** — a custom distributed transcription system built at the **repo root** (`*.py` files, `docker/`, shell scripts). This is where nearly all active development happens (current branch: VAD-guided audio splitting).

## Development Commands

### Installation
```bash
pip install -e .[dev]          # core library + dev tools
pip install -r requirements.txt  # full service dependencies
# torch is deliberately NOT in requirements.txt — install manually (needed by silero VAD)
```

### Tests
```bash
pytest -v tests/                          # upstream library suite (no server/GPU needed)
pytest tests/test_transcribe.py -k jfk    # single test by name
```

The root-level `test_*.py` files are **ad-hoc scripts, not pytest tests** — run them with `python3`:
- `test_vad_splitting.py`, `test_audio_splitting_optimization.py`, `test_srt_merging_only.py` — pure logic, runnable locally.
- `test_enhanced_splitting.py` — needs real audio; **contains a stale hardcoded path `/home/cat/faster-whisper/66.wav`** (fix to repo root before running).
- `test_load_balancer.py`, `test_stats.py`, `test_async_api.py` — **require a live server** (load balancer on :5001, async API on :5020).
- `tus_client.py` doubles as the end-to-end test client; it writes results to `verification_results/` (server never writes there).

### Formatting / Linting (upstream CI enforces on `master` only)
```bash
black . && isort . && flake8 .
```

### Running the service
```bash
docker-compose up --build              # full stack (see architecture below)
./start_dynamic_services.sh            # host: one faster_whisper_api.py per GPU + load balancer
./start_multi_instance_per_gpu.sh      # host: INSTANCES_PER_GPU instances per GPU
```

## Service Architecture

Request lifecycle: upload → queue → GPU processing → result → callback.

```
Client → tus_server (:1080, resumable uploads) / tus_api_server (:8000)
       → Redis queues (tus:upload_completed → tus:asr_processing → tus:asr_completed/failed)
       → asr_worker (:8081, consumes queue, posts audio to load balancer)
       → load_balancer (:5001, round-robin + health checks)
       → faster_whisper_api backends (:5002+, one per GPU, CUDA_VISIBLE_DEVICES per instance)
       → callback_service (posts result to client URL with retries)
```

Key files at repo root:
- `load_balancer.py` — request distribution; auto-generates backend list from `NUM_GPUS`/`INSTANCES_PER_GPU`/`START_PORT` if `BACKEND_SERVICES` is unset.
- `distributed_processor.py` — splits files > `DISTRIBUTED_THRESHOLD_MB` (10MB) into chunks, fans out to backends, merges resulting SRTs.
- `message_queue.py` — Redis queue names and task state.
- `tus_client.py` — reference client / E2E test driver.

Compose mounts `./:/app` into the dynamic/worker/callback containers — code edits need a container restart to take effect.

## Audio Splitting & SRT Merging (current branch focus)

Goal: split long audio for distributed transcription **without timeline corruption or duplicated content**.

- **VAD-guided split points** (`audio_splitter_enhanced.py`, `hybrid_vad_detector.py`): split at silences > 0.3s near the theoretical chunk boundary (score = `distance / (duration + 0.1)`); falls back to even splitting. Hybrid VAD combines librosa energy VAD (sensitive) with silero-vad (precise); recent commits add progressive/adaptive silence thresholds for continuous speech.
- **Overlap reduced 2.0s → 0.5s**; `OVERLAP_SECONDS` defaults to 0.0 in `distributed_processor.py` (config in `audio_splitter_config.env`, which is **sourced by shell scripts, not auto-loaded by Python**).
- **Timestamp alignment is the hard part**: each chunk tracks both *theoretical* (even-split) and *actual* (VAD-adjusted) start/end — passed around as `(actual_start, actual_end, theoretical_start, theoretical_end)` tuples. `srt_merger.py` maps per-chunk relative timestamps back to global time and strips overlap regions; dedup uses SequenceMatcher similarity.
- Known pain points (see `WORK_SUMMARY_2025-11-09.md`): merge strategy flip-flopped between over-fragmented (229 subtitles) and over-merged (1 subtitle); negative-time bugs. **Test the no-merge baseline first, then add merging incrementally.**
- `preprocess_audio.py` — standalone ffmpeg wrapper: converts input to 16kHz mono WAV.

## Environment Variables (key ones)

- **GPU backends**: `API_PORT`, `GPU_DEVICE_ID`, `INSTANCE_ID`, `MAX_FILE_SIZE` (20MB in `faster_whisper_api.py`, 500MB in `tus_server.py` — intentional, TUS handles large uploads), `CHUNK_TIMEOUT`/`REQUEST_TIMEOUT` (3600).
- **Load balancer**: `BACKEND_SERVICES` (CSV), `LB_PORT`, `NUM_GPUS`, `INSTANCES_PER_GPU`, `START_PORT`, `REQUEST_TIMEOUT` (1800), `LARGE_FILE_THRESHOLD` (50MB), `DATABASE_PATH`, `UPLOAD_DIR`.
- **Distributed processing**: `DISTRIBUTED_THRESHOLD_MB`, `OVERLAP_SECONDS`, `MIN_CHUNK_SIZE_MB`, `USE_ENHANCED_SPLITTER`, `MAX_CONCURRENT_DISTRIBUTED`.
- **Shared**: `API_KEY` (unset = open access), `REDIS_URL` (default `redis://redis:6379` — the **docker hostname**; override for bare-host runs), `SRT_STORAGE_DIR`, `LOAD_BALANCER_URL`.

## Operational Gotchas

- **GPU/CUDA required** for the service: backends load `WhisperModel("large-v3-turbo", compute_type="float32")` (float32 is deliberate for accuracy); model load takes **2–3 minutes per instance**. Warmup uses a `tiny` int8 model.
- Start scripts set `LD_LIBRARY_PATH` from pip-installed `nvidia.cublas.lib`/`nvidia.cudnn.lib` packages.
- Start scripts assume CWD `/app` (container layout).
- `Dockerfile.worker` healthcheck ends with `|| 1` (never fails — likely a bug).
- SQLite DBs (`tus_uploads.db`, `data/tasks.db`) live in-repo; `data/` holds uploads and SRT results.

## Core Library (upstream fork)

- `faster_whisper/transcribe.py` — `WhisperModel` (main transcription class) and `BatchedInferencePipeline` (batched processing, significant speedup).
- `faster_whisper/vad.py` — VAD filtering; `get_speech_timestamps` is reused by the root-level hybrid VAD.
- `faster_whisper/audio.py` — audio decoding (av-based); `feature_extractor.py` — mel features; `tokenizer.py` — text tokenization.
- Core deps: `ctranslate2>=4.0,<5`, `huggingface_hub>=0.13`, `tokenizers>=0.13,<1`, `onnxruntime>=1.14,<2`, `av>=11`. Python >= 3.9.
