# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Faster Whisper, a reimplementation of OpenAI's Whisper model using CTranslate2 for faster inference. It provides up to 4x faster transcription with the same accuracy while using less memory.

## Key Components

1. **WhisperModel** - Main class for transcription with various options
2. **BatchedInferencePipeline** - For batched transcription processing
3. **Tokenizer** - Handles text tokenization for different languages
4. **FeatureExtractor** - Processes audio into features for the model
5. **VAD (Voice Activity Detection)** - Filters out non-speech audio segments

## Development Commands

### Installation
```bash
pip install -e .[dev]
```

### Running Tests
```bash
pytest -v tests/
```

### Code Formatting
```bash
# Check formatting
black --check .
isort --check-only .

# Apply formatting
black .
isort .
```

### Linting
```bash
flake8 .
```

### Building Package
```bash
python3 setup.py sdist bdist_wheel
```

## Project Structure

- `faster_whisper/` - Main package directory
  - `transcribe.py` - Core transcription logic with WhisperModel and BatchedInferencePipeline
  - `tokenizer.py` - Tokenization utilities
  - `feature_extractor.py` - Audio feature extraction
  - `vad.py` - Voice activity detection
  - `audio.py` - Audio decoding and processing
  - `utils.py` - Utility functions
- `tests/` - Test suite
- `setup.py` - Package setup
- `requirements.txt` - Core dependencies

## Key Dependencies

- `ctranslate2` - Fast inference engine
- `huggingface_hub` - Model downloading
- `tokenizers` - Text tokenization
- `onnxruntime` - ONNX runtime support
- `av` - Audio/video processing
- `numpy` - Numerical computing
- `tqdm` - Progress bars

## Core Classes

### WhisperModel
Main class for transcription with methods:
- `transcribe()` - Transcribe audio files with various options
- `detect_language()` - Detect language in audio
- `generate_segments()` - Generate transcription segments

### BatchedInferencePipeline
For batched processing:
- `transcribe()` - Batched transcription

## Testing

Tests are written using pytest and can be run with:
```bash
pytest -v tests/
```

## Docker Deployment

The project includes Docker configuration for deployment with load balancing across multiple GPU instances:

1. Build and start services:
   ```bash
   docker-compose up --build
   ```

2. Access the load balancer API at `http://localhost:5001`

The Docker setup includes:
- A multi-stage build process
- GPU support via NVIDIA Docker runtime
- Volume mounting for models and logs
- Environment variable configuration