# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Faster Whisper, a reimplementation of OpenAI's Whisper model using CTranslate2 for faster inference. It provides up to 4x faster transcription with the same accuracy while using less memory. The project includes both the core transcription library and a comprehensive API service with Docker deployment capabilities.

## Key Components

### Core Library
1. **WhisperModel** - Main class for transcription with various options (`faster_whisper/transcribe.py:606-1828`)
2. **BatchedInferencePipeline** - For batched transcription processing (`faster_whisper/transcribe.py:111-604`)
3. **Tokenizer** - Handles text tokenization for different languages
4. **FeatureExtractor** - Processes audio into features for the model
5. **VAD (Voice Activity Detection)** - Filters out non-speech audio segments

### API Service Architecture
The system implements a distributed processing architecture:
- **Load Balancer** (`load_balancer.py`) - Distributes requests across GPU services
- **TUS Server** (`tus_server.py`) - Resumable file uploads using TUS protocol
- **ASR Worker** (`asr_worker.py`) - Coordinates transcription tasks
- **Message Queue** (`message_queue.py`) - Redis-based task management

## Development Commands

### Installation
```bash
# Core library with dev dependencies
pip install -e .[dev]

# API service dependencies
pip install pydub>=0.25.1 numpy>=1.21.0
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

### Docker Deployment
```bash
# Build and start all services
docker-compose up --build

# Start multi-GPU instances
./start_multi_instance.sh

# Start load balancer
./start_load_balancer.sh
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
- `docker/` - Docker configuration files
- `data/` - Data storage for uploads and results

## Distributed Architecture

The service architecture supports high-throughput transcription across multiple GPU instances:

```
Client Apps → Load Balancer (Port 5001) → Backend Services (Ports 5002-5008)
     ↓              ↓                        ↓
Large Files    Queue System        GPU Instances
Concurrent     Round-Robin        File Chunking
Requests       Health Checks      Serial Processing
```

### Key Features
- **TUS Protocol Support**: Resumable uploads for reliable file transfers
- **Multi-GPU Load Balancing**: Automatic distribution across available GPUs
- **Redis Message Queue**: Asynchronous task management
- **Health Monitoring**: Continuous service health checks
- **Callback Service**: HTTP callbacks for async result delivery

## Key Dependencies

### Core Library
- `ctranslate2>=4.0,<5` - Fast inference engine
- `huggingface_hub>=0.13` - Model downloading
- `tokenizers>=0.13,<1` - Text tokenization
- `onnxruntime>=1.14,<2` - ONNX runtime support
- `av>=11` - Audio/video processing
- `numpy>=1.21.0` - Numerical computing

### API Service
- `fastapi>=0.68.0` - Web framework
- `uvicorn>=0.15.0` - ASGI server
- `aiohttp>=3.8.0` - Async HTTP client/server
- `torch>=1.10.0` - PyTorch for ML operations
- `redis` - Message queue and caching
- `psutil>=7.0.0` - System monitoring

## Configuration

### Environment Variables
- `MAX_QUEUE_SIZE` - Queue management (default: 10)
- `MAX_FILE_SIZE` - File size limit in MB (default: 20)
- `GPU_DEVICE_ID` - GPU device selection
- `API_PORT` - Service port configuration
- `API_KEY` - Authentication for API access

## Performance Optimizations

- **Batched Processing**: Use `BatchedInferencePipeline` for multiple segments
- **VAD Filtering**: Enable VAD to remove silence and reduce processing time
- **Memory Management**: 20MB chunk limits prevent memory overflow
- **Quantization Support**: INT8 quantization for reduced memory usage

## Core Classes

### WhisperModel
Main class for transcription with methods:
- `transcribe()` - Transcribe audio files with various options
- `detect_language()` - Detect language in audio
- `generate_segments()` - Generate transcription segments

### BatchedInferencePipeline
For batched processing:
- `transcribe()` - Batched transcription with significant performance improvements

## Testing

Tests are written using pytest and can be run with:
```bash
pytest -v tests/
```

Test data is available in `tests/data/` with sample audio files for testing transcription functionality.