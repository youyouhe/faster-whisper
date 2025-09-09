# Docker Deployment for Faster Whisper

This directory contains Docker configuration files for deploying the Faster Whisper service with load balancing across multiple GPU instances.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 1.29+
- NVIDIA Docker runtime (for GPU support)
- NVIDIA drivers compatible with CUDA 12.3

## Quick Start

1. Build and start the services:
   ```bash
   docker-compose up --build
   ```

2. Access the load balancer API at `http://localhost:5001`

## Configuration

The services will be available at:
- Load balancer: http://localhost:5001
- GPU service 0: http://localhost:5002
- GPU service 1: http://localhost:5003
- GPU service 2: http://localhost:5004

## Environment Variables

You can customize the deployment by setting environment variables in a `.env` file:

```
MAX_QUEUE_SIZE=100
REQUEST_TIMEOUT=1800
HEALTH_CHECK_INTERVAL=30
```

## Model Management

Models will be automatically downloaded to the `./models` directory on first use. You can also pre-download models by modifying the Dockerfile.

## Logs

Service logs are available in the `./logs` directory or via `docker-compose logs`.