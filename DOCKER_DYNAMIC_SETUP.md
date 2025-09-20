# Dynamic GPU Service Scaling for Faster Whisper

This setup allows the Faster Whisper service to automatically detect the number of available GPUs and start the appropriate number of ASR services with a load balancer.

## How it works

1. The system automatically detects the number of available GPUs using `nvidia-smi`
2. Based on the GPU count, it starts multiple ASR service instances on different ports
3. A load balancer is started to distribute requests across all GPU services
4. The load balancer runs on port 5001, with GPU services on ports 5002+

## Files

- `docker/Dockerfile.dynamic` - New Dockerfile with GPU detection capabilities
- `docker/start_dynamic_services.sh` - Script that detects GPUs and starts services
- `docker-compose.dynamic.yml` - Docker Compose file using the dynamic approach

## Usage

### Using Docker Compose

```bash
# Start the dynamic service setup
docker-compose -f docker-compose.dynamic.yml up --build
```

### Using Regular Docker

```bash
# Build the image
docker build -f docker/Dockerfile.dynamic -t faster-whisper-dynamic .

# Run the container (automatically detects GPUs)
docker run --gpus all -p 5001:5001 -p 5002:5002 -p 5003:5003 -p 5004:5004 faster-whisper-dynamic
```

## GPU Detection Logic

The system detects GPUs in the following order:

1. Using `nvidia-smi` command (most reliable)
2. From `CUDA_VISIBLE_DEVICES` environment variable
3. From `NVIDIA_VISIBLE_DEVICES` environment variable
4. Defaults to 1 GPU if none detected

## Scaling

The system can handle up to 7 GPUs (ports 5002-5008), with the load balancer on port 5001.

## Health Checks

- Load balancer health: http://localhost:5001/health
- Individual GPU services: http://localhost:5002/health, http://localhost:5003/health, etc.

## Environment Variables

- `NVIDIA_VISIBLE_DEVICES` - Controls which GPUs are visible to the container (set to "all" for all GPUs)
- `CUDA_VISIBLE_DEVICES` - Alternative way to specify visible GPUs