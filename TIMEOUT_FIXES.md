# Timeout Handling Improvements for Large Files

This document describes the improvements made to handle large audio file processing timeouts in the Faster Whisper load balancer and backend services.

## Problem

The original implementation had several timeout issues when processing large audio files:

1. **Fixed timeout values**: All requests used the same 30-minute timeout regardless of file size
2. **Chunk processing timeouts**: Individual chunks could timeout during processing (54+ seconds per chunk)
3. **Connection timeouts**: HTTP connections to backends had fixed short timeouts
4. **Health check interference**: Busy backends were incorrectly marked as unhealthy

## Solution

### 1. Progressive Timeout System

Implemented file size-based timeout calculation in `load_balancer.py`:

```python
def calculate_request_timeout(file_size: Optional[int]) -> int:
    """Calculate progressive timeout based on file size"""
    if not file_size:
        return REQUEST_TIMEOUT
    
    file_size_mb = file_size / (1024 * 1024)
    
    # Progressive timeout:
    # - Small files (< 10MB): 30 minutes
    # - Medium files (10-50MB): 45 minutes  
    # - Large files (> 50MB): 60 minutes
    if file_size_mb < 10:
        return 1800  # 30 minutes
    elif file_size_mb < 50:
        return 2700  # 45 minutes
    else:
        return REQUEST_TIMEOUT  # 60 minutes
```

### 2. Improved Backend Timeouts

Updated `faster_whisper_api.py` with separate timeouts:

- **Request timeout**: Increased to 60 minutes for overall processing
- **Chunk timeout**: 20 minutes per individual chunk
- **Environment variable control**: Configurable via `CHUNK_TIMEOUT`

### 3. Enhanced Connection Management

Improved `load_balancer.py` connection handling:

```python
timeout = aiohttp.ClientTimeout(
    total=total_timeout,      # Progressive total timeout
    connect=connect_timeout,  # Connection establishment
    sock_connect=connect_timeout,  # Socket connection
    sock_read=total_timeout   # Socket read timeout
)
```

### 4. Better Health Check System

Enhanced health check logic:

- **Busy backend handling**: Skip health checks for busy backends
- **Longer health check timeout**: Configurable via `HEALTH_CHECK_TIMEOUT`
- **Graceful degradation**: Don't immediately mark backends as unhealthy on timeout

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REQUEST_TIMEOUT` | `3600` | Overall request timeout in seconds (60 minutes) |
| `CHUNK_TIMEOUT` | `1200` | Individual chunk processing timeout (20 minutes) |
| `CHUNK_PROCESSING_TIMEOUT` | `600` | Load balancer chunk processing timeout (10 minutes) |
| `LARGE_FILE_THRESHOLD` | `50` | File size threshold for progressive timeout (50MB) |
| `HEALTH_CHECK_TIMEOUT` | `15` | Health check timeout in seconds |
| `HEALTH_CHECK_INTERVAL` | `30` | Health check interval in seconds |
| `MAX_QUEUE_SIZE` | `100` | Maximum requests in queue |
| `MAX_FILE_SIZE` | `20` | Maximum chunk size in MB |

### Timeout Progression

- **< 10MB**: 30 minutes total timeout
- **10-50MB**: 45 minutes total timeout
- **> 50MB**: 60 minutes total timeout

## Files Modified

### load_balancer.py
- Added progressive timeout calculation
- Enhanced connection timeout management
- Improved health check logic
- Added file size tracking for requests

### faster_whisper_api.py
- Increased request timeout to 60 minutes
- Added separate chunk timeout (20 minutes)
- Updated chunk processing to use individual timeouts

### New Files

### start_services.sh
- Automated startup script with proper environment variables
- Starts load balancer and multiple backend services
- Includes health checks and logging

### test_timeout_fixes.py
- Test script to verify timeout handling
- Creates test audio files of different sizes
- Validates progressive timeout behavior

## Usage

### Starting Services

```bash
# Start all services with improved timeout handling
./start_services.sh

# Or manually with environment variables
export REQUEST_TIMEOUT=3600
export CHUNK_TIMEOUT=1200
export LARGE_FILE_THRESHOLD=50
python load_balancer.py &
python faster_whisper_api.py &
```

### Testing

```bash
# Run timeout handling tests
python test_timeout_fixes.py
```

### Monitoring

Check service health:

```bash
# Load balancer health
curl http://localhost:5001/health

# Individual backend health
curl http://localhost:5002/health
```

## Benefits

1. **Reduced timeouts**: Large files now get appropriate timeout values
2. **Better resource utilization**: Progressive timeouts based on actual processing needs
3. **Improved reliability**: Enhanced connection and health check handling
4. **Configurable**: All timeouts can be adjusted via environment variables
5. **Monitoring**: Better logging and status tracking

## Error Handling

The improved system handles:

- **Progressive timeouts**: Different timeouts based on file size
- **Connection errors**: Better retry logic and timeout handling
- **Backend failures**: Graceful fallback and queue management
- **Health check interference**: Proper handling of busy backends

## Performance Considerations

- **Memory usage**: Larger timeouts may increase memory usage for queued requests
- **Queue management**: Monitor queue length and adjust `MAX_QUEUE_SIZE` as needed
- **Backend scaling**: Add more backend services for better throughput
- **File splitting**: Adjust `MAX_FILE_SIZE` based on your GPU capabilities

## Troubleshooting

### Common Issues

1. **Still getting timeouts**:
   - Increase timeout values via environment variables
   - Check backend service logs for errors
   - Verify GPU memory availability

2. **Queue full errors**:
   - Increase `MAX_QUEUE_SIZE`
   - Add more backend services
   - Reduce file processing time

3. **Backend not responding**:
   - Check backend service logs
   - Verify health check timeouts
   - Ensure sufficient GPU memory

### Log Files

- `lb.log`: Load balancer logs
- `backend1.log`, `backend2.log`, etc.: Individual backend logs
- Check for timeout and error messages

## Future Improvements

1. **Dynamic timeout calculation**: Based on audio duration and model complexity
2. **Adaptive chunking**: Dynamic chunk size based on processing performance
3. **Load-based scaling**: Automatically add backends based on queue length
4. **Circuit breakers**: Prevent cascading failures across backends
5. **Metrics collection**: Detailed performance metrics and monitoring