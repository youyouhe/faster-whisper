# API Key Authentication Setup

This document explains how to configure API key authentication for the ASR services.

## Overview

The ASR system now supports API key authentication to protect your services from unauthorized access. The authentication covers:

- **TUS API Server** (port 8000): Task creation, status queries, and SRT downloads
- **TUS Upload Server** (port 1080): File upload operations
- **Health endpoints**: Remain publicly accessible for monitoring

## Configuration

### 1. Set API Key Environment Variable

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file with your secure API key
nano .env
```

Generate a secure API key:

```bash
# Using OpenSSL
openssl rand -hex 32

# Using Python
python3 -c "import secrets; print(secrets.token_hex(32))"
```

Example `.env` file:
```
API_KEY=a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456
```

### 2. Update Docker Compose

The `docker-compose.yml` file is already configured to use the API key from the environment variable:

```yaml
environment:
  - API_KEY=${API_KEY:-your-secret-api-key-here}
```

### 3. Restart Services

```bash
# Stop existing services
docker-compose down

# Start with new configuration
docker-compose up --build
```

## Usage

### With API Key (Recommended)

Add the API key to your request headers:

```bash
# Example: Create ASR task with API key
curl -X POST "http://localhost:8000/api/v1/asr-tasks" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key-here" \
  -d '{
    "filename": "audio.wav",
    "filesize": 1024000,
    "metadata": {"language": "en", "model": "large-v3-turbo"}
  }'

# Example: Query task status
curl -X GET "http://localhost:8000/api/v1/asr-tasks/task-id" \
  -H "X-API-Key: your-secret-api-key-here"

# Example: Upload file using TUS protocol
curl -X POST "http://localhost:1080/files" \
  -H "Upload-Length: 1024000" \
  -H "Upload-Metadata: filename audio.wav" \
  -H "X-API-Key: your-secret-api-key-here"
```

### Without API Key (Not Recommended for Production)

If no API key is configured, the services will be open to all requests (backward compatibility).

## Testing

Run the test script to verify your API key configuration:

```bash
python3 test_api_key.py
```

This script will test:
- Health endpoints (should work without API key)
- Protected endpoints without API key (should fail if API key is configured)
- Protected endpoints with valid API key (should succeed)
- Protected endpoints with invalid API key (should fail)

## Security Considerations

1. **Use Strong API Keys**: Generate cryptographically secure random keys
2. **Keep Keys Secret**: Never commit API keys to version control
3. **Rotate Keys Regularly**: Change API keys periodically
4. **Use HTTPS**: In production, always use HTTPS to protect API keys in transit
5. **Monitor Access**: Set up logging to monitor API access and detect unauthorized attempts

## Troubleshooting

### Common Issues

1. **401 Unauthorized Error**
   - Check that the API key is correctly set in environment variables
   - Verify the `X-API-Key` header is included in requests
   - Ensure the API key matches exactly (no extra spaces)

2. **Services Not Starting**
   - Verify the `.env` file exists and is readable
   - Check Docker logs for authentication-related errors
   - Ensure environment variables are properly passed to containers

3. **Missing Environment Variable**
   - If API_KEY is not set, services will run without authentication
   - Check logs for warnings about missing API key

### Docker Logs

Check service logs for authentication issues:

```bash
# API Server logs
docker logs tus-api-server

# TUS Server logs
docker logs tus-server

# All services
docker-compose logs
```

## Implementation Details

The API key authentication is implemented using:

- **FastAPI middleware** for the API server
- **Header-based authentication** using the `X-API-Key` header
- **Environment variable configuration** via the `API_KEY` environment variable
- **Graceful fallback** - if no API key is configured, services remain open

All endpoints except health checks require authentication when an API key is configured.

## Migration Notes

- Existing integrations will continue to work without API keys (backward compatibility)
- To enable security, set the `API_KEY` environment variable and restart services
- Update client applications to include the `X-API-Key` header in requests