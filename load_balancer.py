#!/usr/bin/env python3
"""
Load balancer for faster-whisper API services
Distributes requests across multiple GPU instances
"""

import os
import asyncio
import aiohttp
from aiohttp import web, ClientSession
import json
from typing import List, Dict, Optional, Tuple, Any
import logging
from collections import deque
import uuid
from dataclasses import dataclass
from concurrent.futures import TimeoutError
import subprocess
import tempfile
import re

# Import distributed processing components
from distributed_processor import DistributedProcessor



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def parse_multipart_data(data: bytes, boundary: str) -> bytes:
    """Parse multipart data and extract audio file content"""
    try:
        # Convert boundary to bytes with proper formatting
        boundary_bytes = f"--{boundary}".encode('utf-8')
        end_boundary_bytes = f"--{boundary}--".encode('utf-8')

        # Find the start of the audio data (after headers)
        # Look for the boundary, then skip headers until we find the empty line
        start_idx = 0
        while start_idx < len(data):
            # Find next boundary
            boundary_idx = data.find(boundary_bytes, start_idx)
            if boundary_idx == -1:
                logger.error("Could not find starting boundary in multipart data")
                raise ValueError("Could not find starting boundary in multipart data")

            # Find the end of headers (double newline)
            header_end_idx = data.find(b'\r\n\r\n', boundary_idx)
            if header_end_idx == -1:
                # Try with just \n\n
                header_end_idx = data.find(b'\n\n', boundary_idx)
                if header_end_idx == -1:
                    start_idx = boundary_idx + len(boundary_bytes)
                    continue

            # Extract content between headers and next boundary
            content_start = header_end_idx + 4 if data[header_end_idx:header_end_idx+4] == b'\r\n\r\n' else header_end_idx + 2
            content_end = data.find(boundary_bytes, content_start)

            # If we found another boundary, this is our content
            if content_end != -1:
                # Check if this section contains the audio file
                # Look for filename in the headers part
                header_part = data[boundary_idx:header_end_idx].decode('utf-8', errors='ignore')
                if 'filename=' in header_part or 'name="audio"' in header_part:
                    # This is the audio content
                    return data[content_start:content_end].strip()

            start_idx = header_end_idx + 4

        logger.error("No audio file found in multipart data")
        raise ValueError("No audio file found in multipart data")

    except Exception as e:
        logger.error(f"Error parsing multipart data: {e}")
        raise

# Load backend services from environment variables
BACKEND_SERVICES = os.getenv("BACKEND_SERVICES", "http://localhost:5002,http://localhost:5003,http://localhost:5004,http://localhost:5005").split(",") 
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))  # seconds
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "100"))  # Maximum requests in queue
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "1800"))  # seconds (30 minutes for large audio files)
CHUNK_PROCESSING_TIMEOUT = int(os.getenv("CHUNK_PROCESSING_TIMEOUT", "600"))  # seconds (10 minutes per chunk)
LARGE_FILE_THRESHOLD = int(os.getenv("LARGE_FILE_THRESHOLD", "50"))  # MB
HEALTH_CHECK_TIMEOUT = int(os.getenv("HEALTH_CHECK_TIMEOUT", "15"))  # seconds

@dataclass
class QueuedRequest:
    """Represents a queued request"""
    request_id: str
    request: web.Request
    request_body: bytes
    form_data: Optional[aiohttp.FormData]
    future: asyncio.Future
    timestamp: float
    file_size: Optional[int] = None  # Store file size for progressive timeout

# Global state
BACKEND_STATUS = {service: True for service in BACKEND_SERVICES}  # Assume all healthy initially
BACKEND_BUSY = {service: False for service in BACKEND_SERVICES}  # Track busy backends
REQUEST_QUEUE = deque(maxlen=MAX_QUEUE_SIZE)  # Queue for pending requests
ACTIVE_REQUESTS = {}  # Track active requests by backend
current_index = 0  # For round-robin distribution
queue_processor_task = None  # Background task for processing queue

# Initialize distributed processor
distributed_processor = DistributedProcessor()

def check_audio_format(file_data: bytes) -> Dict[str, Any]:
    """
    Use ffprobe to check audio file format and duration

    Args:
        file_data: Audio file data as bytes

    Returns:
        Dictionary with format information
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_file:
            temp_file.write(file_data)
            temp_file_path = temp_file.name

        try:
            # Run ffprobe to get format information
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                temp_file_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode != 0:
                logger.error(f"ffprobe failed: {result.stderr}")
                return {"error": result.stderr}

            # Parse JSON output
            probe_data = json.loads(result.stdout)

            # Extract relevant information
            format_info = {
                "format": probe_data.get("format", {}).get("format_name", "unknown"),
                "duration": float(probe_data.get("format", {}).get("duration", 0)),
                "size": int(probe_data.get("format", {}).get("size", 0)),
                "bit_rate": int(probe_data.get("format", {}).get("bit_rate", 0)),
                "streams": []
            }

            # Extract stream information
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    stream_info = {
                        "codec": stream.get("codec_name", "unknown"),
                        "sample_rate": stream.get("sample_rate", "unknown"),
                        "channels": stream.get("channels", "unknown"),
                        "duration": float(stream.get("duration", 0))
                    }
                    format_info["streams"].append(stream_info)

            logger.info(f"Audio format check: {format_info}")
            return format_info

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    except subprocess.TimeoutError:
        logger.error("ffprobe timeout")
        return {"error": "ffprobe timeout"}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse ffprobe output: {e}")
        return {"error": f"JSON parse error: {e}"}
    except Exception as e:
        logger.error(f"Error checking audio format: {e}")
        return {"error": str(e)}

async def health_check_task():
    """Periodically check health of backend services"""
    logger.info("Health check task started")
    logger.info("Waiting for backend services to start up...")
    await asyncio.sleep(5)
    logger.info("Starting health checks...")
    
    async with ClientSession() as session:
        cycle = 0
        while True:
            cycle += 1
            logger.info(f"Health check cycle {cycle} started")
            try:
                for service in BACKEND_SERVICES:
                    # Skip busy backends
                    if BACKEND_BUSY.get(service, False):
                        logger.info(f"Skipping health check for busy backend {service}")
                        continue
                    
                    try:
                        logger.info(f"Checking health of {service}")
                        timeout = aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT, connect=5)
                        async with session.get(f"{service}/health", timeout=timeout) as response:
                            if response.status == 200:
                                if not BACKEND_STATUS[service]:  # Log only status changes
                                    logger.info(f"Backend {service} is now healthy")
                                BACKEND_STATUS[service] = True
                            else:
                                if BACKEND_STATUS[service]:  # Log only status changes
                                    logger.warning(f"Backend {service} is now unhealthy: HTTP {response.status}")
                                BACKEND_STATUS[service] = False
                    except asyncio.TimeoutError:
                        # Don't immediately mark as unhealthy on timeout
                        logger.warning(f"Backend {service} health check timeout (may be busy)")
                    except Exception as e:
                        if BACKEND_STATUS[service]:  # Log only status changes
                            logger.error(f"Backend {service} is now unhealthy: {str(e)}")
                        BACKEND_STATUS[service] = False
                
                # Log current status
                healthy_count = sum(1 for status in BACKEND_STATUS.values() if status)
                logger.info(f"Health check cycle {cycle} complete: {healthy_count}/{len(BACKEND_SERVICES)} backends healthy")
            except Exception as e:
                logger.error(f"Error in health check cycle {cycle}: {str(e)}")
            
            logger.info(f"Waiting {HEALTH_CHECK_INTERVAL} seconds before next health check")
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)

def get_healthy_backends() -> List[str]:
    """Get list of currently healthy backends"""
    return [service for service, is_healthy in BACKEND_STATUS.items() if is_healthy]

def get_available_backends() -> List[str]:
    """Get list of currently available (healthy and not busy) backends"""
    return [service for service, is_healthy in BACKEND_STATUS.items() 
            if is_healthy and not BACKEND_BUSY.get(service[0], False)]

def get_idle_backend() -> Optional[str]:
    """Get an idle backend using round-robin algorithm"""
    global current_index
    available_backends = [
        service for service in BACKEND_STATUS.items() 
        if service[1] and not BACKEND_BUSY.get(service[0], False)
    ]
    
    if not available_backends:
        return None
    
    # Round-robin selection from available backends
    backend = available_backends[current_index % len(available_backends)][0]
    current_index = (current_index + 1) % len(available_backends)
    return backend

async def add_request_to_queue(request: web.Request, request_body: bytes) -> asyncio.Future:
    """Add a request to the queue and return a future for the result"""
    if len(REQUEST_QUEUE) >= MAX_QUEUE_SIZE:
        raise web.HTTPServiceUnavailable(reason="Request queue is full")
    
    request_id = str(uuid.uuid4())
    future = asyncio.get_event_loop().create_future()
    
    # Handle multipart data
    form_data = None
    file_size = None
    if request.content_type and 'multipart/form-data' in request.content_type:
        # Parse multipart data and store as FormData
        try:
            form_data = aiohttp.FormData()
            reader = await request.multipart()
            
            while True:
                field = await reader.next()
                if field is None:
                    break
                
                if field.filename:
                    # File field - read the content
                    content = await field.read()
                    form_data.add_field(field.name, content, filename=field.filename, content_type=field.content_type)
                    # Store file size for progressive timeout
                    file_size = len(content)
                else:
                    # Regular field
                    content = await field.text()
                    form_data.add_field(field.name, content)
        except Exception as e:
            logger.error(f"Error parsing multipart data: {e}")
            form_data = None
    else:
        # For non-multipart requests, use request body size
        file_size = len(request_body)
    
    queued_request = QueuedRequest(
        request_id=request_id,
        request=request,
        request_body=request_body,
        form_data=form_data,
        future=future,
        timestamp=asyncio.get_event_loop().time(),
        file_size=file_size
    )
    
    REQUEST_QUEUE.append(queued_request)
    logger.info(f"Added request {request_id} to queue. Queue length: {len(REQUEST_QUEUE)}")
    if file_size:
        logger.info(f"Request {request_id} file size: {file_size / (1024*1024):.2f}MB")
    return future

async def process_queue():
    """Process requests from the queue when backends become available"""
    logger.info("Queue processor task started")
    
    while True:
        try:
            # Wait for either queue items or available backends
            if not REQUEST_QUEUE:
                await asyncio.sleep(0.1)
                continue
            
            # Try to get an idle backend
            backend = get_idle_backend()
            if not backend:
                await asyncio.sleep(0.1)
                continue
            
            # Get the next request from queue
            queued_request = REQUEST_QUEUE.popleft()
            logger.info(f"Processing queued request {queued_request.request_id} on backend {backend}")
            
            # Mark backend as busy
            BACKEND_BUSY[backend] = True
            ACTIVE_REQUESTS[backend] = queued_request
            
            # Create task to process the request
            asyncio.create_task(
                process_queued_request(backend, queued_request)
            )
            
        except Exception as e:
            logger.error(f"Error in queue processor: {e}")
            await asyncio.sleep(1)

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

async def process_queued_request(backend: str, queued_request: QueuedRequest):
    """Process a single queued request on a specific backend"""
    try:
        # Process request with progressive timeout
        timeout = calculate_request_timeout(queued_request.file_size)
        logger.info(f"Processing request {queued_request.request_id} with timeout {timeout}s")
        
        result = await asyncio.wait_for(
            process_request_on_backend(backend, queued_request.request, queued_request.request_body, queued_request.form_data),
            timeout=timeout
        )
        
        # Fulfill the future
        queued_request.future.set_result(result)
        
    except asyncio.TimeoutError:
        logger.error(f"Request {queued_request.request_id} timed out on backend {backend}")
        queued_request.future.set_exception(
            web.HTTPGatewayTimeout(reason=f"Request timed out on backend {backend}")
        )
    except Exception as e:
        logger.error(f"Error processing request {queued_request.request_id} on backend {backend}: {e}")
        queued_request.future.set_exception(
            web.HTTPInternalServerError(reason=f"Error processing request: {str(e)}")
        )
    finally:
        # Clean up
        if backend in ACTIVE_REQUESTS:
            del ACTIVE_REQUESTS[backend]
        BACKEND_BUSY[backend] = False
        logger.info(f"Backend {backend} is now free")

async def process_request_on_backend(backend: str, request: web.Request, request_body: bytes, form_data: Optional[aiohttp.FormData] = None) -> web.Response:
    """Process a request on a specific backend"""
    try:
        # Use progressive connection timeout based on file size
        file_size = len(request_body) if request_body else 0
        connect_timeout = 30  # Base connection timeout
        total_timeout = calculate_request_timeout(file_size)
        
        # Create session with longer timeouts for large files
        timeout = aiohttp.ClientTimeout(
            total=total_timeout,
            connect=connect_timeout,
            sock_connect=connect_timeout,
            sock_read=total_timeout
        )
        
        async with ClientSession(timeout=timeout) as session:
            # Prepare headers
            headers = {}
            for key, value in request.headers.items():
                if key.lower() not in ['content-length', 'host']:
                    headers[key] = value
            
            # Forward request
            if form_data:
                # Use pre-parsed FormData for multipart requests
                async with session.post(
                    f"{backend}/inference",
                    data=form_data,
                    headers=headers
                ) as response:
                    response_data = await response.read()
                    return web.Response(
                        body=response_data,
                        status=response.status,
                        headers={"Content-Type": response.headers.get("Content-Type", "application/json")}
                    )
            else:
                # Non-multipart request
                async with session.post(
                    f"{backend}/inference",
                    data=request_body,
                    headers=headers
                ) as response:
                    response_data = await response.read()
                    return web.Response(
                        body=response_data,
                        status=response.status,
                        headers={"Content-Type": response.headers.get("Content-Type", "application/json")}
                    )
    except Exception as e:
        logger.error(f"Error forwarding request to backend {backend}: {e}")
        raise

async def inference_handler(request):
    """Handle inference requests by queuing and forwarding to backend services"""
    try:
        request_id = str(uuid.uuid4())
        logger.info(f"Received inference request {request_id}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request content length: {request.content_length}")
        logger.info(f"Request content type: {request.headers.get('Content-Type')}")

        # Read request body once
        try:
            # 检查是否是multipart/form-data上传
            content_type = request.headers.get('Content-Type', '')
            if 'multipart/form-data' in content_type:
                # 使用更简单的方式读取multipart数据
                logger.info(f"开始接收multipart数据: {request.content_length} bytes")

                # 直接读取整个请求体，但增加超时保护
                try:
                    request_body = await asyncio.wait_for(
                        request.read(),
                        timeout=300.0  # 5分钟读取超时
                    )
                    logger.info(f"✅ Multipart数据接收完成！总大小: {len(request_body)} bytes")
                except asyncio.TimeoutError:
                    logger.error(f"读取Multipart数据超时 (5分钟)")
                    raise web.HTTPRequestTimeout(reason="Multipart data read timeout (5 minutes)")
                except Exception as read_error:
                    logger.error(f"读取Multipart数据失败: {read_error}")
                    raise web.HTTPInternalServerError(reason=f"Failed to read multipart data: {read_error}")
            else:
                # 非multipart数据，直接读取
                try:
                    request_body = await asyncio.wait_for(
                        request.read(),
                        timeout=60.0  # 1分钟读取超时
                    )
                    logger.info(f"Request body size: {len(request_body)} bytes")
                except asyncio.TimeoutError:
                    logger.error(f"读取请求体超时 (1分钟)")
                    raise web.HTTPRequestTimeout(reason="Request body read timeout (1 minute)")
                except Exception as read_error:
                    logger.error(f"读取请求体失败: {read_error}")
                    raise web.HTTPInternalServerError(reason=f"Failed to read request body: {read_error}")

        except Exception as e:
            logger.error(f"请求体处理失败: {e}")
            raise web.HTTPInternalServerError(reason=f"Request processing failed: {e}")

        # Extract actual audio file from multipart data
        audio_data = request_body
        if 'multipart/form-data' in content_type:
            # Parse multipart to extract the audio file
            try:
                logger.info("Parsing multipart data to extract audio file...")

                # Use the already read request_body to manually parse multipart data
                # Extract boundary from Content-Type header
                import re
                boundary_match = re.search(r'boundary=(.+)', content_type)
                if not boundary_match:
                    logger.error("No boundary found in Content-Type header")
                    raise web.HTTPBadRequest(reason="No boundary found in Content-Type header")

                boundary = boundary_match.group(1)
                logger.info(f"Found boundary: {boundary}")

                # Parse multipart data manually
                audio_data = await parse_multipart_data(request_body, boundary)
                logger.info(f"Extracted audio file: {len(audio_data)} bytes")

            except Exception as e:
                logger.error(f"Failed to parse multipart data: {e}")
                raise web.HTTPBadRequest(reason=f"Failed to parse multipart data: {e}")

        # Check audio format using ffprobe
        logger.info("Checking audio format with ffprobe...")
        format_info = check_audio_format(audio_data)

        if "error" in format_info:
            logger.error(f"Audio format check failed: {format_info['error']}")
            raise web.HTTPBadRequest(reason=f"Invalid audio format: {format_info['error']}")

        # Log audio format information
        audio_format = format_info.get("format", "unknown")
        audio_duration = format_info.get("duration", 0)
        audio_bitrate = format_info.get("bit_rate", 0)

        logger.info(f"Audio format: {audio_format}, Duration: {audio_duration:.2f}s, Bitrate: {audio_bitrate}bps")

        # Validate audio format
        supported_formats = ["wav", "mp3", "flac", "aac", "ogg", "m4a", "wma"]
        if audio_format not in supported_formats and "unknown" not in audio_format:
            logger.warning(f"Unsupported audio format: {audio_format}")
            # We'll try to process it anyway, as ffmpeg might handle it

        # Check if there are any healthy backends
        healthy_backends = get_healthy_backends()
        if not healthy_backends:
            logger.error("No healthy backends available")
            raise web.HTTPServiceUnavailable(reason="No healthy backend services available")

        # Get available backends
        available_backends = get_available_backends()
        logger.info(f"Available backends: {len(available_backends)}/{len(healthy_backends)}")

        # Check if we should use distributed processing
        should_use_distributed = await distributed_processor.should_distribute(
            len(audio_data), len(available_backends)
        )

        if should_use_distributed and len(available_backends) > 1:
            logger.info(f"Using distributed processing for request {request_id}")
            return await process_distributed_request(request, request_body, available_backends, content_type)
        else:
            logger.info(f"Using single worker processing for request {request_id}")
            return await process_single_worker_request(request, request_body, available_backends)

    except web.HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise web.HTTPInternalServerError(reason=f"Error processing request: {str(e)}")

async def process_distributed_request(request, request_body: bytes, available_backends: List[str], content_type: str):
    """Process request using distributed workers"""
    request_id = str(uuid.uuid4())
    timeout = calculate_request_timeout(len(request_body))

    logger.info(f"Processing distributed request {request_id} with {len(available_backends)} workers")
    logger.info(f"Using timeout of {timeout}s")

    # Create session with timeout
    timeout_obj = aiohttp.ClientTimeout(
        total=timeout,
        connect=30,
        sock_connect=30,
        sock_read=timeout
    )

    try:
        async with ClientSession(timeout=timeout_obj) as session:
            # Prepare headers
            headers = {}
            for key, value in request.headers.items():
                if key.lower() not in ['content-length', 'host']:
                    headers[key] = value

            # Process distributed
            result = await asyncio.wait_for(
                distributed_processor.process_distributed(
                    request_body, available_backends, session, headers
                ),
                timeout=timeout
            )

            # Debug: check result
            logger.info(f"DEBUG: Distributed result length: {len(result) if result else 'None/Empty'}")
            if not result or len(result.strip()) == 0:
                logger.warning("WARNING: Empty SRT result returned from distributed processing!")
                raise web.HTTPInternalServerError(reason="Distributed processing returned empty SRT")

            logger.info(f"Distributed request {request_id} completed successfully")

            # Return JSON response like individual workers do
            return web.json_response({
                "code": 0,
                "msg": "ok",
                "data": result
            })

    except asyncio.TimeoutError:
        logger.error(f"Distributed request {request_id} timed out")
        raise web.HTTPGatewayTimeout(reason="Distributed processing timed out")
    except Exception as e:
        logger.error(f"Error in distributed processing for request {request_id}: {e}")
        raise web.HTTPInternalServerError(reason=f"Distributed processing failed: {str(e)}")

async def process_single_worker_request(request, request_body: bytes, available_backends: List[str]):
    """Process request using single worker (original logic)"""
    request_id = str(uuid.uuid4())
    timeout = calculate_request_timeout(len(request_body))
    logger.info(f"Using timeout of {timeout}s for request {request_id}")

    # Try to get an idle backend immediately
    backend = get_idle_backend()
    if backend:
        logger.info(f"Found idle backend {backend}, processing immediately")
        try:
            # Mark backend as busy
            BACKEND_BUSY[backend] = True

            # Process request immediately
            result = await asyncio.wait_for(
                process_request_on_backend(backend, request, request_body),
                timeout=timeout
            )
            BACKEND_BUSY[backend] = False
            logger.info(f"Request {request_id} completed immediately on backend {backend}")
            return result
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out on backend {backend}")
            BACKEND_BUSY[backend] = False
            raise web.HTTPGatewayTimeout(reason=f"Request timed out on backend {backend}")
        except Exception as e:
            logger.error(f"Error processing request {request_id} immediately: {e}")
            BACKEND_BUSY[backend] = False
            # Fall back to queue
            logger.info(f"Falling back to queue for request {request_id}")

    # No idle backend available, add to queue
    logger.info(f"No idle backend available, queueing request {request_id}")
    try:
        future = await add_request_to_queue(request, request_body)
        logger.info(f"Waiting for queued request {request_id} to be processed")

        # Wait for the result with progressive timeout
        result = await asyncio.wait_for(future, timeout=timeout)
        logger.info(f"Queued request {request_id} completed successfully")
        return result

    except asyncio.TimeoutError:
        logger.error(f"Queued request {request_id} timed out")
        raise web.HTTPGatewayTimeout(reason="Request timed out in queue")
    except Exception as e:
        logger.error(f"Error with queued request {request_id}: {e}")
        if isinstance(e, web.HTTPException):
            raise
        raise web.HTTPInternalServerError(reason=f"Error processing queued request: {str(e)}")

async def health_handler(request):
    """Health check endpoint"""
    healthy_backends = get_healthy_backends()
    available_backends = [
        service for service in BACKEND_STATUS.items()
        if service[1] and not BACKEND_BUSY.get(service[0], False)
    ]

    # Get distributed processing stats
    distributed_stats = distributed_processor.get_processing_stats()

    status = {
        "status": "healthy" if healthy_backends else "degraded",
        "healthy_backends": len(healthy_backends),
        "available_backends": len(available_backends),
        "total_backends": len(BACKEND_SERVICES),
        "queue_length": len(REQUEST_QUEUE),
        "max_queue_size": MAX_QUEUE_SIZE,
        "distributed_processing": distributed_stats,
        "backends": {
            service: {
                "healthy": BACKEND_STATUS[service],
                "busy": BACKEND_BUSY.get(service, False),
                "active_request": ACTIVE_REQUESTS.get(service, {}).get('request_id') if service in ACTIVE_REQUESTS else None
            }
            for service in BACKEND_SERVICES
        }
    }
    return web.json_response(status)

async def start_background_tasks(app):
    """Start background tasks after app startup"""
    global queue_processor_task
    
    logger.info("Starting background tasks...")
    
    # Start health check task
    asyncio.create_task(health_check_task())
    
    # Start queue processor task
    queue_processor_task = asyncio.create_task(process_queue())
    logger.info("Queue processor task started")

async def cleanup_background_tasks(app):
    """Clean up background tasks on app shutdown"""
    global queue_processor_task
    
    logger.info("Cleaning up background tasks...")
    
    if queue_processor_task:
        queue_processor_task.cancel()
        try:
            await queue_processor_task
        except asyncio.CancelledError:
            logger.info("Queue processor task cancelled")
    
    logger.info("Background tasks cleaned up")

def init_app():
    """Initialize application"""
    app = web.Application(client_max_size=500*1024*1024)  # 500MB limit
    
    # Add routes
    app.router.add_post('/inference', inference_handler)
    app.router.add_get('/health', health_handler)
    
    # Start background tasks after app is fully initialized
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    
    return app

if __name__ == '__main__':
    port = int(os.getenv("LB_PORT", "5001"))
    logger.info(f"Starting enhanced load balancer on port {port}")
    logger.info(f"Backend services: {BACKEND_SERVICES}")
    logger.info(f"Max queue size: {MAX_QUEUE_SIZE}")
    logger.info(f"Request timeout: {REQUEST_TIMEOUT}s")
    logger.info(f"Health check interval: {HEALTH_CHECK_INTERVAL}s")
    logger.info("Initializing app...")
    
    app = init_app()
    logger.info("App initialized, starting web server...")
    
    try:
        logger.info("Starting web server...")
        web.run_app(app, host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Failed to start web server: {e}")
        raise
