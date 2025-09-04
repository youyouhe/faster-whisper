#!/usr/bin/env python3
"""
Load balancer for faster-whisper API services
Distributes requests across multiple GPU instances
"""

import os
import asyncio
import aiohttp
from aiohttp import web, ClientSession, MultipartReader
import json
from typing import List, Dict, Optional, Tuple, Any
import logging
from collections import deque
import uuid
from dataclasses import dataclass
from concurrent.futures import TimeoutError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load backend services from environment variables
BACKEND_SERVICES = os.getenv("BACKEND_SERVICES", "http://localhost:5002,http://localhost:5003,http://localhost:5004").split(",")
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))  # seconds
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "100"))  # Maximum requests in queue
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "1800"))  # seconds (30 minutes for large audio files)

@dataclass
class QueuedRequest:
    """Represents a queued request"""
    request_id: str
    request: web.Request
    request_body: bytes
    form_data: Optional[aiohttp.FormData]
    future: asyncio.Future
    timestamp: float

# Global state
BACKEND_STATUS = {service: True for service in BACKEND_SERVICES}  # Assume all healthy initially
BACKEND_BUSY = {service: False for service in BACKEND_SERVICES}  # Track busy backends
REQUEST_QUEUE = deque(maxlen=MAX_QUEUE_SIZE)  # Queue for pending requests
ACTIVE_REQUESTS = {}  # Track active requests by backend
current_index = 0  # For round-robin distribution
queue_processor_task = None  # Background task for processing queue

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
                        timeout = aiohttp.ClientTimeout(total=10, connect=5)
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
                else:
                    # Regular field
                    content = await field.text()
                    form_data.add_field(field.name, content)
        except Exception as e:
            logger.error(f"Error parsing multipart data: {e}")
            form_data = None
    
    queued_request = QueuedRequest(
        request_id=request_id,
        request=request,
        request_body=request_body,
        form_data=form_data,
        future=future,
        timestamp=asyncio.get_event_loop().time()
    )
    
    REQUEST_QUEUE.append(queued_request)
    logger.info(f"Added request {request_id} to queue. Queue length: {len(REQUEST_QUEUE)}")
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

async def process_queued_request(backend: str, queued_request: QueuedRequest):
    """Process a single queued request on a specific backend"""
    try:
        # Process request with timeout
        result = await asyncio.wait_for(
            process_request_on_backend(backend, queued_request.request, queued_request.request_body, queued_request.form_data),
            timeout=REQUEST_TIMEOUT
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
        async with ClientSession() as session:
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
        request_body = await request.read()
        logger.info(f"Request body size: {len(request_body)} bytes")
        
        # Check if there are any healthy backends
        healthy_backends = get_healthy_backends()
        if not healthy_backends:
            logger.error("No healthy backends available")
            raise web.HTTPServiceUnavailable(reason="No healthy backend services available")
        
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
                    timeout=REQUEST_TIMEOUT
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
            
            # Wait for the result with timeout
            result = await asyncio.wait_for(future, timeout=REQUEST_TIMEOUT)
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
                
    except web.HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise web.HTTPInternalServerError(reason=f"Error processing request: {str(e)}")

async def health_handler(request):
    """Health check endpoint"""
    healthy_backends = get_healthy_backends()
    available_backends = [
        service for service in BACKEND_STATUS.items() 
        if service[1] and not BACKEND_BUSY.get(service[0], False)
    ]
    
    status = {
        "status": "healthy" if healthy_backends else "degraded",
        "healthy_backends": len(healthy_backends),
        "available_backends": len(available_backends),
        "total_backends": len(BACKEND_SERVICES),
        "queue_length": len(REQUEST_QUEUE),
        "max_queue_size": MAX_QUEUE_SIZE,
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