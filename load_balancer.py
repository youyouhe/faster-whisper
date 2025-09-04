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
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load backend services from environment variables
BACKEND_SERVICES = os.getenv("BACKEND_SERVICES", "http://localhost:5002,http://localhost:5003,http://localhost:5004").split(",")
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))  # seconds

# Global state
backend_status = {service: True for service in BACKEND_SERVICES}  # Assume all healthy initially
current_index = 0  # For round-robin distribution

async def health_check_task():
    """Periodically check health of backend services"""
    logger.info("Waiting for backend services to start up...")
    await asyncio.sleep(5)
    logger.info("Starting health checks...")
    
    async with ClientSession() as session:
        while True:
            for service in BACKEND_SERVICES:
                try:
                    async with session.get(f"{service}/health", timeout=10) as response:
                        if response.status == 200:
                            if not backend_status[service]:  # Log only status changes
                                logger.info(f"Backend {service} is now healthy")
                            backend_status[service] = True
                        else:
                            if backend_status[service]:  # Log only status changes
                                logger.warning(f"Backend {service} is now unhealthy: HTTP {response.status}")
                            backend_status[service] = False
                except Exception as e:
                    if backend_status[service]:  # Log only status changes
                        logger.error(f"Backend {service} is now unhealthy: {str(e)}")
                    backend_status[service] = False
            
            # Log current status
            healthy_count = sum(1 for status in backend_status.values() if status)
            logger.info(f"Health check complete: {healthy_count}/{len(BACKEND_SERVICES)} backends healthy")
            
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)

def get_healthy_backends() -> List[str]:
    """Get list of currently healthy backends"""
    return [service for service, is_healthy in backend_status.items() if is_healthy]

def select_backend() -> str:
    """Select a backend using round-robin algorithm"""
    global current_index
    healthy_backends = get_healthy_backends()
    
    if not healthy_backends:
        raise web.HTTPServiceUnavailable(reason="No healthy backend services available")
    
    # Round-robin selection
    backend = healthy_backends[current_index % len(healthy_backends)]
    current_index = (current_index + 1) % len(healthy_backends)
    return backend

async def inference_handler(request):
    """Handle inference requests by forwarding to backend services"""
    try:
        request_id = id(request)
        logger.info(f"Received inference request {request_id}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request content length: {request.content_length}")
        logger.info(f"Request content type: {request.headers.get('Content-Type')}")
        
        # Select a backend
        backend = select_backend()
        logger.info(f"Forwarding request {request_id} to {backend}")
        
        # Forward request - use streaming to avoid boundary issues
        async with ClientSession() as session:
            # Prepare headers - exclude content-length as aiohttp will calculate it
            headers = {}
            for key, value in request.headers.items():
                if key.lower() not in ['content-length', 'host']:
                    headers[key] = value
            
            # Try body streaming first
            if request.content_type and 'multipart/form-data' in request.content_type:
                logger.info(f"Using body streaming for multipart data")
                # Stream the raw body
                request_body = await request.read()
                logger.info(f"Request body size: {len(request_body)} bytes")
                
                async with session.post(
                    f"{backend}/inference",
                    data=request_body,
                    headers={'Content-Type': request.content_type, **headers}
                ) as response:
                    logger.info(f"Received response from {backend} for request {request_id}, status: {response.status}")
                    # Forward response back
                    response_data = await response.read()
                    logger.info(f"Response data size: {len(response_data)} bytes")
                    response_obj = web.Response(
                        body=response_data,
                        status=response.status,
                        headers={"Content-Type": response.headers.get("Content-Type", "application/json")}
                    )
                    logger.info(f"Returning response for request {request_id}")
                    return response_obj
            else:
                # Forward as regular request body
                request_body = await request.read()
                logger.info(f"Request body size: {len(request_body)} bytes")
                
                async with session.post(
                    f"{backend}/inference",
                    data=request_body,
                    headers=headers
                ) as response:
                    logger.info(f"Received response from {backend} for request {request_id}, status: {response.status}")
                    # Forward response back
                    response_data = await response.read()
                    logger.info(f"Response data size: {len(response_data)} bytes")
                    response_obj = web.Response(
                        body=response_data,
                        status=response.status,
                        headers={"Content-Type": response.headers.get("Content-Type", "application/json")}
                    )
                    logger.info(f"Returning response for request {request_id}")
                    return response_obj
                
    except web.HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise web.HTTPInternalServerError(reason=f"Error processing request: {str(e)}")

async def health_handler(request):
    """Health check endpoint"""
    healthy_backends = get_healthy_backends()
    status = {
        "status": "healthy" if healthy_backends else "degraded",
        "healthy_backends": len(healthy_backends),
        "total_backends": len(BACKEND_SERVICES),
        "backends": backend_status
    }
    return web.json_response(status)

def init_app():
    """Initialize application"""
    app = web.Application(client_max_size=100*1024*1024)  # 100MB limit
    
    # Add routes
    app.router.add_post('/inference', inference_handler)
    app.router.add_get('/health', health_handler)
    
    return app

if __name__ == '__main__':
    port = int(os.getenv("LB_PORT", "5001"))
    logger.info(f"Starting load balancer on port {port}")
    logger.info(f"Backend services: {BACKEND_SERVICES}")
    app = init_app()
    web.run_app(app, host='0.0.0.0', port=port)