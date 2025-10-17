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
import time

# Import distributed processing components
from distributed_processor import DistributedProcessor

# Import authentication middleware
from auth_middleware import get_auth

# SQLite database setup
import sqlite3
from pathlib import Path



# Configure logging
logging.basicConfig(level=logging.DEBUG)  # 改为DEBUG级别
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", "tus_uploads.db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/tus_uploads")

# Ensure upload directory exists
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


class TusDatabase:
    """SQLite database manager for Tus uploads and ASR tasks"""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database and create tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create tus_uploads table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tus_uploads (
                    id TEXT PRIMARY KEY,
                    offset INTEGER DEFAULT 0,
                    length INTEGER,
                    file_path TEXT,
                    metadata TEXT,
                    status TEXT DEFAULT 'uploading',
                    task_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create asr_tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS asr_tasks (
                    task_id TEXT PRIMARY KEY,
                    filename TEXT,
                    file_size INTEGER,
                    language TEXT,
                    model TEXT,
                    callback_url TEXT,
                    status TEXT DEFAULT 'pending_upload',
                    upload_id TEXT,
                    wav_file_path TEXT,
                    srt_file_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (upload_id) REFERENCES tus_uploads (id)
                )
            """)

            conn.commit()
            logger.info("Database initialized successfully")

    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)

    def create_upload(self, upload_id: str, length: int, metadata: dict = None) -> str:
        """Create a new upload record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            file_path = f"{UPLOAD_DIR}/{upload_id}"
            metadata_str = json.dumps(metadata) if metadata else None

            cursor.execute("""
                INSERT INTO tus_uploads (id, length, file_path, metadata)
                VALUES (?, ?, ?, ?)
            """, (upload_id, length, file_path, metadata_str))

            conn.commit()
            return upload_id

    def get_upload(self, upload_id: str) -> dict:
        """Get upload record by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tus_uploads WHERE id = ?", (upload_id,))
            row = cursor.fetchone()

            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None

    def update_upload_offset(self, upload_id: str, offset: int):
        """Update upload offset"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE tus_uploads SET offset = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (offset, upload_id))
            conn.commit()

    def complete_upload(self, upload_id: str, task_id: str = None):
        """Mark upload as completed"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE tus_uploads SET status = 'completed', task_id = ?,
                updated_at = CURRENT_TIMESTAMP WHERE id = ?
            """, (task_id, upload_id))
            conn.commit()

    def create_asr_task(self, task_id: str, filename: str, file_size: int,
                        metadata: dict, callback_url: str = None) -> str:
        """Create a new ASR task record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO asr_tasks (task_id, filename, file_size, language,
                model, callback_url)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (task_id, filename, file_size, metadata.get('language'),
                   metadata.get('model'), callback_url))
            conn.commit()
            return task_id

    def get_asr_task(self, task_id: str) -> dict:
        """Get ASR task by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM asr_tasks WHERE task_id = ?", (task_id,))
            row = cursor.fetchone()

            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None

    def update_task_status(self, task_id: str, status: str, wav_file_path: str = None):
        """Update ASR task status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE asr_tasks SET status = ?, wav_file_path = ?,
                updated_at = CURRENT_TIMESTAMP WHERE task_id = ?
            """, (status, wav_file_path, task_id))
            conn.commit()

    def complete_task(self, task_id: str, srt_file_path: str):
        """Complete ASR task with SRT file path"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE asr_tasks SET status = 'completed', srt_file_path = ?,
                updated_at = CURRENT_TIMESTAMP WHERE task_id = ?
            """, (srt_file_path, task_id))
            conn.commit()


# Initialize database
ts_db = TusDatabase()


async def parse_multipart_data(data: bytes, boundary: str) -> bytes:
    """Parse multipart data and extract audio file content"""
    try:
        # Convert boundary to bytes with proper formatting
        boundary_bytes = f"--{boundary}".encode('utf-8')
        end_boundary_bytes = f"--{boundary}--".encode('utf-8')

        logger.debug(f"Parsing multipart data with boundary: {boundary}")
        logger.debug(f"Boundary bytes: {boundary_bytes}")
        logger.debug(f"Total data size: {len(data)} bytes")

        # First, try to find boundaries with exact match
        boundary_positions = []
        start_idx = 0
        while True:
            boundary_idx = data.find(boundary_bytes, start_idx)
            if boundary_idx == -1:
                break
            boundary_positions.append(boundary_idx)
            start_idx = boundary_idx + len(boundary_bytes)

        logger.debug(f"Found {len(boundary_positions)} boundary positions with exact match")

        # If no exact matches, try a more flexible approach
        if len(boundary_positions) == 0:
            logger.debug("No exact boundary matches, trying flexible approach")
            # Look for the boundary without the leading --
            flexible_boundary = boundary.encode('utf-8')
            flexible_positions = []
            start_idx = 0
            while True:
                boundary_idx = data.find(flexible_boundary, start_idx)
                if boundary_idx == -1:
                    break
                # Check if this looks like a valid boundary (preceded by -- or newline)
                if boundary_idx == 0 or data[boundary_idx-1:boundary_idx] in [b'-', b'\n']:
                    flexible_positions.append(boundary_idx)
                start_idx = boundary_idx + len(flexible_boundary)

            if len(flexible_positions) > 0:
                boundary_positions = flexible_positions
                logger.debug(f"Found {len(boundary_positions)} boundary positions with flexible match")

        if len(boundary_positions) == 0:
            # Last resort: try to find any section that looks like audio data
            logger.debug("No boundaries found, trying to locate audio data directly")
            # Look for RIFF header (WAV), ID3 (MP3), or OggS (OGG)
            audio_patterns = [b'RIFF', b'ID3', b'OggS']
            for pattern in audio_patterns:
                pattern_pos = data.find(pattern)
                if pattern_pos != -1 and pattern_pos > 100:  # Not at the very beginning (skip headers)
                    # Found what looks like audio data
                    logger.debug(f"Found audio pattern {pattern} at position {pattern_pos}")
                    # Extract everything from this point until near the end
                    # Leave some buffer for the end boundary
                    content = data[pattern_pos:len(data)-100]
                    return content

            raise ValueError(f"Could not find boundary '{boundary}' in multipart data")

        logger.debug(f"Processing {len(boundary_positions)} boundary sections")

        # Process each section between boundaries
        for i in range(len(boundary_positions) - 1):
            current_boundary = boundary_positions[i]
            next_boundary = boundary_positions[i + 1]

            # Find the end of headers (double newline)
            header_end_idx = data.find(b'\r\n\r\n', current_boundary)
            if header_end_idx == -1:
                # Try with just \n\n
                header_end_idx = data.find(b'\n\n', current_boundary)
                if header_end_idx == -1:
                    logger.debug(f"No header end found for section {i}")
                    continue

            # Extract content between headers and next boundary
            content_start = header_end_idx + 4 if data[header_end_idx:header_end_idx+4] == b'\r\n\r\n' else header_end_idx + 2
            content_end = next_boundary

            # Ensure content_end is valid
            if content_end <= content_start:
                continue

            # Check if this section contains the audio file
            header_part = data[current_boundary:header_end_idx].decode('utf-8', errors='ignore')
            logger.debug(f"Section {i} headers: {header_part[:200]}...")

            if 'filename=' in header_part or 'name="audio"' in header_part or 'name="file"' in header_part:
                # This is the audio content
                content = data[content_start:content_end].rstrip(b'\r\n')
                logger.debug(f"Found audio content: {len(content)} bytes")
                return content

        # Also check the last section (before final boundary)
        if len(boundary_positions) >= 1:
            last_boundary = boundary_positions[-1]
            header_end_idx = data.find(b'\r\n\r\n', last_boundary)
            if header_end_idx == -1:
                header_end_idx = data.find(b'\n\n', last_boundary)

            if header_end_idx != -1:
                content_start = header_end_idx + 4 if data[header_end_idx:header_end_idx+4] == b'\r\n\r\n' else header_end_idx + 2
                # Find the end boundary
                end_boundary_idx = data.find(end_boundary_bytes, content_start)
                if end_boundary_idx != -1:
                    content_end = end_boundary_idx
                else:
                    content_end = len(data)

                header_part = data[last_boundary:header_end_idx].decode('utf-8', errors='ignore')
                if 'filename=' in header_part or 'name="audio"' in header_part or 'name="file"' in header_part:
                    content = data[content_start:content_end].rstrip(b'\r\n')
                    logger.debug(f"Found audio content in last section: {len(content)} bytes")
                    return content

        logger.error("No audio file found in multipart data")
        # Debug: print first 500 bytes to understand the structure
        logger.debug(f"First 500 bytes of data: {data[:500]}")
        logger.debug(f"Last 500 bytes of data: {data[-500:]}")
        raise ValueError("No audio file found in multipart data")

    except Exception as e:
        logger.error(f"Error parsing multipart data: {e}")
        # Debug: show more details about the error
        logger.debug(f"Boundary: {boundary}")
        logger.debug(f"Boundary bytes: {boundary_bytes}")
        logger.debug(f"Data size: {len(data)}")
        raise

def generate_backend_services():
    """动态生成后端服务列表"""
    try:
        num_gpus = int(os.getenv("NUM_GPUS", "4"))
        instances_per_gpu = int(os.getenv("INSTANCES_PER_GPU", "2"))
        start_port = int(os.getenv("START_PORT", "5002"))

        services = []
        for gpu_id in range(num_gpus):
            for instance in range(instances_per_gpu):
                port = start_port + gpu_id * instances_per_gpu + instance
                services.append(f"http://localhost:{port}")

        logger.info(f"Generated {len(services)} backend services for {num_gpus} GPUs x {instances_per_gpu} instances: {services}")
        return services
    except Exception as e:
        logger.error(f"Error generating backend services: {e}")
        # 回退到默认配置
        return ["http://localhost:5002", "http://localhost:5003", "http://localhost:5004", "http://localhost:5005"]

# Load backend services from environment variables or generate dynamically
BACKEND_SERVICES_ENV = os.getenv("BACKEND_SERVICES")
if BACKEND_SERVICES_ENV:
    BACKEND_SERVICES = BACKEND_SERVICES_ENV.split(",")
    logger.info(f"Using BACKEND_SERVICES from environment: {BACKEND_SERVICES}")
else:
    BACKEND_SERVICES = generate_backend_services() 
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
    audio_data: Optional[bytes] = None  # Pre-extracted audio data to avoid re-parsing

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
    available = []
    for service, is_healthy in BACKEND_STATUS.items():
        is_busy = BACKEND_BUSY.get(service, False)
        is_distributed_busy = service in distributed_processor.busy_backends
        if is_healthy and not is_busy and not is_distributed_busy:
            available.append(service)
        logger.debug(f"Backend {service}: healthy={is_healthy}, busy={is_busy}, distributed_busy={is_distributed_busy}, available={is_healthy and not is_busy and not is_distributed_busy}")
    return available

def get_idle_backend() -> Optional[str]:
    """Get an idle backend using round-robin algorithm"""
    global current_index
    available_backends = []
    for service, is_healthy in BACKEND_STATUS.items():
        is_busy = BACKEND_BUSY.get(service, False)
        is_distributed_busy = service in distributed_processor.busy_backends
        if is_healthy and not is_busy and not is_distributed_busy:
            available_backends.append((service, is_healthy))

    if not available_backends:
        return None

    # Round-robin selection from available backends
    backend = available_backends[current_index % len(available_backends)][0]
    current_index = (current_index + 1) % len(available_backends)
    logger.debug(f"Selected idle backend: {backend}")
    return backend

async def add_request_to_queue(request: web.Request, request_body: bytes) -> Tuple[asyncio.Future, str]:
    """Add a request to the queue and return a future for the result and the request ID"""
    if len(REQUEST_QUEUE) >= MAX_QUEUE_SIZE:
        raise web.HTTPServiceUnavailable(reason="Request queue is full")

    request_id = str(uuid.uuid4())
    future = asyncio.get_event_loop().create_future()

    # Handle multipart data
    form_data = None
    # Determine content type from full header to preserve boundary
    content_type_header = request.headers.get('Content-Type', '') or ''
    is_multipart = 'multipart/form-data' in content_type_header.lower()
    # Default file size to raw request_body length; update later if we extract audio_data
    file_size = len(request_body) if request_body else 0

    # Pre-extract audio data for multipart requests to avoid re-parsing later
    audio_data = None
    if is_multipart:
        try:
            import re
            # Extract boundary from full Content-Type header
            boundary_match = re.search(r'boundary\s*=\s*([^;\s]+)', content_type_header, re.IGNORECASE)
            if not boundary_match:
                # Fallback to string splitting if regex fails
                if 'boundary=' in content_type_header:
                    boundary = content_type_header.split('boundary=')[-1].split(';')[0].strip(' "\'')
                else:
                    raise ValueError(f"No boundary found in Content-Type: {content_type_header}")
            else:
                boundary = boundary_match.group(1).strip(' "\'')
            logger.debug(f"Using boundary '{boundary}' (length: {len(boundary)}) from Content-Type")
            logger.debug(f"Pre-extracting audio data for queued request {request_id}")
            audio_data = await parse_multipart_data(request_body, boundary)
            file_size = len(audio_data)
            logger.info(f"✅ Pre-extracted audio data for queued request {request_id}: {len(audio_data)} bytes")

            # For SRT-only support, we always use 'srt' format
            # Build a minimal FormData to send downstream with fixed SRT format
            form_data = aiohttp.FormData()
            filename = f"audio_{int(time.time())}.wav"
            form_data.add_field('file', audio_data, filename=filename, content_type='audio/wav')
            form_data.add_field('response_format', 'srt')  # Always SRT
        except Exception as e:
            logger.warning(f"Failed to pre-extract audio data for queued request {request_id}: {e}")
            # Do not re-read the request stream; fall back to pattern-based extraction on the buffered body
            try:
                logger.debug("Using audio pattern extraction as fallback")
                # Look for audio headers - RIFF, ID3, OggS after some headers
                header_end = request_body.find(b'\r\n\r\n')
                if header_end > 0:
                    search_start = max(100, header_end - 1000)  # Search around headers
                    for pattern in [b'RIFF', b'ID3', b'OggS']:
                        pattern_pos = request_body.find(pattern, search_start)
                        if pattern_pos > 0 and pattern_pos < len(request_body) - 100:
                            # Extract everything from the pattern to near the end
                            audio_data = request_body[pattern_pos:len(request_body)-100]
                            file_size = len(audio_data)
                            logger.info(f"✅ Pattern-based audio extraction succeeded for queued request {request_id}: {len(audio_data)} bytes")
                            break
            except Exception as pattern_error:
                logger.error(f"Pattern extraction also failed for queued request {request_id}: {pattern_error}")
                audio_data = None

            # Build minimal FormData if we at least have audio_data
            form_data = None
            if audio_data is not None:
                form_data = aiohttp.FormData()
                filename = f"audio_{int(time.time())}.wav"
                form_data.add_field('file', audio_data, filename=filename, content_type='audio/wav')
                form_data.add_field('response_format', 'srt')  # default when we cannot parse

            # If still no audio_data, leave file_size as original request_body length
            # form_data may remain None and process_request_on_backend will send raw body


















    queued_request = QueuedRequest(
        request_id=request_id,
        request=request,
        request_body=request_body,
        form_data=form_data,
        future=future,
        timestamp=asyncio.get_event_loop().time(),
        file_size=file_size,
        audio_data=audio_data
    )

    REQUEST_QUEUE.append(queued_request)
    logger.info(f"Added request {request_id} to queue. Queue length: {len(REQUEST_QUEUE)}")
    if file_size:
        logger.info(f"Request {request_id} file size: {file_size / (1024*1024):.2f}MB")
    return future, request_id

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

async def process_queued_request_with_retry(backend: str, queued_request: QueuedRequest, max_retries: int = 2):
    """Process a single queued request on a specific backend with retry logic"""
    original_backend = backend
    tried_backends = [backend]

    for attempt in range(max_retries + 1):
        try:
            # Process request with progressive timeout
            timeout = calculate_request_timeout(queued_request.file_size)
            if attempt == 0:
                logger.info(f"Processing request {queued_request.request_id} with timeout {timeout}s on backend {backend}")
            else:
                logger.info(f"Retry attempt {attempt} for request {queued_request.request_id} on backend {backend}")

            result = await asyncio.wait_for(
                process_request_on_backend(backend, queued_request.request, queued_request.request_body, queued_request.form_data, queued_request.audio_data),
                timeout=timeout
            )

            # Success! Fulfill the future
            queued_request.future.set_result(result)
            logger.info(f"Request {queued_request.request_id} completed successfully on backend {backend}" +
                       (f" (attempt {attempt})" if attempt > 0 else ""))
            # Make sure backend is marked as not busy
            BACKEND_BUSY[backend] = False
            logger.info(f"🔓 Backend {backend} marked as available after completing request {queued_request.request_id}")
            return

        except asyncio.TimeoutError:
            logger.warning(f"Request {queued_request.request_id} timed out on backend {backend} (attempt {attempt})")
            if attempt < max_retries:
                # Mark current backend as temporarily unhealthy and try a different one
                BACKEND_STATUS[backend] = False
                logger.info(f"Marked backend {backend} as temporarily unhealthy due to timeout")

                # Try to find a different healthy backend
                new_backend = get_idle_backend()
                if new_backend and new_backend not in tried_backends:
                    tried_backends.append(new_backend)
                    backend = new_backend
                    continue
            else:
                queued_request.future.set_exception(
                    web.HTTPGatewayTimeout(reason=f"Request timed out after {attempt + 1} attempts")
                )
                break

        except Exception as e:
            error_msg = str(e)
            is_connection_error = any(keyword in error_msg.lower() for keyword in
                                     ['connection refused', 'connection reset', 'connection failed',
                                      'connect call failed', 'max retries exceeded'])

            if is_connection_error and attempt < max_retries:
                logger.warning(f"Request {queued_request.request_id} failed with connection error on backend {backend}: {error_msg} (attempt {attempt})")

                # Mark current backend as temporarily unhealthy
                BACKEND_STATUS[backend] = False
                logger.info(f"Marked backend {backend} as temporarily unhealthy due to connection error")

                # Try to find a different healthy backend
                new_backend = get_idle_backend()
                if new_backend and new_backend not in tried_backends:
                    tried_backends.append(new_backend)
                    backend = new_backend
                    logger.info(f"Retrying request {queued_request.request_id} on different backend {backend}")
                    continue
                else:
                    logger.error(f"No alternative backend available for retry of request {queued_request.request_id}")
                    queued_request.future.set_exception(
                        web.HTTPInternalServerError(reason=f"All backends failed or busy after {attempt + 1} attempts: {error_msg}")
                    )
                    break
            else:
                logger.error(f"Request {queued_request.request_id} failed on backend {backend}: {error_msg} (attempt {attempt})")
                queued_request.future.set_exception(
                    web.HTTPInternalServerError(reason=f"Error processing request: {error_msg}")
                )
                break

    # Clean up - note that we might be using a different backend than the original
    final_backend = backend if attempt < max_retries else original_backend
    if final_backend in ACTIVE_REQUESTS:
        del ACTIVE_REQUESTS[final_backend]
    BACKEND_BUSY[final_backend] = False
    logger.info(f"Backend {final_backend} is now free (request failed after retries)")

async def process_queued_request(backend: str, queued_request: QueuedRequest):
    """Process a single queued request on a specific backend (wrapper for compatibility)"""
    await process_queued_request_with_retry(backend, queued_request)

async def process_request_on_backend(backend: str, request: web.Request, request_body: bytes, form_data: Optional[aiohttp.FormData] = None, audio_data: Optional[bytes] = None) -> web.Response:
    """Process a request on a specific backend"""
    try:
        # Use progressive connection timeout based on file size
        file_size = len(request_body) if request_body else 0
        connect_timeout = 600  # 增加到600秒（10分钟），与master_process保持一致
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
                # For pre-built FormData, ensure proper Content-Type
                if 'Content-Type' in headers:
                    del headers['Content-Type']  # Let aiohttp set the correct Content-Type with boundary
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
            elif audio_data is not None:
                # Use pre-extracted audio data for multipart requests
                # Create new FormData with the audio data (SRT-only)
                multipart_data = aiohttp.FormData()
                filename = f"audio_{int(time.time())}.wav"
                multipart_data.add_field('file', audio_data, filename=filename, content_type='audio/wav')
                multipart_data.add_field('response_format', 'srt')  # Always SRT

                # Set proper Content-Type header with boundary
                if 'Content-Type' in headers:
                    del headers['Content-Type']  # Let aiohttp set the correct Content-Type with boundary

                async with session.post(
                    f"{backend}/inference",
                    data=multipart_data,
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
        error_msg = str(e)
        is_connection_error = any(keyword in error_msg.lower() for keyword in
                                 ['server disconnected', 'connection refused', 'connection reset',
                                  'connection failed', 'connect call failed', 'max retries exceeded'])

        if is_connection_error:
            logger.error(f"Backend {backend} connection error: {error_msg} - marking as temporarily unhealthy")
            BACKEND_STATUS[backend] = False
        else:
            logger.error(f"Error forwarding request to backend {backend}: {e}")
        raise

async def inference_handler(request):
    """Handle inference requests by queuing and forwarding to backend services"""
    # Check API key authentication
    auth = get_auth()
    if auth.api_key:  # Only check if API key is configured
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != auth.api_key:
            logger.warning(f"Invalid or missing API key from {request.remote}")
            return web.Response(
                status=401,
                text='{"code": 401, "msg": "Invalid or missing API key", "data": ""}',
                content_type='application/json'
            )

    try:
        request_id = str(uuid.uuid4())
        logger.info(f"🔥 Received inference request {request_id}")
        logger.info(f"📋 Request headers: {dict(request.headers)}")
        logger.info(f"📦 Request content length: {request.content_length}")
        logger.info(f"📦 Request content type: {request.headers.get('Content-Type')}")
        logger.info(f"🔥 Request remote: {request.remote}")
        logger.info(f"🔥 User-Agent: {request.headers.get('User-Agent', 'Unknown')}")

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
                logger.info(f"🔍 {request_id} Parsing multipart data to extract audio file...")
                logger.debug(f"🔍 {request_id} Content-Type: {content_type}")

                # Use the already read request_body to manually parse multipart data
                # Extract boundary from Content-Type header - consistent with queue processing
                import re
                boundary_match = re.search(r'boundary\s*=\s*([^;\s]+)', content_type, re.IGNORECASE)
                if not boundary_match:
                    # Fallback to original pattern for compatibility
                    boundary_match = re.search(r'boundary=(.+)', content_type)
                    if boundary_match:
                        boundary = boundary_match.group(1).strip(' "\'')
                    else:
                        logger.error("No boundary found in Content-Type header")
                        raise web.HTTPBadRequest(reason="No boundary found in Content-Type header")
                else:
                    boundary = boundary_match.group(1).strip(' "\'')
                logger.info(f"Found boundary: {boundary}")

                # Parse multipart data manually
                try:
                    audio_data = await parse_multipart_data(request_body, boundary)
                    logger.info(f"Extracted audio file: {len(audio_data)} bytes")
                except ValueError as mp_error:
                    logger.warning(f"First multipart parse attempt failed: {mp_error}")
                    # Try alternative parsing approach
                    try:
                        # Use aiohttp's built-in parsing as fallback
                        form_data = aiohttp.FormData()
                        reader = await request.multipart()

                        while True:
                            field = await reader.next()
                            if field is None:
                                break

                            if field.filename:
                                # File field - read the content
                                content = await field.read()
                                logger.info(f"Extracted audio file via fallback: {len(content)} bytes")
                                audio_data = content
                                break
                    except Exception as fallback_error:
                        logger.error(f"Fallback parsing also failed: {fallback_error}")
                        raise web.HTTPBadRequest(reason=f"Failed to parse multipart data: {mp_error}")

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
            len(audio_data), len(healthy_backends)
        )

        if should_use_distributed and len(healthy_backends) > 1:
            # Check if we have enough available workers for distributed processing
            if len(available_backends) > 1:
                logger.info(f"🔄 Using distributed processing for request {request_id} with {len(available_backends)} workers")
                return await process_distributed_request(request, request_body, available_backends, content_type)
            else:
                # Not enough available workers, queue the request for distributed processing
                logger.info(f"🔄 Request {request_id} qualifies for distributed processing but not enough workers available. Queueing for distributed processing.")
                return await queue_for_distributed_processing(request, request_body, healthy_backends, content_type)
        else:
            logger.info(f"🔧 Using single worker processing for request {request_id} with {len(available_backends)} workers")
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
        connect=600,  # 增加到600秒，与其他超时保持一致
        sock_connect=600,
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

async def queue_for_distributed_processing(request, request_body: bytes, healthy_backends: List[str], content_type: str):
    """Queue a request for distributed processing when workers are busy"""
    request_id = str(uuid.uuid4())
    timeout = calculate_request_timeout(len(request_body))
    logger.info(f"Queueing distributed request {request_id} with timeout {timeout}s")

    # Create a distributed request queue entry
    future = asyncio.get_event_loop().create_future()

    # Add to a special distributed processing queue
    if not hasattr(queue_for_distributed_processing, 'distributed_queue'):
        queue_for_distributed_processing.distributed_queue = deque(maxlen=MAX_QUEUE_SIZE)

    distributed_request = {
        'request_id': request_id,
        'request': request,
        'request_body': request_body,
        'healthy_backends': healthy_backends,
        'content_type': content_type,
        'future': future,
        'timestamp': asyncio.get_event_loop().time()
    }

    if len(queue_for_distributed_processing.distributed_queue) >= MAX_QUEUE_SIZE:
        raise web.HTTPServiceUnavailable(reason="Distributed processing queue is full")

    queue_for_distributed_processing.distributed_queue.append(distributed_request)
    logger.info(f"Added distributed request {request_id} to queue. Queue length: {len(queue_for_distributed_processing.distributed_queue)}")

    # Start distributed queue processor if not already running
    if not hasattr(queue_for_distributed_processing, 'processor_task') or queue_for_distributed_processing.processor_task.done():
        queue_for_distributed_processing.processor_task = asyncio.create_task(process_distributed_queue())

    try:
        # Wait for the distributed processing to complete
        result = await asyncio.wait_for(future, timeout=timeout)
        logger.info(f"Distributed queued request {request_id} completed successfully")
        return result
    except asyncio.TimeoutError:
        logger.error(f"Distributed queued request {request_id} timed out")
        raise web.HTTPGatewayTimeout(reason="Distributed processing timed out in queue")
    except Exception as e:
        logger.error(f"Error with distributed queued request {request_id}: {e}")
        if isinstance(e, web.HTTPException):
            raise
        raise web.HTTPInternalServerError(reason=f"Error in distributed queue processing: {str(e)}")

async def process_distributed_queue():
    """Process distributed requests from the queue when enough workers become available"""
    logger.info("Distributed queue processor task started")

    while True:
        try:
            # Check if there are queued distributed requests
            if not hasattr(queue_for_distributed_processing, 'distributed_queue') or not queue_for_distributed_processing.distributed_queue:
                await asyncio.sleep(0.1)
                continue

            # Check if we have enough available workers for distributed processing
            available_backends = get_available_backends()
            if len(available_backends) <= 1:
                await asyncio.sleep(0.5)  # Wait longer for distributed processing
                continue

            # Get the next distributed request from queue
            distributed_request = queue_for_distributed_processing.distributed_queue.popleft()
            request_id = distributed_request['request_id']
            logger.info(f"Processing distributed queued request {request_id} with {len(available_backends)} workers")

            # Process the distributed request asynchronously
            asyncio.create_task(
                process_distributed_queued_request(distributed_request, available_backends)
            )

        except Exception as e:
            logger.error(f"Error in distributed queue processor: {e}")
            await asyncio.sleep(1)

async def process_distributed_queued_request(distributed_request: dict, available_backends: List[str]):
    """Process a single distributed queued request"""
    request_id = distributed_request['request_id']
    request = distributed_request['request']
    request_body = distributed_request['request_body']
    content_type = distributed_request['content_type']
    future = distributed_request['future']

    try:
        timeout = calculate_request_timeout(len(request_body))
        logger.info(f"Processing distributed queued request {request_id} with timeout {timeout}s")

        result = await asyncio.wait_for(
            process_distributed_request(request, request_body, available_backends, content_type),
            timeout=timeout
        )

        # Success! Fulfill the future
        future.set_result(result)
        logger.info(f"Distributed queued request {request_id} completed successfully")

    except Exception as e:
        logger.error(f"Error processing distributed queued request {request_id}: {e}")
        if not future.done():
            future.set_exception(e)

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

            # Process request immediately (no retry for direct requests)
            result = await asyncio.wait_for(
                process_request_on_backend(backend, request, request_body),
                timeout=timeout
            )
            BACKEND_BUSY[backend] = False
            logger.info(f"Request {request_id} completed immediately on backend {backend}")
            logger.info(f"🔓 Backend {backend} marked as available after completing immediate request {request_id}")
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
        future, queued_request_id = await add_request_to_queue(request, request_body)
        logger.info(f"Waiting for queued request {queued_request_id} to be processed")

        # Wait for the result with progressive timeout
        result = await asyncio.wait_for(future, timeout=timeout)
        logger.info(f"Queued request {queued_request_id} completed successfully")
        return result

    except asyncio.TimeoutError:
        logger.error(f"Queued request {queued_request_id} timed out")
        raise web.HTTPGatewayTimeout(reason="Request timed out in queue")
    except Exception as e:
        logger.error(f"Error with queued request {queued_request_id}: {e}")
        if isinstance(e, web.HTTPException):
            raise
        raise web.HTTPInternalServerError(reason=f"Error processing queued request: {str(e)}")

async def collect_backend_stats(service: str) -> Dict[str, Any]:
    """收集单个后端实例的统计数据"""
    try:
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{service}/stats") as response:
                if response.status == 200:
                    stats_data = await response.json()
                    return {
                        "service": service,
                        "status": "healthy",
                        "stats": stats_data.get("stats", {}),
                        "last_updated": time.time()
                    }
                else:
                    return {
                        "service": service,
                        "status": "error",
                        "error": f"HTTP {response.status}",
                        "last_updated": time.time()
                    }
    except Exception as e:
        return {
            "service": service,
            "status": "error",
            "error": str(e),
            "last_updated": time.time()
        }

async def stats_handler(request):
    """详细统计信息接口"""
    # 收集所有后端实例的统计数据
    backend_stats = []
    if BACKEND_SERVICES:
        # 并发收集所有后端统计数据
        tasks = [collect_backend_stats(service) for service in BACKEND_SERVICES]
        backend_stats = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤异常结果
        backend_stats = [stat for stat in backend_stats if isinstance(stat, dict)]

    # 计算汇总统计
    total_stats = {
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "total_files_processed": 0,
        "total_file_size_mb": 0.0,
        "total_chunks_processed": 0,
        "total_processing_time": 0.0,
        "total_upload_time": 0.0,
        "healthy_instances": 0,
        "total_instances": len(BACKEND_SERVICES)
    }

    # 实例性能统计
    instance_performance = []

    for backend_stat in backend_stats:
        if backend_stat.get("status") == "healthy" and "stats" in backend_stat:
            stats = backend_stat["stats"]
            instance_stats = stats.get("request_stats", {})
            file_stats = stats.get("file_stats", {})
            performance_stats = stats.get("performance_stats", {})
            instance_info = stats.get("instance_info", {})

            # 累加到总计
            total_stats["total_requests"] += instance_stats.get("total_requests", 0)
            total_stats["successful_requests"] += instance_stats.get("successful_requests", 0)
            total_stats["failed_requests"] += instance_stats.get("failed_requests", 0)
            total_stats["total_files_processed"] += file_stats.get("total_files_processed", 0)
            total_stats["total_file_size_mb"] += file_stats.get("total_file_size_mb", 0)
            total_stats["total_chunks_processed"] += file_stats.get("total_chunks_processed", 0)
            total_stats["total_processing_time"] += performance_stats.get("total_processing_time", 0)
            total_stats["total_upload_time"] += performance_stats.get("total_upload_time", 0)

            if backend_stat.get("status") == "healthy":
                total_stats["healthy_instances"] += 1

            # 实例性能详情
            instance_performance.append({
                "instance_id": instance_info.get("instance_id", "unknown"),
                "service": backend_stat["service"],
                "port": instance_info.get("port", 0),
                "gpu_device": instance_info.get("gpu_device", "unknown"),
                "model_size": instance_info.get("model_size", "unknown"),
                "uptime_seconds": instance_info.get("uptime_seconds", 0),
                "request_stats": instance_stats,
                "file_stats": file_stats,
                "performance_stats": performance_stats,
                "current_status": stats.get("current_status", {}),
                "status": backend_stat.get("status", "unknown")
            })

    # 计算平均值和比率
    total_success_rate = (total_stats["successful_requests"] / total_stats["total_requests"] * 100) if total_stats["total_requests"] > 0 else 0
    avg_processing_time = (total_stats["total_processing_time"] / total_stats["successful_requests"]) if total_stats["successful_requests"] > 0 else 0
    avg_file_size = (total_stats["total_file_size_mb"] / total_stats["total_files_processed"]) if total_stats["total_files_processed"] > 0 else 0
    avg_chunks_per_file = (total_stats["total_chunks_processed"] / total_stats["total_files_processed"]) if total_stats["total_files_processed"] > 0 else 0

    # 负载均衡器状态
    healthy_backends = get_healthy_backends()
    available_backends = [
        service for service in BACKEND_STATUS.items()
        if service[1] and not BACKEND_BUSY.get(service[0], False)
    ]

    # 分布式处理统计
    distributed_stats = distributed_processor.get_processing_stats()

    stats_response = {
        "load_balancer": {
            "status": "healthy" if healthy_backends else "degraded",
            "healthy_backends": len(healthy_backends),
            "available_backends": len(available_backends),
            "total_backends": len(BACKEND_SERVICES),
            "queue_length": len(REQUEST_QUEUE),
            "max_queue_size": MAX_QUEUE_SIZE,
            "active_requests": len(ACTIVE_REQUESTS)
        },
        "aggregated_stats": {
            "total_requests": total_stats["total_requests"],
            "successful_requests": total_stats["successful_requests"],
            "failed_requests": total_stats["failed_requests"],
            "success_rate_percent": round(total_success_rate, 2),
            "total_files_processed": total_stats["total_files_processed"],
            "total_file_size_mb": round(total_stats["total_file_size_mb"], 2),
            "total_chunks_processed": total_stats["total_chunks_processed"],
            "average_file_size_mb": round(avg_file_size, 2),
            "average_chunks_per_file": round(avg_chunks_per_file, 2),
            "total_processing_time_seconds": round(total_stats["total_processing_time"], 2),
            "total_upload_time_seconds": round(total_stats["total_upload_time"], 2),
            "average_processing_time_seconds": round(avg_processing_time, 2),
            "healthy_instances": total_stats["healthy_instances"],
            "total_instances": total_stats["total_instances"]
        },
        "instance_details": instance_performance,
        "distributed_processing": distributed_stats,
        "backend_status": {
            service: {
                "healthy": BACKEND_STATUS[service],
                "busy": BACKEND_BUSY.get(service, False),
                "active_request": getattr(ACTIVE_REQUESTS.get(service), 'request_id', None) if service in ACTIVE_REQUESTS else None
            }
            for service in BACKEND_SERVICES
        },
        "timestamp": time.time()
    }

    return web.json_response(stats_response)

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
                "active_request": getattr(ACTIVE_REQUESTS.get(service), 'request_id', None) if service in ACTIVE_REQUESTS else None
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
    app.router.add_get('/stats', stats_handler)  # 新增详细统计接口

    # Start background tasks after app is fully initialized
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)

    return app

if __name__ == '__main__':
    port = int(os.getenv("LB_PORT", "5001"))
    logger.info(f"Starting enhanced load balancer on port {port}")
    logger.info(f"Backend services: {BACKEND_SERVICES}")
    logger.info(f"Total backends configured: {len(BACKEND_SERVICES)}")
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
