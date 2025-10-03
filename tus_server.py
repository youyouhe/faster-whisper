#!/usr/bin/env python3
"""
Tus.io Resumable Upload Server
Handles resumable file uploads with proper Tus protocol support
Integrates with SQLite database for upload state tracking
"""

import os
import uuid
import hashlib
import json
import gzip
import bz2
import lzma
import tempfile
from typing import Dict, Optional, Any
from pathlib import Path
from urllib.parse import unquote
import asyncio
import aiohttp
from aiohttp import web, FormData
import logging
import shutil

# Import database components
from tus_database import TusDatabase
from auth_middleware import get_auth, require_auth

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/tus_uploads")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "500")) * 1024 * 1024  # 500MB default
CHUNK_SIZE = 64 * 1024  # 64KB chunks
CLEANUP_INTERVAL = int(os.getenv("CLEANUP_INTERVAL", "3600"))  # 1 hour
TUS_SERVER_PORT = int(os.getenv("TUS_SERVER_PORT", "1080"))
SHOW_PROGRESS_LOGS = os.getenv("SHOW_PROGRESS_LOGS", "true").lower() == "true"  # Enable detailed progress logs

# Ensure upload directory exists
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

def decompress_file(file_path: Path, temp_dir: Path, original_filename: str = None) -> Path:
    """
    Decompress a file if it's compressed, return the path to the decompressed file.
    If not compressed, return the original file path.

    Supported formats: .gz, .bz2, .xz

    Args:
        file_path: Path to the file to decompress
        temp_dir: Directory to store decompressed file
        original_filename: Original filename from upload metadata (optional)
    """
    # Use original filename if provided, otherwise use current file path name
    file_name = original_filename if original_filename else file_path.name

    # Check for compression extensions
    if file_name.lower().endswith('.gz'):
        decompressed_name = file_name[:-3]  # Remove .gz
        decompressed_path = temp_dir / decompressed_name
        try:
            with gzip.open(file_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Decompressed {file_path} (gzip) to {decompressed_path}")
            return decompressed_path
        except Exception as e:
            logger.error(f"Failed to decompress gzip file {file_path}: {e}")
            return file_path

    elif file_name.lower().endswith('.bz2'):
        decompressed_name = file_name[:-4]  # Remove .bz2
        decompressed_path = temp_dir / decompressed_name
        try:
            with bz2.open(file_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Decompressed {file_path} (bzip2) to {decompressed_path}")
            return decompressed_path
        except Exception as e:
            logger.error(f"Failed to decompress bzip2 file {file_path}: {e}")
            return file_path

    elif file_name.lower().endswith('.xz'):
        decompressed_name = file_name[:-3]  # Remove .xz
        decompressed_path = temp_dir / decompressed_name
        try:
            with lzma.open(file_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Decompressed {file_path} (xz) to {decompressed_path}")
            return decompressed_path
        except Exception as e:
            logger.error(f"Failed to decompress xz file {file_path}: {e}")
            return file_path

    else:
        # Not a compressed file
        return file_path


class TusServer:
    """Tus.io compatible resumable upload server"""

    def __init__(self):
        self.db = TusDatabase()
        self.upload_dir = Path(UPLOAD_DIR)
        self.max_file_size = MAX_FILE_SIZE
        self.show_progress_logs = SHOW_PROGRESS_LOGS
        self.active_uploads = {}  # Track active uploads with progress data

        # Create aiohttp app with large client size for file uploads
        self.app = web.Application(client_max_size=600*1024*1024)  # 600MB

        # Setup routes
        self.app.router.add_post('/files', self.create_upload)
        self.app.router.add_head('/files/{upload_id}', self.head_upload)
        self.app.router.add_patch('/files/{upload_id}', self.patch_upload)
        self.app.router.add_delete('/files/{upload_id}', self.delete_upload)
        self.app.router.add_get('/health', self.health_check)

    async def create_upload(self, request):
        """Create a new upload (POST /files) with API key authentication"""
        # Check API key authentication
        auth = get_auth()
        if auth.api_key:  # Only check if API key is configured
            api_key = request.headers.get('X-API-Key')
            if not api_key or api_key != auth.api_key:
                logger.warning(f"Invalid or missing API key from {request.remote}")
                return web.Response(status=401, text="Invalid or missing API key")

        try:
            # Parse Tus headers
            upload_length = request.headers.get('Upload-Length')
            upload_metadata = request.headers.get('Upload-Metadata', '')
            content_type = request.headers.get('Content-Type', '')

            # Validate required headers
            if not upload_length:
                return web.Response(status=400, text="Missing Upload-Length header")

            try:
                upload_length = int(upload_length)
            except ValueError:
                return web.Response(status=400, text="Invalid Upload-Length header")

            # Check file size limit
            if upload_length > self.max_file_size:
                return web.Response(
                    status=413,
                    text=f"File too large. Maximum size: {self.max_file_size // (1024*1024)}MB"
                )

            # Generate upload ID
            upload_id = str(uuid.uuid4())

            # Parse metadata
            metadata = {}
            if upload_metadata:
                for item in upload_metadata.split(','):
                    item = item.strip()
                    if ' ' in item:
                        key, value = item.split(' ', 1)
                        metadata[key] = unquote(value)
                    else:
                        metadata[item] = ''

            # Create file path
            file_path = self.upload_dir / upload_id

            # Store upload info in database
            upload_record = self.db.create_upload(
                upload_id=upload_id,
                length=upload_length,
                metadata=metadata
            )

            if not upload_record:
                return web.Response(status=500, text="Failed to create upload record")

            # Create empty file
            file_path.touch()
            logger.info(f"Created upload {upload_id} for {upload_length} bytes")

            # Return response with Location header
            response = web.Response(
                status=201,
                headers={
                    'Location': f'/files/{upload_id}',
                    'Tus-Resumable': '1.0.0'
                }
            )
            return response

        except Exception as e:
            logger.error(f"Error creating upload: {e}")
            return web.Response(status=500, text="Internal server error")

    async def head_upload(self, request):
        """Check upload status (HEAD /files/{upload_id})"""
        upload_id = request.match_info['upload_id']

        # Check API key authentication
        auth = get_auth()
        if auth.api_key:  # Only check if API key is configured
            api_key = request.headers.get('X-API-Key')
            if not api_key or api_key != auth.api_key:
                logger.warning(f"Invalid or missing API key from {request.remote}")
                return web.Response(status=401, text="Invalid or missing API key")

        try:
            # Get upload record
            upload_record = self.db.get_upload(upload_id)
            if not upload_record:
                return web.Response(status=404, text="Upload not found")

            # Check if upload is completed
            if upload_record['status'] == 'completed':
                return web.Response(
                    status=200,
                    headers={
                        'Upload-Offset': str(upload_record['length']),
                        'Upload-Length': str(upload_record['length']),
                        'Tus-Resumable': '1.0.0'
                    }
                )

            # Check file existence
            file_path = Path(upload_record['file_path'])
            if not file_path.exists():
                return web.Response(status=404, text="Upload file not found")

            # Get current file size as offset
            current_offset = file_path.stat().st_size

            # Check if upload is complete
            if current_offset >= upload_record['length']:
                # Mark as completed
                self.db.complete_upload(upload_id)
                return web.Response(
                    status=200,
                    headers={
                        'Upload-Offset': str(upload_record['length']),
                        'Upload-Length': str(upload_record['length']),
                        'Tus-Resumable': '1.0.0'
                    }
                )

            return web.Response(
                status=200,
                headers={
                    'Upload-Offset': str(current_offset),
                    'Upload-Length': str(upload_record['length']),
                    'Tus-Resumable': '1.0.0'
                }
            )

        except Exception as e:
            logger.error(f"Error checking upload {upload_id}: {e}")
            return web.Response(status=500, text="Internal server error")

    async def patch_upload(self, request):
        """Upload file chunk (PATCH /files/{upload_id})"""
        upload_id = request.match_info['upload_id']

        # Check API key authentication
        auth = get_auth()
        if auth.api_key:  # Only check if API key is configured
            api_key = request.headers.get('X-API-Key')
            if not api_key or api_key != auth.api_key:
                logger.warning(f"Invalid or missing API key from {request.remote}")
                return web.Response(status=401, text="Invalid or missing API key")

        try:
            # Validate Tus headers
            content_type = request.headers.get('Content-Type', '')
            if content_type != 'application/offset+octet-stream':
                return web.Response(status=400, text="Invalid Content-Type header")

            upload_offset = request.headers.get('Upload-Offset')
            if not upload_offset:
                return web.Response(status=400, text="Missing Upload-Offset header")

            try:
                upload_offset = int(upload_offset)
            except ValueError:
                return web.Response(status=400, text="Invalid Upload-Offset header")

            # Get upload record
            upload_record = self.db.get_upload(upload_id)
            if not upload_record:
                return web.Response(status=404, text="Upload not found")

            # Check offset validity
            file_path = Path(upload_record['file_path'])
            current_size = file_path.stat().st_size if file_path.exists() else 0

            if upload_offset != current_size:
                return web.Response(
                    status=409,
                    text=f"Offset mismatch. Expected: {current_size}, Got: {upload_offset}"
                )

            # Read request body
            chunk_data = await request.read()

            # Validate chunk size doesn't exceed file length
            new_offset = upload_offset + len(chunk_data)
            if new_offset > upload_record['length']:
                return web.Response(status=413, text="Chunk would exceed upload length")

            # Write chunk to file
            with open(file_path, 'ab') as f:
                f.write(chunk_data)

            # Update offset in database
            self.db.update_upload_offset(upload_id, new_offset)

            # Check if upload is complete
            if new_offset >= upload_record['length']:
                # Mark upload as completed
                self.db.complete_upload(upload_id)

                # Trigger upload completion event
                asyncio.create_task(self.handle_upload_completion(upload_id, upload_record))

                logger.info(f"Upload {upload_id} completed")

                # Clear progress tracking data
                if upload_id in self.active_uploads:
                    del self.active_uploads[upload_id]
            else:
                # Calculate progress percentage
                progress_percent = (new_offset / upload_record['length']) * 100

                # Track upload progress for speed estimation
                import time
                current_time = time.time()

                if self.show_progress_logs:
                    if upload_id not in self.active_uploads:
                        # First chunk for this upload
                        self.active_uploads[upload_id] = {
                            'start_time': current_time,
                            'start_offset': upload_offset,
                            'last_time': current_time,
                            'last_offset': upload_offset
                        }
                        logger.info(f"Upload {upload_id} progress: {new_offset}/{upload_record['length']} bytes ({progress_percent:.1f}%)")
                    else:
                        # Update progress tracking data
                        upload_data = self.active_uploads[upload_id]
                        previous_time = upload_data['last_time']
                        previous_offset = upload_data['last_offset']
                        upload_data['last_time'] = current_time
                        upload_data['last_offset'] = new_offset

                        # Calculate instantaneous speed (bytes/second) over last chunk
                        chunk_size = new_offset - previous_offset
                        chunk_time = current_time - previous_time + 0.001  # Add small value to prevent division by zero
                        instant_speed = chunk_size / chunk_time

                        # Calculate overall speed (bytes/second)
                        overall_time = current_time - upload_data['start_time']
                        overall_speed = (new_offset - upload_data['start_offset']) / (overall_time + 0.001)  # Add small value to prevent division by zero

                        # Estimate remaining time
                        remaining_bytes = upload_record['length'] - new_offset
                        if overall_speed > 0:
                            remaining_time = remaining_bytes / overall_speed
                            # Format remaining time in a human-readable format
                            if remaining_time < 60:
                                remaining_str = f"{remaining_time:.0f}s"
                            elif remaining_time < 3600:
                                remaining_str = f"{remaining_time/60:.1f}m"
                            else:
                                remaining_str = f"{remaining_time/3600:.1f}h"
                        else:
                            remaining_str = "unknown"

                        # Format speeds in human-readable format
                        def format_speed(speed_bytes):
                            if speed_bytes < 1024:
                                return f"{speed_bytes:.0f} B/s"
                            elif speed_bytes < 1024 * 1024:
                                return f"{speed_bytes/1024:.1f} KB/s"
                            else:
                                return f"{speed_bytes/(1024*1024):.1f} MB/s"

                        logger.info(f"Upload {upload_id} progress: {new_offset}/{upload_record['length']} bytes ({progress_percent:.1f}%) - "
                                  f"Speed: {format_speed(overall_speed)} (inst: {format_speed(instant_speed)}) - "
                                  f"ETA: {remaining_str}")

            return web.Response(
                status=204,
                headers={
                    'Upload-Offset': str(new_offset),
                    'Tus-Resumable': '1.0.0'
                }
            )

        except Exception as e:
            logger.error(f"Error processing upload {upload_id}: {e}")
            return web.Response(status=500, text="Internal server error")

    async def delete_upload(self, request):
        """Delete an upload (DELETE /files/{upload_id})"""
        upload_id = request.match_info['upload_id']

        # Check API key authentication
        auth = get_auth()
        if auth.api_key:  # Only check if API key is configured
            api_key = request.headers.get('X-API-Key')
            if not api_key or api_key != auth.api_key:
                logger.warning(f"Invalid or missing API key from {request.remote}")
                return web.Response(status=401, text="Invalid or missing API key")

        try:
            # Get upload record
            upload_record = self.db.get_upload(upload_id)
            if not upload_record:
                return web.Response(status=404, text="Upload not found")

            # Delete file
            file_path = Path(upload_record['file_path'])
            if file_path.exists():
                file_path.unlink()

            # Mark as cancelled in database
            self.db.update_upload_offset(upload_id, -1)  # Special value for cancelled

            # Clear progress tracking data
            if upload_id in self.active_uploads:
                del self.active_uploads[upload_id]

            logger.info(f"Deleted upload {upload_id}")
            return web.Response(status=204)

        except Exception as e:
            logger.error(f"Error deleting upload {upload_id}: {e}")
            return web.Response(status=500, text="Internal server error")

    async def handle_upload_completion(self, upload_id: str, upload_record: dict):
        """Handle upload completion by publishing message to queue"""
        try:
            file_path = upload_record['file_path']
            file_size = Path(file_path).stat().st_size

            logger.info(f"Upload {upload_id} completed. File: {file_path}, Size: {file_size}")

            # Parse metadata first to get original filename for decompression
            metadata_str = upload_record.get('metadata', '')
            metadata = {}
            original_filename = None
            if metadata_str:
                try:
                    metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                    original_filename = metadata.get('filename') if isinstance(metadata, dict) else None
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Invalid metadata for upload {upload_id}: {metadata_str}")

            # Check if file is compressed and decompress if needed
            file_path_obj = Path(file_path)
            temp_dir = file_path_obj.parent  # Use same directory as original file
            decompressed_file_path = decompress_file(file_path_obj, temp_dir, original_filename)

            # If file was decompressed, update file path and size
            uuid_filename = Path(file_path).name
            final_filename = original_filename if original_filename else uuid_filename
            if decompressed_file_path != file_path_obj:
                file_path = str(decompressed_file_path)
                file_size = Path(file_path).stat().st_size
                if original_filename:
                    final_filename = decompressed_file_path.name
                    logger.info(f"Decompressed {original_filename} to {final_filename}, Size: {file_size}")
                else:
                    logger.info(f"Decompressed file to {final_filename}, Size: {file_size}")

                # Remove the original compressed file to save disk space
                try:
                    file_path_obj.unlink()
                    logger.debug(f"Removed original compressed file: {file_path_obj}")
                except Exception as e:
                    logger.warning(f"Failed to remove original compressed file {file_path_obj}: {e}")

            # Import message queue here to avoid circular imports
            from message_queue import get_message_queue, QueueMessage

            # Get task_id from metadata
            task_id = metadata.get('task_id') if isinstance(metadata, dict) else None

            if not task_id:
                logger.error(f"No task_id found in metadata for upload {upload_id}")
                return

            # Create message queue instance
            mq = get_message_queue()

            # Publish upload completion message
            success = mq.publish_upload_completed(
                upload_id=upload_id,
                file_path=file_path,  # Use decompressed file path
                task_id=task_id
            )

            if success:
                logger.info(f"Published upload completion message for {upload_id}")
            else:
                logger.error(f"Failed to publish upload completion message for {upload_id}")

            # Always try to publish ASR processing request since upload_id == task_id
            try:
                # Get task metadata from tasks database
                from task_model import get_task_manager
                task_manager = get_task_manager()
                task_record = task_manager.get_task(task_id)
                if task_record:
                    # Update task status to uploaded and set audio file path
                    task_manager.update_audio_file(task_id, file_path)  # Use decompressed file path

                    # Publish ASR processing request
                    asr_success = mq.publish_asr_request(
                        task_id=task_id,
                        audio_file_path=file_path,  # Use decompressed file path
                        metadata={
                            'filename': final_filename,  # Use decompressed filename if applicable
                            'filesize': file_size,  # Use decompressed file size
                            'language': getattr(task_record, 'language', 'auto'),
                            'model': getattr(task_record, 'model', 'large-v3-turbo'),
                            'callback_url': task_record.callback_url
                        }
                    )

                    if asr_success:
                        logger.info(f"Published ASR processing request for task {task_id}")
                    else:
                        logger.error(f"Failed to publish ASR processing request for task {task_id}")
                else:
                    logger.warning(f"No task record found for task_id {task_id}")
            except Exception as e:
                logger.error(f"Error publishing ASR request for task {task_id}: {e}")

        except Exception as e:
            logger.error(f"Error handling upload completion for {upload_id}: {e}")

    async def notify_upload_completion(self, upload_id: str, file_path: str):
        """Legacy method - kept for backwards compatibility"""
        logger.info(f"Upload completion notification: {upload_id} -> {file_path}")

    async def health_check(self, request):
        """Health check endpoint"""
        # Basic health check - could be enhanced
        health_data = {
            "status": "healthy",
            "upload_dir": str(self.upload_dir),
            "max_file_size": self.max_file_size,
            "active_uploads": len(self.active_uploads)
        }
        return web.json_response(health_data)

    async def cleanup_old_uploads(self):
        """Cleanup old/incomplete uploads periodically"""
        while True:
            try:
                # Sleep for cleanup interval
                await asyncio.sleep(CLEANUP_INTERVAL)

                # Find old uploads to clean up
                cutoff_time = asyncio.get_event_loop().time() - CLEANUP_INTERVAL

                logger.info("Starting cleanup of old uploads")

                # This would need to be enhanced to check database for old uploads
                # and clean up their files

            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying

async def main():
    """Main function to start Tus server"""
    tus_server = TusServer()

    # Start cleanup task
    asyncio.create_task(tus_server.cleanup_old_uploads())

    logger.info(f"Starting Tus.io server on port {TUS_SERVER_PORT}")
    logger.info(f"Upload directory: {tus_server.upload_dir}")
    logger.info(f"Max file size: {tus_server.max_file_size // (1024*1024)}MB")
    logger.info(f"Progress logs enabled: {tus_server.show_progress_logs}")

    runner = web.AppRunner(
        tus_server.app,
        access_log=None,
        client_max_size=600*1024*1024  # 600MB limit
    )
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', TUS_SERVER_PORT)
    await site.start()

    try:
        # Keep the server running
        while True:
            await asyncio.sleep(1)
    finally:
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())