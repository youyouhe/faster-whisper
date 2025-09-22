#!/usr/bin/env python3
"""
ASR Worker Service
Consumes messages from the message queue and processes audio files to SRT
"""

import os
import time
import signal
import logging
import asyncio
import threading
import requests
import io
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile

# Import existing ASR components
from task_model import get_task_manager, TaskManager
from message_queue import get_message_queue, QueueMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))
SHUTDOWN_TIMEOUT = int(os.getenv("SHUTDOWN_TIMEOUT", "30"))
WORKER_POLL_INTERVAL = int(os.getenv("WORKER_POLL_INTERVAL", "5"))
SRT_STORAGE_DIR = os.getenv("SRT_STORAGE_DIR", "/data/srt_results")
LOAD_BALANCER_URL = os.getenv("LOAD_BALANCER_URL", "http://faster-whisper-dynamic:5001")

# Ensure SRT storage directory exists
Path(SRT_STORAGE_DIR).mkdir(parents=True, exist_ok=True)

class ASRWorker:
    """ASR worker that processes audio files to SRT"""

    def __init__(self, worker_id: int, redis_url: str = REDIS_URL, load_balancer_url: str = LOAD_BALANCER_URL):
        self.worker_id = worker_id
        self.redis_url = redis_url
        self.load_balancer_url = load_balancer_url
        self.message_queue = get_message_queue(redis_url)
        self.task_manager = get_task_manager()
        self.running = False
        self.processing_task = None

        # Worker statistics
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = time.time()

    def start(self):
        """Start the worker"""
        self.running = True
        self.processing_task = threading.Thread(
            target=self._process_loop,
            name=f"ASRWorker-{self.worker_id}",
            daemon=True
        )
        self.processing_task.start()
        logger.info(f"ASR Worker {self.worker_id} started")

    def stop(self):
        """Stop the worker"""
        self.running = False
        if self.processing_task:
            self.processing_task.join(timeout=SHUTDOWN_TIMEOUT)
            logger.info(f"ASR Worker {self.worker_id} stopped")

    def _process_loop(self):
        """Main processing loop"""
        logger.info(f"Starting ASR Worker {self.worker_id} processing loop")

        while self.running:
            try:
                # Check for new messages
                message = self.message_queue.consume_message("tus:asr_processing")

                if message:
                    logger.info(f"Worker {self.worker_id} received task {message.task_id}")
                    self._process_asr_task(message)
                else:
                    # No message available, wait before next poll
                    time.sleep(WORKER_POLL_INTERVAL)

            except Exception as e:
                logger.error(f"Error in worker {self.worker_id} processing loop: {e}")
                time.sleep(1)  # Brief pause before retry

    def _process_asr_task(self, message: QueueMessage):
        """Process a single ASR task by delegating to load balancer"""
        task_id = message.task_id
        data = message.data

        audio_file_path = data.get("audio_file_path")

        if not audio_file_path:
            logger.error(f"Worker {self.worker_id}: No audio file path in task {task_id}")
            self._fail_task(task_id, "No audio file path provided")
            return

        # Check if audio file exists
        if not Path(audio_file_path).exists():
            logger.error(f"Worker {self.worker_id}: Audio file not found: {audio_file_path}")
            self._fail_task(task_id, f"Audio file not found: {audio_file_path}")
            return

        # Update task status to processing
        success = self.task_manager.start_processing(task_id)
        if not success:
            logger.error(f"Worker {self.worker_id}: Failed to update task {task_id} status to processing")
            return

        try:
            logger.info(f"Worker {self.worker_id}: Starting ASR processing for task {task_id}")

            # Get file size for logging
            file_size = Path(audio_file_path).stat().st_size
            logger.info(f"Worker {self.worker_id}: Processing {audio_file_path} ({file_size / (1024*1024):.2f}MB)")

            # Record processing start time
            processing_start_time = time.time()

            # Read audio file data
            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()

            logger.info(f"Worker {self.worker_id}: Read {len(audio_data)} bytes of audio data")

            # Send to load balancer for processing
            result = self._call_load_balancer(audio_data)

            if result and result.get("code") == 0 and result.get("data"):
                srt_content = result["data"]
                processing_time = time.time() - processing_start_time

                logger.info(f"Worker {self.worker_id}: Received SRT content ({len(srt_content)} chars)")

                # Generate SRT filename
                srt_filename = f"{task_id}.srt"
                srt_file_path = Path(SRT_STORAGE_DIR) / srt_filename

                # Save SRT file
                with open(srt_file_path, 'w', encoding='utf-8') as f:
                    f.write(srt_content)

                # Update task with SRT file path
                self.task_manager.update_audio_file(task_id, str(audio_file_path))

                # Complete the task
                success = self.task_manager.complete_task(
                    task_id=task_id,
                    srt_file_path=str(srt_file_path),
                    processing_time=processing_time
                )

                if success:
                    # Publish completion message
                    srt_url = f"/api/v1/tasks/{task_id}/download"
                    self.message_queue.publish_asr_completed(
                        task_id=task_id,
                        srt_file_path=str(srt_file_path),
                        processing_time=processing_time,
                        srt_url=srt_url
                    )

                    self.processed_count += 1
                    logger.info(
                        f"Worker {self.worker_id}: Task {task_id} completed successfully in {processing_time:.1f}s"
                    )
                else:
                    self._fail_task(task_id, "Failed to update task completion status")
            else:
                error_msg = f"Invalid response from load balancer: {result}"
                logger.error(f"Worker {self.worker_id}: {error_msg}")
                self._fail_task(task_id, error_msg)

        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error processing task {task_id}: {e}")
            self._fail_task(task_id, f"ASR processing error: {str(e)}")

        finally:
            # Clean up: delete the audio file after processing
            try:
                if audio_file_path and Path(audio_file_path).exists():
                    Path(audio_file_path).unlink()
                    logger.debug(f"Worker {self.worker_id}: Cleaned up audio file {audio_file_path}")
            except Exception as e:
                logger.warning(f"Worker {self.worker_id}: Failed to cleanup audio file {audio_file_path}: {e}")

    def _call_load_balancer(self, audio_data: bytes) -> dict:
        """Call load balancer to process audio data"""
        try:
            import json

            logger.info(f"Worker {self.worker_id}: Calling load balancer at {self.load_balancer_url}/inference")

            url = f"{self.load_balancer_url}/inference"

            # Create multipart form data
            files = {
                'file': ('audio.wav', io.BytesIO(audio_data), 'audio/wav')
            }

            data = {
                'task': 'asr',
                'language': 'auto',
                'model': 'large-v3-turbo'
            }

            # Set timeout based on file size (similar to load balancer logic)
            file_size_mb = len(audio_data) / (1024 * 1024)
            if file_size_mb < 10:
                timeout_seconds = 1800  # 30 minutes
            elif file_size_mb < 50:
                timeout_seconds = 2700  # 45 minutes
            else:
                timeout_seconds = 3600  # 60 minutes

            logger.info(f"Worker {self.worker_id}: Using {timeout_seconds}s timeout for {file_size_mb:.1f}MB file")

            response = requests.post(
                url,
                files=files,
                data=data,
                timeout=timeout_seconds
            )

            response.raise_for_status()

            result = response.json()
            logger.info(f"Worker {self.worker_id}: Load balancer returned: code={result.get('code')}, has_data={'data' in result}")

            return result

        except requests.exceptions.Timeout as e:
            logger.error(f"Worker {self.worker_id}: Timeout calling load balancer: {e}")
            raise Exception(f"Timeout: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Worker {self.worker_id}: HTTP error calling load balancer: {e}")
            raise Exception(f"HTTP error: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Worker {self.worker_id}: JSON decode error from load balancer: {e}")
            raise Exception(f"Invalid JSON response: {e}")
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Unexpected error calling load balancer: {e}")
            raise Exception(f"Unexpected error: {e}")

    def _fail_task(self, task_id: str, error_message: str):
        """Mark task as failed and publish failure message"""
        try:
            logger.info(f"Worker {self.worker_id}: Failing task {task_id} with error: {error_message}")

            # Update task status
            success = self.task_manager.fail_task(task_id, error_message)

            if success:
                logger.info(f"Worker {self.worker_id}: Successfully updated task {task_id} status to failed")
                # Publish failure message
                logger.info(f"Worker {self.worker_id}: Publishing ASR failed message for task {task_id}")
                publish_success = self.message_queue.publish_asr_failed(task_id, error_message)
                if publish_success:
                    logger.info(f"Worker {self.worker_id}: Successfully published ASR failed message for task {task_id}")
                else:
                    logger.error(f"Worker {self.worker_id}: Failed to publish ASR failed message for task {task_id}")

                self.failed_count += 1
                logger.error(f"Worker {self.worker_id}: Task {task_id} failed: {error_message}")
            else:
                logger.error(f"Worker {self.worker_id}: Failed to update task {task_id} failure status")
                # Try to get task details for debugging
                try:
                    task = self.task_manager.get_task(task_id)
                    if task:
                        logger.debug(f"Worker {self.worker_id}: Task {task_id} details: status={task.status}, callback_url={task.callback_url}")
                    else:
                        logger.error(f"Worker {self.worker_id}: Task {task_id} not found in database")
                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: Error getting task details for {task_id}: {e}")

        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Error failing task {task_id}: {e}")
            logger.exception(e)  # Log full traceback

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics"""
        uptime = time.time() - self.start_time
        return {
            "worker_id": self.worker_id,
            "processed_count": self.processed_count,
            "failed_count": self.failed_count,
            "uptime_seconds": uptime,
            "running": self.running
        }

class ASRWorkerPool:
    """Pool of ASR workers"""

    def __init__(self, num_workers: int = MAX_WORKERS, redis_url: str = REDIS_URL, load_balancer_url: str = LOAD_BALANCER_URL):
        self.num_workers = num_workers
        self.redis_url = redis_url
        self.load_balancer_url = load_balancer_url
        self.workers = []
        self.message_queue = None
        self.running = False

        # Signal handling
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down worker pool...")
        self.stop()

    def start(self):
        """Start the worker pool"""
        logger.info(f"Starting ASR Worker Pool with {self.num_workers} workers")

        self.running = True
        self.message_queue = get_message_queue(self.redis_url)

        # Create and start workers
        for i in range(self.num_workers):
            worker = ASRWorker(i, self.redis_url, self.load_balancer_url)
            self.workers.append(worker)
            worker.start()

        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
                # Could add health checks here
        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        self.stop()

    def stop(self):
        """Stop the worker pool"""
        if not self.running:
            return

        logger.info("Stopping ASR Worker Pool...")
        self.running = False

        # Stop all workers
        for worker in self.workers:
            worker.stop()

        # Close message queue
        if self.message_queue:
            self.message_queue.close()

        logger.info("ASR Worker Pool stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        worker_stats = [worker.get_stats() for worker in self.workers]

        total_processed = sum(stat["processed_count"] for stat in worker_stats)
        total_failed = sum(stat["failed_count"] for stat in worker_stats)

        return {
            "num_workers": self.num_workers,
            "total_processed": total_processed,
            "total_failed": total_failed,
            "running": self.running,
            "worker_stats": worker_stats
        }

async def health_check_server():
    """Simple health check server for monitoring"""
    from aiohttp import web

    async def health_handler(request):
        pool_stats = pool.get_stats()
        return web.json_response({
            "status": "healthy" if pool_stats["running"] else "stopped",
            "workers": pool_stats
        })

    app = web.Application()
    app.router.add_get("/health", health_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', 8081)
    await site.start()
    logger.info("ASR Worker health check server started on port 8081")

def main():
    """Main function"""
    # Configuration from environment
    num_workers = int(os.getenv("MAX_WORKERS", "3"))
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    load_balancer_url = os.getenv("LOAD_BALANCER_URL", "http://faster-whisper-dynamic:5001")

    logger.info("Starting ASR Worker Service")
    logger.info(f"Redis URL: {redis_url}")
    logger.info(f"Load Balancer URL: {load_balancer_url}")
    logger.info(f"Number of workers: {num_workers}")
    logger.info(f"SRT storage directory: {SRT_STORAGE_DIR}")

    # Create worker pool
    global pool
    pool = ASRWorkerPool(num_workers=num_workers, redis_url=redis_url, load_balancer_url=load_balancer_url)

    # Start health check server in background thread
    health_thread = threading.Thread(target=lambda: asyncio.run(health_check_server()))
    health_thread.daemon = True
    health_thread.start()

    # Start worker pool
    try:
        pool.start()
    except Exception as e:
        logger.error(f"Error starting ASR Worker Pool: {e}")
        pool.stop()
        raise

if __name__ == "__main__":
    main()