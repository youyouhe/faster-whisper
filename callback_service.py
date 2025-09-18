#!/usr/bin/env python3
"""
Callback Service
Handles HTTP callbacks for ASR task completions and failures
"""

import os
import time
import asyncio
import aiohttp
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

# Import components
from message_queue import get_message_queue, QueueMessage
from task_model import get_task_manager

logger = logging.getLogger(__name__)

@dataclass
class CallbackAttempt:
    """Represents a callback attempt"""
    task_id: str
    callback_url: str
    payload: Dict[str, Any]
    attempt: int
    max_attempts: int
    next_attempt_time: datetime
    error_message: Optional[str] = None

class CallbackService:
    """Service for handling HTTP callbacks"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.message_queue = get_message_queue(redis_url)
        self.task_manager = get_task_manager()
        self.running = False

        # Callback configuration
        self.max_attempts = int(os.getenv("CALLBACK_MAX_ATTEMPTS", "3"))
        self.retry_delay = int(os.getenv("CALLBACK_RETRY_DELAY", "10"))  # seconds
        self.timeout = int(os.getenv("CALLBACK_TIMEOUT", "30"))  # seconds
        self.concurrency_limit = int(os.getenv("CALLBACK_CONCURRENCY", "10"))

        # Semaphore for limiting concurrent callbacks
        self.semaphore = asyncio.Semaphore(self.concurrency_limit)

    async def start_service(self):
        """Start the callback service with async context"""
        self.running = True
        logger.info("Starting Callback Service")

        # Subscribe to events
        self.message_queue.subscribe_async(
            "tus:asr_completed",
            self.handle_asr_completed
        )

        self.message_queue.subscribe_async(
            "tus:asr_failed",
            self.handle_asr_failed
        )

        # Start the callback processing loop
        asyncio.create_task(self._process_callbacks())

        # Keep service running
        await self._run_service_loop()

        self.stop()

    async def _run_service_loop(self):
        """Run the service main loop"""
        try:
            # Use asyncio.create_task instead of run_forever to avoid event loop conflicts
            stop_event = asyncio.Event()

            def signal_handler():
                stop_event.set()

            # Setup signal handlers
            import signal
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGINT, signal_handler)
            loop.add_signal_handler(signal.SIGTERM, signal_handler)

            await stop_event.wait()

        except KeyboardInterrupt:
            logger.info("Callback Service interrupted")
        except Exception as e:
            logger.error(f"Callback service error: {e}")

    def stop(self):
        """Stop the callback service"""
        self.running = False
        self.message_queue.close()
        logger.info("Callback Service stopped")

    async def _process_callbacks(self):
        """Main loop for processing callbacks"""
        while self.running:
            try:
                await self._process_pending_callbacks()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in callback processing loop: {e}")
                await asyncio.sleep(1)

    async def _process_pending_callbacks(self):
        """Process tasks that need callbacks"""
        try:
            # Get completed tasks that haven't had their callback sent yet
            completed_tasks = []
            failed_tasks = []

            # In a real implementation, you might want to store pending callbacks
            # in a separate queue or database table

            # For now, we'll rely on the message queue events

        except Exception as e:
            logger.error(f"Error processing pending callbacks: {e}")

    def handle_asr_completed(self, message: QueueMessage):
        """Handle ASR completion event"""
        # Create a task to run the async handler
        import asyncio
        asyncio.create_task(self._handle_callback(message, "completed"))

    def handle_asr_failed(self, message: QueueMessage):
        """Handle ASR failure event"""
        # Create a task to run the async handler
        import asyncio
        asyncio.create_task(self._handle_callback(message, "failed"))

    async def _handle_callback(self, message: QueueMessage, status: str):
        """Handle callback for task completion or failure"""
        task_id = message.task_id

        try:
            # Get task details
            task = self.task_manager.get_task(task_id)
            if not task:
                logger.error(f"Task {task_id} not found for callback")
                return

            if not task.callback_url:
                logger.debug(f"Task {task_id} has no callback URL")
                return

            # Prepare callback payload
            payload = self._prepare_callback_payload(task, message, status)

            # Send callback
            await self._send_callback_with_retry(task_id, task.callback_url, payload)

        except Exception as e:
            logger.error(f"Error handling {status} callback for {task_id}: {e}")

    def _prepare_callback_payload(self, task, message: QueueMessage, status: str) -> Dict[str, Any]:
        """Prepare the callback payload based on status"""
        base_payload = {
            "task_id": task.task_id,
            "status": status,
            "filename": task.filename,
            "created_at": task.created_at.isoformat(),
        }

        if status == "completed":
            payload = {
                **base_payload,
                "srt_url": message.data.get("srt_url", f"/api/v1/tasks/{task.task_id}/download"),
                "processing_time": message.data.get("processing_time", 0),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        elif status == "failed":
            payload = {
                **base_payload,
                "error_message": message.data.get("error_message", "Unknown error"),
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
        else:
            payload = base_payload

        return payload

    async def _send_callback_with_retry(self, task_id: str, callback_url: str, payload: Dict[str, Any]):
        """Send callback with retry logic"""
        async with self.semaphore:
            attempt = 0

            while attempt < self.max_attempts:
                attempt += 1

                try:
                    success = await self._send_single_callback(callback_url, payload)

                    if success:
                        logger.info(f"Callback sent successfully for task {task_id} (attempt {attempt})")
                        return

                    if attempt < self.max_attempts:
                        delay = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                        logger.warning(f"Callback attempt {attempt} failed for {task_id}, retrying in {delay}s")
                        await asyncio.sleep(delay)

                except Exception as e:
                    logger.error(f"Callback attempt {attempt} error for {task_id}: {e}")
                    if attempt < self.max_attempts:
                        await asyncio.sleep(self.retry_delay)
                    else:
                        logger.error(f"All callback attempts failed for task {task_id}")

    async def _send_single_callback(self, callback_url: str, payload: Dict[str, Any]) -> bool:
        """Send a single callback request"""
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    callback_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "Tus-ASR-Callback-Service/1.0"
                    }
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        logger.warning(f"Callback returned status {response.status}: {await response.text()}")
                        return False

        except asyncio.TimeoutError:
            logger.warning(f"Callback timeout for URL: {callback_url}")
            return False
        except Exception as e:
            logger.error(f"Callback error for URL {callback_url}: {e}")
            return False

async def main():
    """Main function for running callback service"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    logger.info("Starting Callback Service")
    logger.info(f"Redis URL: {redis_url}")

    service = CallbackService(redis_url=redis_url)

    try:
        await service.start_service()
    except KeyboardInterrupt:
        logger.info("Callback Service stopped by user")
    except Exception as e:
        logger.error(f"Callback Service failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())