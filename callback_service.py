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

    def __init__(self, redis_url: str = "redis://redis:6379"):
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
        logger.info(f"DEBUG: Callback Service starting - redis_url={self.redis_url}")
        logger.info(f"DEBUG: Callback Service configuration - max_attempts={self.max_attempts}, retry_delay={self.retry_delay}s, timeout={self.timeout}s, concurrency_limit={self.concurrency_limit}")

        # Subscribe to events
        logger.info("DEBUG: Subscribing to tus:asr_completed events")
        self.message_queue.subscribe_async(
            "tus:asr_completed",
            self.handle_asr_completed
        )
        logger.info("DEBUG: Subscribing to tus:asr_failed events")
        self.message_queue.subscribe_async(
            "tus:asr_failed",
            self.handle_asr_failed
        )

        # Start the callback processing loop
        logger.info("DEBUG: Starting callback processing loop")
        asyncio.create_task(self._process_callbacks())

        # Keep service running
        logger.info("DEBUG: Callback Service started, entering main loop")
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
        logger.info(f"DEBUG: Received ASR completion event - task_id={message.task_id}")
        logger.info(f"DEBUG: ASR completed event data: {message.data}")
        # Run the async handler
        import asyncio
        try:
            # Try to get running loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new task in the running loop
                asyncio.create_task(self._handle_callback(message, "completed"))
            else:
                # Run the coroutine directly
                loop.run_until_complete(self._handle_callback(message, "completed"))
        except RuntimeError:
            # No event loop running, create a new one
            asyncio.run(self._handle_callback(message, "completed"))

    def handle_asr_failed(self, message: QueueMessage):
        """Handle ASR failure event"""
        logger.info(f"DEBUG: Received ASR failure event - task_id={message.task_id}")
        logger.info(f"DEBUG: ASR failed event data: {message.data}")
        # Run the async handler
        import asyncio
        try:
            # Try to get running loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new task in the running loop
                asyncio.create_task(self._handle_callback(message, "failed"))
            else:
                # Run the coroutine directly
                loop.run_until_complete(self._handle_callback(message, "failed"))
        except RuntimeError:
            # No event loop running, create a new one
            asyncio.run(self._handle_callback(message, "failed"))

    async def _handle_callback(self, message: QueueMessage, status: str):
        """Handle callback for task completion or failure"""
        task_id = message.task_id

        try:
            logger.info(f"Handling {status} callback for task {task_id}")
            logger.info(f"DEBUG: Starting callback process - task_id={task_id}, status={status}")
            logger.debug(f"Message data: {message.data}")

            # Get task details
            task = self.task_manager.get_task(task_id)
            if not task:
                logger.error(f"Task {task_id} not found for callback")
                return

            logger.info(f"DEBUG: Retrieved task details - task_id={task.task_id}, status={task.status}, callback_url={task.callback_url}")
            logger.info(f"DEBUG: Callback configuration - max_attempts={self.max_attempts}, retry_delay={self.retry_delay}s, timeout={self.timeout}s")

            if not task.callback_url:
                logger.info(f"DEBUG: Task {task_id} has no callback URL - skipping callback")
                logger.debug(f"Task {task_id} has no callback URL")
                return

            # Prepare callback payload
            payload = self._prepare_callback_payload(task, message, status)
            logger.info(f"DEBUG: Prepared callback payload for task {task_id} - payload_size={len(str(payload))} bytes")
            logger.debug(f"Prepared callback payload for task {task_id}: {payload}")

            # Send callback
            logger.info(f"DEBUG: Initiating callback for task {task_id} to URL: {task.callback_url}")
            logger.info(f"Sending callback for task {task_id} to {task.callback_url}")
            await self._send_callback_with_retry(task_id, task.callback_url, payload)

        except Exception as e:
            logger.error(f"Error handling {status} callback for {task_id}: {e}")
            logger.exception(e)  # Log full traceback

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
        logger.info(f"Starting callback retry process for task {task_id} to {callback_url}")
        logger.info(f"DEBUG: Callback retry initiated - task_id={task_id}, url={callback_url}")
        logger.debug(f"Callback payload: {payload}")
        logger.info(f"DEBUG: Retry configuration - max_attempts={self.max_attempts}, retry_delay={self.retry_delay}s, timeout={self.timeout}s")
        logger.debug(f"Retry configuration: max_attempts={self.max_attempts}, retry_delay={self.retry_delay}s, timeout={self.timeout}s")

        async with self.semaphore:
            attempt = 0
            total_attempts = self.max_attempts

            while attempt < total_attempts:
                attempt += 1
                logger.info(f"Callback attempt {attempt}/{total_attempts} for task {task_id}")
                logger.info(f"DEBUG: Attempt {attempt}/{total_attempts} - task_id={task_id}, url={callback_url}")

                try:
                    success = await self._send_single_callback(callback_url, payload)

                    if success:
                        logger.info(f"Callback sent successfully for task {task_id} (attempt {attempt})")
                        logger.info(f"DEBUG: Callback SUCCESS - task_id={task_id}, attempt={attempt}, url={callback_url}")
                        return True

                    if attempt < total_attempts:
                        delay = self.retry_delay * (2 ** (attempt - 1))  # Exponential backoff
                        logger.warning(f"Callback attempt {attempt} failed for {task_id}, retrying in {delay}s")
                        logger.warning(f"DEBUG: Callback FAILED, retrying - task_id={task_id}, attempt={attempt}, next_delay={delay}s")
                        await asyncio.sleep(delay)

                except Exception as e:
                    logger.error(f"Callback attempt {attempt} error for {task_id}: {e}")
                    logger.error(f"DEBUG: Callback EXCEPTION - task_id={task_id}, attempt={attempt}, error={str(e)}")
                    logger.exception(e)  # Log full traceback
                    if attempt < total_attempts:
                        await asyncio.sleep(self.retry_delay)
                    else:
                        logger.error(f"All callback attempts failed for task {task_id}")
                        logger.error(f"DEBUG: All callback attempts FAILED - task_id={task_id}, total_attempts={total_attempts}")
                        return False

            logger.error(f"All {total_attempts} callback attempts exhausted for task {task_id}")
            logger.error(f"DEBUG: Callback retry process COMPLETED WITH FAILURE - task_id={task_id}, url={callback_url}")
            return False

    async def _send_single_callback(self, callback_url: str, payload: Dict[str, Any]) -> bool:
        """Send a single callback request"""
        logger.info(f"Sending single callback request to {callback_url}")
        logger.info(f"DEBUG: Preparing HTTP request - url={callback_url}, method=POST, timeout={self.timeout}s")
        logger.debug(f"Request payload: {payload}")
        logger.debug(f"Request timeout: {self.timeout}s")

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            logger.debug(f"Created aiohttp client session with timeout {timeout.total}s")
            logger.info(f"DEBUG: Created HTTP session - timeout={timeout.total}s")

            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.debug("Created client session, sending POST request")
                logger.info(f"DEBUG: Sending POST request to {callback_url}")

                async with session.post(
                    callback_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "Tus-ASR-Callback-Service/1.0"
                    }
                ) as response:
                    logger.info(f"Callback response received: status={response.status}")
                    logger.info(f"DEBUG: HTTP response received - status={response.status}, url={callback_url}")

                    if response.status == 200:
                        logger.info(f"Callback successful for {callback_url}")
                        logger.info(f"DEBUG: Callback HTTP SUCCESS - status=200, url={callback_url}")

                        # Try to read response body for additional debugging
                        try:
                            response_text = await response.text()
                            if response_text:
                                logger.info(f"DEBUG: Response body - {response_text[:200]}...")
                        except Exception as e:
                            logger.debug(f"DEBUG: Could not read response body: {e}")

                        return True
                    else:
                        response_text = await response.text()
                        logger.warning(f"Callback returned status {response.status}: {response_text}")
                        logger.warning(f"DEBUG: Callback HTTP ERROR - status={response.status}, url={callback_url}, response={response_text[:200]}...")
                        return False

        except asyncio.TimeoutError:
            logger.warning(f"Callback timeout for URL: {callback_url} (timeout={self.timeout}s)")
            logger.warning(f"DEBUG: Callback TIMEOUT - url={callback_url}, timeout={self.timeout}s")
            return False
        except Exception as e:
            logger.error(f"Callback error for URL {callback_url}: {e}")
            logger.error(f"DEBUG: Callback EXCEPTION - url={callback_url}, error={str(e)}")
            logger.exception(e)  # Log full traceback
            return False

async def main():
    """Main function for running callback service"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")

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