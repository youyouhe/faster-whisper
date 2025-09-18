#!/usr/bin/env python3
"""
Event Handler Service
Handles events from message queue and coordinates between Tus server and ASR workers
"""

import os
import time
import logging
from typing import Dict, Any

# Import components
from message_queue import get_message_queue, QueueMessage
from task_model import get_task_manager
from tus_api_server import callback_handler

logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))

class EventHandler:
    """Handles events from the message queue"""

    def __init__(self, redis_url: str = REDIS_URL):
        self.redis_url = redis_url
        self.message_queue = get_message_queue(redis_url)
        self.task_manager = get_task_manager()
        self.running = False

    def start(self):
        """Start the event handler"""
        self.running = True
        logger.info("Starting Event Handler Service")

        # Subscribe to upload completion events
        self.message_queue.subscribe_async(
            "tus:upload_completed",
            self.handle_upload_completed
        )

        # Subscribe to ASR completion events
        self.message_queue.subscribe_async(
            "tus:asr_completed",
            self.handle_asr_completed
        )

        # Subscribe to ASR failure events
        self.message_queue.subscribe_async(
            "tus:asr_failed",
            self.handle_asr_failed
        )

        # Keep service running
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Event Handler interrupted")

        self.stop()

    def stop(self):
        """Stop the event handler"""
        self.running = False
        self.message_queue.close()
        logger.info("Event Handler Service stopped")

    async def handle_upload_completed(self, message: QueueMessage):
        """Handle upload completion event"""
        try:
            logger.info(f"Processing upload completion for task {message.task_id}")

            # Extract data from message
            upload_id = message.data.get("upload_id")
            file_path = message.data.get("file_path")

            if not upload_id or not file_path:
                logger.error(f"Incomplete upload completion message: {message.data}")
                return

            # Decode task_id if it's base64 encoded
            task_id = message.task_id
            try:
                import base64
                decoded_task_id = base64.b64decode(message.task_id).decode('utf-8')
                logger.info(f"Decoded task_id from base64: {message.task_id} -> {decoded_task_id}")
                task_id = decoded_task_id
            except Exception as e:
                logger.info(f"Task_id not base64 encoded (using as-is): {message.task_id}")

            # Check if this upload is associated with an ASR task
            # We can determine this by checking if the task exists in our task manager
            task = self.task_manager.get_task(task_id)

            if task:
                # This is an ASR task - trigger ASR processing
                logger.info(f"Triggering ASR processing for task {task_id}")

                # Publish ASR processing request
                metadata = {
                    "language": task.language or "auto",
                    "model": task.model or "large-v3-turbo",
                    "callback_url": task.callback_url
                }

                success = self.message_queue.publish_asr_request(
                    task_id=task_id,
                    audio_file_path=file_path,
                    metadata=metadata
                )

                if success:
                    logger.info(f"Published ASR request for task {message.task_id}")
                else:
                    logger.error(f"Failed to publish ASR request for task {message.task_id}")

            else:
                # This might be a standalone upload without ASR task
                logger.info(f"Upload {upload_id} completed, but no ASR task found for {message.task_id}")

        except Exception as e:
            logger.error(f"Error handling upload completion for {message.task_id}: {e}")

    async def handle_asr_completed(self, message: QueueMessage):
        """Handle ASR completion event"""
        try:
            logger.info(f"Processing ASR completion for task {message.task_id}")

            task_id = message.task_id
            srt_file_path = message.data.get("srt_file_path")
            srt_url = message.data.get("srt_url")
            processing_time = message.data.get("processing_time", 0)

            # Update task with completion information
            success = self.task_manager.complete_task(
                task_id=task_id,
                srt_file_path=srt_file_path,
                processing_time=processing_time
            )

            if success:
                logger.info(f"Updated task {task_id} with completion status")
            else:
                logger.error(f"Failed to update completion status for task {task_id}")

        except Exception as e:
            logger.error(f"Error handling ASR completion for {message.task_id}: {e}")

    async def handle_asr_failed(self, message: QueueMessage):
        """Handle ASR failure event"""
        try:
            logger.info(f"Processing ASR failure for task {message.task_id}")

            task_id = message.task_id
            error_message = message.data.get("error_message", "Unknown error")

            # Update task with failure information
            success = self.task_manager.fail_task(task_id, error_message)

            if success:
                logger.info(f"Updated task {task_id} with failure status")
            else:
                logger.error(f"Failed to update failure status for task {task_id}")

        except Exception as e:
            logger.error(f"Error handling ASR failure for {message.task_id}: {e}")

def run_event_handler():
    """Run the event handler with proper error handling"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Event Handler Service starting...")
    logger.info(f"Redis URL: {REDIS_URL}")
    logger.info(f"Poll interval: {POLL_INTERVAL}s")

    try:
        handler = EventHandler(redis_url=REDIS_URL)
        handler.start()
    except KeyboardInterrupt:
        logger.info("Event Handler stopped by user")
    except Exception as e:
        logger.error(f"Event Handler failed: {e}")
        raise

if __name__ == "__main__":
    run_event_handler()