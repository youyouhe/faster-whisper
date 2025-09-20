#!/usr/bin/env python3
"""
Redis-based message queue for Tus.io ASR system
Handles communication between Tus server, API server, and ASR workers
"""

import redis
import json
import time
from typing import Dict, Any, Optional, Callable
import asyncio
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import threading

logger = logging.getLogger(__name__)

@dataclass
class QueueMessage:
    """Message structure for queue communication"""
    message_id: str
    message_type: str
    task_id: str
    data: Dict[str, Any]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type,
            'task_id': self.task_id,
            'data': self.data,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueueMessage':
        """Create from dictionary"""
        return cls(
            message_id=data['message_id'],
            message_type=data['message_type'],
            task_id=data['task_id'],
            data=data['data'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

class MessageQueue:
    """Redis-based message queue handler"""

    def __init__(self, redis_url: str = "redis://redis:6379"):
        """
        Initialize message queue

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis_client = None
        self._connect()

        # Queue names
        self.upload_completed_queue = "tus:upload_completed"
        self.asr_processing_queue = "tus:asr_processing"
        self.asr_completed_queue = "tus:asr_completed"
        self.asr_failed_queue = "tus:asr_failed"

        # Subscribers registry
        self.subscribers = {}
        self._running = False

    def _connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        return f"msg_{int(time.time() * 1000000)}"

    def publish_upload_completed(self, upload_id: str, file_path: str, task_id: Optional[str] = None) -> bool:
        """
        Publish upload completion message

        Args:
            upload_id: Tus upload ID
            file_path: Path to uploaded file
            task_id: Associated ASR task ID (if available)
        """
        message = QueueMessage(
            message_id=self._generate_message_id(),
            message_type="upload_completed",
            task_id=task_id or upload_id,
            data={
                "upload_id": upload_id,
                "file_path": file_path,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return self._publish(self.upload_completed_queue, message)

    def publish_asr_request(self, task_id: str, audio_file_path: str, metadata: Dict[str, Any]) -> bool:
        """
        Publish ASR processing request

        Args:
            task_id: ASR task ID
            audio_file_path: Path to audio file
            metadata: ASR processing metadata
        """
        message = QueueMessage(
            message_id=self._generate_message_id(),
            message_type="asr_process",
            task_id=task_id,
            data={
                "audio_file_path": audio_file_path,
                "language": metadata.get("language", "auto"),
                "model": metadata.get("model", "large-v3-turbo"),
                "callback_url": metadata.get("callback_url"),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return self._publish(self.asr_processing_queue, message)

    def publish_asr_completed(self, task_id: str, srt_file_path: str, processing_time: float,
                            srt_url: Optional[str] = None) -> bool:
        """
        Publish ASR completion message

        Args:
            task_id: ASR task ID
            srt_file_path: Path to generated SRT file
            processing_time: Time taken for processing
            srt_url: Public URL to access SRT file
        """
        message = QueueMessage(
            message_id=self._generate_message_id(),
            message_type="asr_completed",
            task_id=task_id,
            data={
                "srt_file_path": srt_file_path,
                "srt_url": srt_url,
                "processing_time": processing_time,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return self._publish(self.asr_completed_queue, message)

    def publish_asr_failed(self, task_id: str, error_message: str) -> bool:
        """
        Publish ASR failure message

        Args:
            task_id: ASR task ID
            error_message: Error description
        """
        message = QueueMessage(
            message_id=self._generate_message_id(),
            message_type="asr_failed",
            task_id=task_id,
            data={
                "error_message": error_message,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

        return self._publish(self.asr_failed_queue, message)

    def _publish(self, queue_name: str, message: QueueMessage) -> bool:
        """Publish message to queue"""
        try:
            message_json = json.dumps(message.to_dict())
            self.redis_client.lpush(queue_name, message_json)

            # Also publish to pub/sub for immediate notification
            self.redis_client.publish(queue_name, message_json)

            logger.info(f"Published {message.message_type} message to {queue_name}: {message.task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish message to {queue_name}: {e}")
            return False

    def subscribe_async(self, queue_name: str, callback: Callable[[QueueMessage], None]):
        """
        Subscribe to queue asynchronously

        Args:
            queue_name: Name of queue to subscribe to
            callback: Function to call when message received
        """
        if queue_name not in self.subscribers:
            self.subscribers[queue_name] = []

        self.subscribers[queue_name].append(callback)

        # Start listening if not already running
        if not self._running:
            self._running = True
            threading.Thread(
                target=self._listen_worker,
                args=(queue_name,),
                daemon=True
            ).start()

    def _listen_worker(self, queue_name: str):
        """Worker thread to listen for messages"""
        logger.info(f"Starting listener for queue: {queue_name}")

        pubsub = self.redis_client.pubsub()
        pubsub.subscribe(queue_name)

        try:
            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        queue_message = QueueMessage.from_dict(data)

                        # Call all registered callbacks
                        for callback in self.subscribers.get(queue_name, []):
                            try:
                                callback(queue_message)
                            except Exception as e:
                                logger.error(f"Error in callback for {queue_name}: {e}")

                    except Exception as e:
                        logger.error(f"Error processing message from {queue_name}: {e}")

        except Exception as e:
            logger.error(f"Error in listener for {queue_name}: {e}")

    def get_message_count(self, queue_name: str) -> int:
        """Get number of messages in queue"""
        try:
            return self.redis_client.llen(queue_name)
        except Exception as e:
            logger.error(f"Error getting message count for {queue_name}: {e}")
            return 0

    def get_pending_messages(self, queue_name: str, limit: int = 10) -> list:
        """Get pending messages from queue without consuming them"""
        try:
            messages = []
            for i in range(min(limit, self.redis_client.llen(queue_name))):
                message_json = self.redis_client.lindex(queue_name, i)
                if message_json:
                    data = json.loads(message_json)
                    messages.append(QueueMessage.from_dict(data))
            return messages
        except Exception as e:
            logger.error(f"Error getting pending messages from {queue_name}: {e}")
            return []

    def consume_message(self, queue_name: str) -> Optional[QueueMessage]:
        """Consume (pop) a message from the queue"""
        try:
            message_json = self.redis_client.rpop(queue_name)
            if message_json:
                data = json.loads(message_json)
                return QueueMessage.from_dict(data)
            return None
        except Exception as e:
            logger.error(f"Error consuming message from {queue_name}: {e}")
            return None

    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()
            logger.info("Redis connection closed")

# Global message queue instance
_message_queue_instance = None
_message_queue_lock = threading.Lock()

def get_message_queue(redis_url: str = "redis://redis:6379") -> MessageQueue:
    """Get singleton message queue instance"""
    global _message_queue_instance

    if _message_queue_instance is None:
        with _message_queue_lock:
            if _message_queue_instance is None:
                _message_queue_instance = MessageQueue(redis_url)
    else:
        # If instance exists but with different URL, recreate it
        if _message_queue_instance.redis_url != redis_url:
            _message_queue_instance = MessageQueue(redis_url)

    return _message_queue_instance