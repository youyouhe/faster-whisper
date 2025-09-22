#!/usr/bin/env python3
"""
Task model and database schema for faster-whisper ASR service
Manages ASR task states with SQLite persistence
"""

import sqlite3
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import threading
import os


@dataclass
class TaskRecord:
    """ASR task record for database storage"""

    task_id: str
    status: str  # pending_upload, uploading, processing, completed, failed, cancelled
    filename: str
    filesize: int
    language: str = "auto"
    model: str = "large-v3-turbo"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    callback_url: Optional[str] = None
    upload_url: Optional[str] = None  # Tus upload URL
    srt_file_path: Optional[str] = None
    audio_file_path: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None  # seconds
    task_metadata: Dict[str, Any] = field(default_factory=dict)

    def _asdict(self):
        """Convert to dictionary for JSON serialization"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class TaskManager:
    """Thread-safe SQLite-based task manager"""

    _lock = threading.Lock()

    def __init__(self, db_path: str = "tasks.db"):
        """Initialize task manager with SQLite database"""
        self.db_path = db_path
        self._ensure_database()

    def _ensure_database(self):
        """Create database tables if they don't exist"""
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    filesize INTEGER NOT NULL,
                    language TEXT NOT NULL DEFAULT 'auto',
                    model TEXT NOT NULL DEFAULT 'large-v3-turbo',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    callback_url TEXT,
                    upload_url TEXT,
                    srt_file_path TEXT,
                    audio_file_path TEXT,
                    error_message TEXT,
                    processing_time REAL,
                    task_metadata TEXT
                )
            ''')

            # Index for common queries
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON tasks(created_at)')
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-safe database connection"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _from_db_row(self, row) -> TaskRecord:
        """Convert database row to TaskRecord"""
        task_dict = dict(row)
        task_dict['created_at'] = datetime.fromisoformat(task_dict['created_at'])
        task_dict['updated_at'] = datetime.fromisoformat(task_dict['updated_at'])

        if task_dict.get('completed_at'):
            task_dict['completed_at'] = datetime.fromisoformat(task_dict['completed_at'])

        if task_dict.get('task_metadata'):
            task_dict['task_metadata'] = json.loads(task_dict['task_metadata'])

        return TaskRecord(**task_dict)

    def create_task(
        self,
        filename: str,
        filesize: int,
        language: str = "auto",
        model: str = "large-v3-turbo",
        callback_url: Optional[str] = None,
        task_metadata: Optional[Dict[str, Any]] = None,
        upload_type: str = "direct"  # 'direct' or 'tus'
    ) -> str:
        """Create a new ASR task and return task_id"""

        import uuid
        task_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        status = "pending_upload" if upload_type == "tus" else "pending"

        task_record = TaskRecord(
            task_id=task_id,
            status=status,
            filename=filename,
            filesize=filesize,
            language=language,
            model=model,
            created_at=created_at,
            updated_at=created_at,
            callback_url=callback_url,
            task_metadata=task_metadata or {}
        )

        with self._lock, self._get_connection() as conn:
            conn.execute('''
                INSERT INTO tasks (
                    task_id, status, filename, filesize, language, model,
                    created_at, updated_at, callback_url, task_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task_record.task_id,
                task_record.status,
                task_record.filename,
                task_record.filesize,
                task_record.language,
                task_record.model,
                task_record.created_at.isoformat(),
                task_record.updated_at.isoformat(),
                task_record.callback_url,
                json.dumps(task_record.task_metadata) if task_record.task_metadata else None
            ))
            conn.commit()

        return task_id

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """Get task by task_id"""
        with self._get_connection() as conn:
            row = conn.execute(
                'SELECT * FROM tasks WHERE task_id = ?',
                (task_id,)
            ).fetchone()

            return self._from_db_row(row) if row else None

    def update_task_status(
        self,
        task_id: str,
        status: str,
        **updates
    ) -> bool:
        """Update task status and other fields"""

        update_fields = ['status', 'updated_at'] + list(updates.keys())
        update_values = [status, datetime.now(timezone.utc).isoformat()]

        # Handle special datetime fields
        for key, value in updates.items():
            if isinstance(value, datetime):
                update_values.append(value.isoformat())
            elif isinstance(value, dict):
                update_values.append(json.dumps(value) if value else None)
            else:
                update_values.append(value)

        query = f"UPDATE tasks SET {', '.join(f'{field} = ?' for field in update_fields)} WHERE task_id = ?"
        update_values.append(task_id)

        with self._lock, self._get_connection() as conn:
            try:
                # Add debug logging
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"[DEBUG] Executing query: {query}")
                logger.info(f"[DEBUG] Query values: {update_values}")
                cursor = conn.execute(query, update_values)
                success = cursor.rowcount > 0
                logger.info(f"[DEBUG] Update result - success: {success}, cursor.rowcount: {cursor.rowcount}")
                if success:
                    conn.commit()
                    logger.info(f"[DEBUG] Successfully updated task {task_id}")
                else:
                    # Check if task exists
                    task_exists = conn.execute('SELECT 1 FROM tasks WHERE task_id = ?', (task_id,)).fetchone()
                    logger.error(f"[DEBUG] No rows updated for task {task_id}. Task exists: {task_exists is not None}")
                return success
            except Exception as e:
                logger.error(f"[DEBUG] Database error updating task {task_id}: {e}")
                conn.rollback()
                return False

    def update_upload_url(self, task_id: str, upload_url: str) -> bool:
        """Update the Tus upload URL for a task"""
        return self.update_task_status(task_id, "pending_upload", upload_url=upload_url)

    def start_processing(self, task_id: str) -> bool:
        """Mark task as being processed"""
        return self.update_task_status(task_id, "processing")

    def complete_task(
        self,
        task_id: str,
        srt_file_path: Optional[str] = None,
        processing_time: Optional[float] = None
    ) -> bool:
        """Mark task as completed"""
        completed_at = datetime.now(timezone.utc)
        success = self.update_task_status(
            task_id,
            "completed",
            completed_at=completed_at,
            srt_file_path=srt_file_path,
            processing_time=processing_time
        )

        # Trigger callback if configured
        task = self.get_task(task_id)
        if task and task.callback_url:
            self._trigger_callback_async(task)

        return success

    def fail_task(self, task_id: str, error_message: str) -> bool:
        """Mark task as failed with error message"""
        return self.update_task_status(task_id, "failed", error_message=error_message)

    def update_audio_file(self, task_id: str, audio_path: str) -> bool:
        """Update the audio file path after upload"""
        return self.update_task_status(task_id, "uploaded", audio_file_path=audio_path)

    def get_tasks_by_status(self, status: str, limit: int = 100) -> List[TaskRecord]:
        """Get tasks by status, ordered by creation time"""
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM tasks
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (status, limit)).fetchall()

            return [self._from_db_row(row) for row in rows]

    def get_recent_tasks(self, limit: int = 50) -> List[TaskRecord]:
        """Get recently created tasks"""
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM tasks
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,)).fetchall()

            return [self._from_db_row(row) for row in rows]

    def cleanup_old_tasks(self, hours_old: int = 24) -> int:
        """Clean up completed tasks older than specified hours"""
        cutoff_time = datetime.now(timezone.utc)
        cutoff_time = cutoff_time.replace(hour=cutoff_time.hour - hours_old)

        with self._lock, self._get_connection() as conn:
            cursor = conn.execute('''
                DELETE FROM tasks
                WHERE status IN ('completed', 'failed')
                AND completed_at < ?
            ''', (cutoff_time.isoformat(),))

            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count

    def _trigger_callback_async(self, task: TaskRecord):
        """Trigger callback asynchronously"""
        import asyncio
        import aiohttp
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"[DEBUG] Triggering async callback for task {task.task_id}")
        logger.debug(f"[DEBUG] Task details: task_id={task.task_id}, status={task.status}, callback_url={task.callback_url}")

        async def _callback():
            try:
                logger.info(f"[DEBUG] Starting async callback for task {task.task_id} to {task.callback_url}")
                logger.info(f"[DEBUG] Task details: task_id={task.task_id}, status={task.status}, callback_url={task.callback_url}")

                async with aiohttp.ClientSession() as session:
                    payload = {
                        "task_id": task.task_id,
                        "status": task.status,
                        "filename": task.filename,
                        "srt_url": f"/api/v1/tasks/{task.task_id}/download" if task.srt_file_path else None
                    }

                    logger.debug(f"[DEBUG] Callback payload: {payload}")
                    logger.info(f"[DEBUG] Sending POST request to {task.callback_url}")

                    response = await session.post(
                        task.callback_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30),
                        headers={
                            "Content-Type": "application/json",
                            "User-Agent": "Tus-ASR-Task-Manager/1.0"
                        }
                    )

                    logger.info(f"[DEBUG] Callback response for task {task.task_id}: status={response.status}")
                    if response.status != 200:
                        response_text = await response.text()
                        logger.warning(f"[DEBUG] Callback for task {task.task_id} returned status {response.status}: {response_text}")
                    else:
                        logger.info(f"[DEBUG] Callback for task {task.task_id} successful")

            except asyncio.TimeoutError as e:
                logger.error(f"[DEBUG] Callback timeout for task {task.task_id}: {e}")
                logger.info(f"[DEBUG] Timeout details: URL={task.callback_url}, timeout=30s")
            except Exception as e:
                logger.error(f"[DEBUG] Callback failed for task {task.task_id}: {e}")
                logger.exception(e)  # Log full traceback

        # Run in background thread pool
        import threading
        logger.info(f"[DEBUG] Starting callback thread for task {task.task_id}")
        threading.Thread(target=lambda: asyncio.run(_callback()), name=f"CallbackThread-{task.task_id}").start()


# Global task manager instance
_task_manager_instance = None
_task_manager_lock = threading.Lock()

def get_task_manager() -> TaskManager:
    """Get singleton task manager instance"""
    global _task_manager_instance

    if _task_manager_instance is None:
        with _task_manager_lock:
            if _task_manager_instance is None:
                # Ensure data directory exists
                data_dir = os.getenv("DATA_DIR", "./data")
                Path(data_dir).mkdir(exist_ok=True)
                db_path = os.path.join(data_dir, "tasks.db")
                _task_manager_instance = TaskManager(db_path)

    return _task_manager_instance