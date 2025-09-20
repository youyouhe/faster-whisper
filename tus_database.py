#!/usr/bin/env python3
"""
SQLite database manager for Tus uploads and ASR tasks
Standalone module to avoid circular dependencies
"""

import os
import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
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