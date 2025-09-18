#!/usr/bin/env python3
"""
Tus.io compatible API server for ASR task management and upload coordination
Implements proper task lifecycle with resumable uploads and callbacks
"""

import os
import uuid
from typing import Dict, Optional, Any
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn
import asyncio
import aiohttp
import logging
import json

# Import existing components
from task_model import get_task_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TUS_SERVER_BASE_URL = os.getenv("TUS_SERVER_BASE_URL", "http://localhost:1080/files")
CALLBACK_TIMEOUT = int(os.getenv("CALLBACK_TIMEOUT", "30"))
API_PORT = int(os.getenv("API_PORT", "8000"))

app = FastAPI(
    title="Tus.io ASR API Server",
    version="1.0.0",
    description="API server for managing ASR tasks with Tus.io resumable uploads"
)

class TaskRequest(BaseModel):
    """Request model for creating ASR tasks"""
    filename: str = Field(..., description="Name of the audio file")
    filesize: int = Field(..., gt=0, description="File size in bytes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="ASR processing metadata")
    callback_url: Optional[str] = Field(None, description="URL to call when processing completes")

class TaskResponse(BaseModel):
    """Response model for created ASR tasks"""
    task_id: str = Field(..., description="Unique task identifier")
    upload_url: str = Field(..., description="Tus upload URL for the file")
    status: str = Field(..., description="Task status")
    created_at: datetime

class TaskStatusResponse(BaseModel):
    """Response model for task status queries"""
    task_id: str
    status: str
    filename: str
    created_at: datetime
    updated_at: datetime
    upload_url: Optional[str] = None
    srt_file_path: Optional[str] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

def generate_tus_upload_url(task_id: str) -> str:
    """Generate a Tus upload URL for the given task"""
    upload_id = str(uuid.uuid4())
    return f"{TUS_SERVER_BASE_URL}/{upload_id}"

async def notify_callback(callback_url: str, task_data: dict):
    """Notify client via callback URL"""
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=CALLBACK_TIMEOUT)) as session:
            payload = {
                "task_id": task_data["task_id"],
                "status": task_data["status"],
                "filename": task_data["filename"],
                "srt_url": task_data.get("srt_url"),
                "completed_at": task_data.get("completed_at"),
                "error_message": task_data.get("error_message")
            }
            await session.post(callback_url, json=payload)
            logger.info(f"Callback sent to {callback_url} for task {task_data['task_id']}")
    except Exception as e:
        logger.error(f"Failed to send callback to {callback_url}: {e}")

@app.post("/api/v1/asr-tasks", response_model=TaskResponse)
async def create_asr_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a new ASR task with Tus resumable upload

    This endpoint creates a task record and returns a Tus upload URL
    where the client can upload their audio file with resumable capabilities.
    """
    try:
        # Get TaskManager instance (consistent with workers)
        task_manager = get_task_manager()

        # Extract metadata
        language = request.metadata.get('language', 'auto')
        model = request.metadata.get('model', 'large-v3-turbo')

        # Create the ASR task in database
        task_id = task_manager.create_task(
            filename=request.filename,
            filesize=request.filesize,
            language=language,
            model=model,
            callback_url=request.callback_url,
            task_metadata=request.metadata,
            upload_type="tus"
        )

        # Generate Tus upload URL and update task with it
        upload_url = generate_tus_upload_url(task_id)
        task_manager.update_upload_url(task_id, upload_url)

        logger.info(f"Created ASR task {task_id} for file {request.filename}")

        return TaskResponse(
            task_id=task_id,
            upload_url=upload_url,
            status="pending_upload",
            created_at=datetime.now(timezone.utc)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating ASR task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")

@app.get("/api/v1/asr-tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the current status of an ASR task

    Returns detailed information about the task including:
    - Current processing status
    - Upload URL (if still uploading)
    - SRT result URL (if completed)
    - Error messages (if failed)
    """
    try:
        task_manager = get_task_manager()
        task = task_manager.get_task(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Prepare response
        response = TaskStatusResponse(
            task_id=task.task_id,
            status=task.status,
            filename=task.filename,
            created_at=task.created_at,
            updated_at=task.updated_at,
            srt_file_path=task.srt_file_path,
            completed_at=task.completed_at,
            error_message=task.error_message,
            upload_url=task.upload_url
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve task: {str(e)}")

@app.get("/api/v1/tasks")
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 50
):
    """
    List ASR tasks with optional filtering by status

    Parameters:
    - status: Filter by task status (optional)
    - limit: Maximum number of tasks to return (default: 50)
    """
    try:
        task_manager = get_task_manager()

        if status:
            tasks = task_manager.get_tasks_by_status(status, limit)
        else:
            tasks = task_manager.get_recent_tasks(limit)

        # Convert TaskRecord objects to dictionaries
        task_list = []
        for task in tasks:
            task_dict = {
                "task_id": task.task_id,
                "status": task.status,
                "filename": task.filename,
                "filesize": task.filesize,
                "language": task.language,
                "model": task.model,
                "created_at": task.created_at,
                "updated_at": task.updated_at,
                "completed_at": task.completed_at,
                "srt_file_path": task.srt_file_path,
                "upload_url": task.upload_url,
                "error_message": task.error_message
            }
            task_list.append(task_dict)

        return {"tasks": task_list, "total": len(task_list)}

    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

@app.get("/api/v1/tasks/{task_id}/download")
async def download_srt(task_id: str):
    """
    Download the SRT file for a completed ASR task

    Returns the SRT subtitle file content as plain text.
    """
    try:
        task_manager = get_task_manager()
        task = task_manager.get_task(task_id)

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if task.status != "completed":
            raise HTTPException(status_code=404, detail=f"Task not completed yet. Status: {task.status}")

        if not task.srt_file_path:
            raise HTTPException(status_code=404, detail="No SRT file available for this task")

        # Check if file exists
        if not os.path.exists(task.srt_file_path):
            raise HTTPException(status_code=404, detail="SRT file not found on disk")

        # Return the SRT file content in API response format (matching load_balancer)
        with open(task.srt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return JSONResponse(content={
            "code": 0,
            "msg": "ok",
            "data": content
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading SRT for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download SRT file: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "tus-api-server"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Tus.io ASR API Server",
        "version": "1.0.0",
        "description": "API server for managing ASR tasks with Tus.io resumable uploads",
        "endpoints": {
            "POST /api/v1/asr-tasks": "Create new ASR task",
            "GET /api/v1/asr-tasks/{task_id}": "Get task status",
            "GET /api/v1/tasks": "List tasks",
            "GET /health": "Health check"
        }
    }

# Callback handling is now done by the separate CallbackService
# This API server focuses only on task creation

@app.on_event("startup")
async def startup_event():
    """App startup event"""
    logger.info("Tus.io compatible ASR API Server")
    logger.info(f"API listening on port {API_PORT}")
    logger.info(f"Tus server base URL: {TUS_SERVER_BASE_URL}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
