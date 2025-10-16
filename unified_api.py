#!/usr/bin/env python3
"""
统一API入口 - 单端口HTTP服务接口
Unified API Gateway - Single-Port HTTP Service Interface
"""

import argparse
import asyncio
import logging
import os
import time
import uuid
import json
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from master_process import MasterProcess

logger = logging.getLogger(__name__)

class UnifiedAPI:
    """统一API服务"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()

        # 初始化FastAPI应用
        self.app = FastAPI(
            title="Whisper Transcription API",
            description="High-performance audio transcription service with shared memory architecture",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        # 设置CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # 主进程管理器
        self.master_process: Optional[MasterProcess] = None

        # 设置路由
        self._setup_routes()

        logger.info("Unified API initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'host': os.getenv('API_HOST', '0.0.0.0'),
            'port': int(os.getenv('API_PORT', '5001')),
            'workers_per_gpu': int(os.getenv('WORKERS_PER_GPU', '2')),
            'max_file_size': int(os.getenv('MAX_FILE_SIZE', '50')) * 1024 * 1024,  # MB to bytes
            'supported_formats': ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm'],
            'log_level': os.getenv('LOG_LEVEL', 'INFO')
        }

    def _setup_routes(self):
        """设置API路由"""

        @self.app.get("/")
        async def root():
            """根路径"""
            return {
                "service": "Whisper Transcription API",
                "version": "1.0.0",
                "status": "running",
                "timestamp": time.time()
            }

        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            if self.master_process is None:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "message": "Master process not initialized"
                    }
                )

            try:
                stats = self.master_process.get_stats()
                return {
                    "status": "healthy",
                    "timestamp": time.time(),
                    "stats": stats
                }
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "message": str(e)
                    }
                )

        @self.app.get("/stats")
        async def get_stats():
            """获取系统统计信息"""
            if self.master_process is None:
                raise HTTPException(status_code=503, detail="Master process not initialized")

            try:
                stats = self.master_process.get_stats()
                return JSONResponse({
                    "code": 0,
                    "msg": "ok",
                    "data": stats
                })
            except Exception as e:
                logger.error(f"Stats error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/transcribe")
        async def transcribe_audio(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            language: str = Form("auto"),
            response_format: str = Form("json"),
            temperature: float = Form(0.0),
            beam_size: int = Form(5)
        ):
            """音频转写接口"""
            if self.master_process is None:
                raise HTTPException(status_code=503, detail="Master process not initialized")

            try:
                # 验证文件格式
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in self.config['supported_formats']:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file format: {file_ext}. Supported formats: {self.config['supported_formats']}"
                    )

                # 读取文件内容
                file_content = await file.read()
                if not file_content:
                    raise HTTPException(status_code=400, detail="Empty file")

                # 检查文件大小
                if len(file_content) > self.config['max_file_size']:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {self.config['max_file_size'] // (1024*1024)}MB"
                    )

                # 生成任务ID
                task_id = str(uuid.uuid4())

                logger.info(f"Received transcription request {task_id}: {file.filename} ({len(file_content)} bytes)")

                # 提交任务到主进程
                task_data = {
                    'client_id': f"api_client_{int(time.time())}",
                    'audio_size': len(file_content),
                    'response_format': response_format,  # 直接放在task_data中，方便worker访问
                    'metadata': {
                        'language': language,
                        'temperature': temperature,
                        'beam_size': beam_size,
                        'original_filename': file.filename,
                        'content_type': file.content_type
                    }
                }

                success, result = await self.master_process.submit_task(task_data)

                if not success:
                    raise HTTPException(status_code=500, detail=f"Task submission failed: {result}")

                task_id = result

                # 如果有共享内存，写入音频数据
                if task_id in self.master_process.tasks:
                    task = self.master_process.tasks[task_id]
                    if task.memory_offset is not None:
                        # 直接从TaskInfo获取GPU ID
                        pool_gpu_id = task.memory_pool_gpu
                        if pool_gpu_id is None:
                            # 备用方案：检查offset属于哪个pool
                            for gpu_id, pool in self.master_process.memory_pools.items():
                                if task.memory_offset < pool.pool_size_bytes:
                                    pool_gpu_id = gpu_id
                                    break

                            if pool_gpu_id is None:
                                pool_gpu_id = 0
                                logger.warning(f"Could not determine GPU ID for task {task_id}, using GPU 0")

                        pool = self.master_process.memory_pools.get(pool_gpu_id)
                        if pool:
                            logger.info(f"Writing {len(file_content)} bytes to shared memory offset {task.memory_offset} (GPU {pool_gpu_id})")

                            # 验证写入操作
                            write_success = pool.write_data(file_content, task.memory_offset, task_id)
                            if not write_success:
                                logger.error(f"Failed to write audio data to shared memory for task {task_id}")
                                pool.free_chunk(task_id)
                                raise HTTPException(status_code=500, detail="Failed to write audio data to shared memory")
                            else:
                                logger.info(f"Successfully wrote audio data to shared memory for task {task_id}")

                            # 验证写入的数据
                            try:
                                # 读取部分数据进行验证
                                verify_data = pool.read_data(task.memory_offset, min(1024, len(file_content)))
                                if verify_data is None or verify_data != file_content[:len(verify_data)]:
                                    pool.free_chunk(task_id)
                                    raise HTTPException(status_code=500, detail="Shared memory write verification failed")

                                logger.info(f"Shared memory write verification successful for {len(file_content)} bytes on GPU {pool_gpu_id}")
                            except Exception as e:
                                pool.free_chunk(task_id)
                                raise HTTPException(status_code=500, detail=f"Shared memory verification failed: {str(e)}")
                        else:
                            logger.error(f"No memory pool found for GPU {pool_gpu_id}")
                            raise HTTPException(status_code=500, detail=f"Memory pool not available for GPU {pool_gpu_id}")

                # 等待结果（同步模式）
                result = await self._wait_for_task_result(task_id, timeout=300)

                return JSONResponse({
                    "code": 0,
                    "msg": "ok",
                    "task_id": task_id,
                    "data": result
                })

            except HTTPException:
                raise
            except Exception as e:
                import traceback
                logger.error(f"Error processing request: {e}")
                logger.error("Full traceback:", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

        @self.app.post("/transcribe_async")
        async def transcribe_audio_async(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            language: str = Form("auto"),
            response_format: str = Form("json"),
            callback_url: Optional[str] = Form(None)
        ):
            """异步音频转写接口"""
            if self.master_process is None:
                raise HTTPException(status_code=503, detail="Master process not initialized")

            try:
                # 验证文件格式
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in self.config['supported_formats']:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file format: {file_ext}"
                    )

                # 读取文件内容
                file_content = await file.read()
                if not file_content:
                    raise HTTPException(status_code=400, detail="Empty file")

                # 检查文件大小
                if len(file_content) > self.config['max_file_size']:
                    raise HTTPException(status_code=413, detail="File too large")

                # 生成任务ID
                task_id = str(uuid.uuid4())

                logger.info(f"Received async transcription request {task_id}: {file.filename}")

                # 提交任务到主进程
                task_data = {
                    'client_id': f"api_async_client_{int(time.time())}",
                    'audio_size': len(file_content),
                    'response_format': response_format,  # 直接放在task_data中，方便worker访问
                    'metadata': {
                        'language': language,
                        'original_filename': file.filename,
                        'content_type': file.content_type,
                        'callback_url': callback_url,
                        'async': True
                    }
                }

                success, result = await self.master_process.submit_task(task_data)

                if not success:
                    raise HTTPException(status_code=500, detail=f"Task submission failed: {result}")

                task_id = result

                # 写入音频数据到共享内存
                if task_id in self.master_process.tasks:
                    task = self.master_process.tasks[task_id]
                    if task.memory_offset is not None:
                        # 直接从TaskInfo获取GPU ID
                        pool_gpu_id = task.memory_pool_gpu
                        if pool_gpu_id is None:
                            # 备用方案：检查offset属于哪个pool
                            for gpu_id, pool in self.master_process.memory_pools.items():
                                if task.memory_offset < pool.pool_size_bytes:
                                    pool_gpu_id = gpu_id
                                    break

                            if pool_gpu_id is None:
                                pool_gpu_id = 0
                                logger.warning(f"ASYNC: Could not determine GPU ID for task {task_id}, using GPU 0")

                        pool = self.master_process.memory_pools.get(pool_gpu_id)
                        if pool:
                            logger.info(f"ASYNC: Writing {len(file_content)} bytes to shared memory offset {task.memory_offset} (GPU {pool_gpu_id})")

                            # 验证写入操作
                            write_success = pool.write_data(file_content, task.memory_offset, task_id)
                            if not write_success:
                                logger.error(f"Failed to write audio data to shared memory for task {task_id}")
                                pool.free_chunk(task_id)
                                raise HTTPException(status_code=500, detail="Failed to write audio data to shared memory")
                            else:
                                logger.info(f"Successfully wrote audio data to shared memory for task {task_id}")

                            # 验证写入的数据
                            try:
                                # 读取部分数据进行验证
                                verify_data = pool.read_data(task.memory_offset, min(1024, len(file_content)))
                                if verify_data is None or verify_data != file_content[:len(verify_data)]:
                                    pool.free_chunk(task_id)
                                    raise HTTPException(status_code=500, detail="Shared memory write verification failed")

                                logger.info(f"ASYNC: Shared memory write verification successful for {len(file_content)} bytes on GPU {pool_gpu_id}")
                            except Exception as e:
                                pool.free_chunk(task_id)
                                raise HTTPException(status_code=500, detail=f"Shared memory verification failed: {str(e)}")
                        else:
                            logger.error(f"ASYNC: No memory pool found for GPU {pool_gpu_id}")
                            raise HTTPException(status_code=500, detail=f"Memory pool not available for GPU {pool_gpu_id}")
                    else:
                        logger.error(f"ASYNC: No memory offset for task {task_id}")
                else:
                    logger.error(f"ASYNC: Task {task_id} not found in master process tasks")

                # 添加回调任务
                if callback_url:
                    background_tasks.add_task(self._send_callback, task_id, callback_url)

                return JSONResponse({
                    "code": 0,
                    "msg": "Task submitted successfully",
                    "task_id": task_id,
                    "status": "processing"
                })

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Async transcription error: {e}")
                raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

        @self.app.get("/task/{task_id}")
        async def get_task_status(task_id: str):
            """获取任务状态"""
            if self.master_process is None:
                raise HTTPException(status_code=503, detail="Master process not initialized")

            try:
                if task_id not in self.master_process.tasks:
                    raise HTTPException(status_code=404, detail="Task not found")

                task = self.master_process.tasks[task_id]

                response = {
                    "code": 0,
                    "msg": "ok",
                    "task_id": task_id,
                    "status": task.status,
                    "created_at": task.created_at,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at,
                    "worker_id": task.worker_id,
                    "retry_count": task.retry_count
                }

                if task.result_data:
                    response["result"] = task.result_data

                if task.error_message:
                    response["error"] = task.error_message

                return JSONResponse(response)

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Task status error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/task/{task_id}")
        async def cancel_task(task_id: str):
            """取消任务"""
            if self.master_process is None:
                raise HTTPException(status_code=503, detail="Master process not initialized")

            try:
                if task_id not in self.master_process.tasks:
                    raise HTTPException(status_code=404, detail="Task not found")

                task = self.master_process.tasks[task_id]

                if task.status in ['completed', 'failed']:
                    raise HTTPException(status_code=400, detail="Task already finished")

                # 释放共享内存
                if task.memory_offset is not None:
                    pool = self.master_process._get_memory_pool_by_gpu(0)  # 假设GPU 0
                    if pool:
                        pool.free_chunk(task_id)

                # 标记任务为已取消
                task.status = 'failed'
                task.error_message = 'Task cancelled by user'
                task.completed_at = time.time()

                return JSONResponse({
                    "code": 0,
                    "msg": "Task cancelled successfully",
                    "task_id": task_id
                })

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Task cancellation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def _wait_for_task_result(self, task_id: str, timeout: int = 300) -> Dict[str, Any]:
        """等待任务结果"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if task_id in self.master_process.tasks:
                task = self.master_process.tasks[task_id]

                if task.status == 'completed':
                    return task.result_data or {}
                elif task.status == 'failed':
                    raise Exception(task.error_message or 'Task failed')

            await asyncio.sleep(0.1)

        raise Exception("Task timeout")

    async def _send_callback(self, task_id: str, callback_url: str):
        """发送回调通知"""
        try:
            import aiohttp

            # 等待任务完成
            max_wait = 3600  # 1小时
            start_time = time.time()

            while time.time() - start_time < max_wait:
                if task_id in self.master_process.tasks:
                    task = self.master_process.tasks[task_id]

                    if task.status in ['completed', 'failed']:
                        # 准备回调数据
                        callback_data = {
                            'task_id': task_id,
                            'status': task.status,
                            'created_at': task.created_at,
                            'completed_at': task.completed_at,
                            'worker_id': task.worker_id
                        }

                        if task.result_data:
                            callback_data['result'] = task.result_data

                        if task.error_message:
                            callback_data['error'] = task.error_message

                        # 发送回调
                        async with aiohttp.ClientSession() as session:
                            async with session.post(
                                callback_url,
                                json=callback_data,
                                timeout=aiohttp.ClientTimeout(total=30)
                            ) as response:
                                if response.status == 200:
                                    logger.info(f"Callback sent successfully for task {task_id}")
                                else:
                                    logger.error(f"Callback failed for task {task_id}: {response.status}")
                        return

                await asyncio.sleep(5)

            logger.warning(f"Callback timeout for task {task_id}")

        except Exception as e:
            logger.error(f"Callback error for task {task_id}: {e}")

    async def start_master_process(self):
        """启动主进程"""
        try:
            self.master_process = MasterProcess(self.config)
            # 在后台启动主进程
            asyncio.create_task(self.master_process.start())
            logger.info("Master process started")
        except Exception as e:
            logger.error(f"Failed to start master process: {e}")
            raise

    async def run(self):
        """运行API服务"""
        # 启动主进程
        await self.start_master_process()

        # 等待工作进程启动
        await asyncio.sleep(3)

        # 启动HTTP服务器
        config = uvicorn.Config(
            app=self.app,
            host=self.config['host'],
            port=self.config['port'],
            log_level=self.config['log_level'].lower()
        )

        server = uvicorn.Server(config)
        await server.serve()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Whisper Transcription API Server')

    parser.add_argument('--host',
                       default=os.getenv('API_HOST', '0.0.0.0'),
                       help='Host to bind to (default: 0.0.0.0 or API_HOST env var)')

    parser.add_argument('--port',
                       type=int,
                       default=int(os.getenv('API_PORT', '5001')),
                       help='Port to bind to (default: 5001 or API_PORT env var)')

    parser.add_argument('--workers-per-gpu',
                       type=int,
                       default=int(os.getenv('WORKERS_PER_GPU', '2')),
                       help='Number of worker processes per GPU (default: 2 or WORKERS_PER_GPU env var)')

    parser.add_argument('--model',
                       default=os.getenv('WHISPER_MODEL', 'large-v3-turbo'),
                       choices=['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large-v3-turbo'],
                       help='Whisper model to use (default: large-v3-turbo or WHISPER_MODEL env var)')

    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       default=os.getenv('LOG_LEVEL', 'INFO'),
                       help='Log level (default: INFO or LOG_LEVEL env var)')

    parser.add_argument('--max-file-size',
                       type=int,
                       default=int(os.getenv('MAX_FILE_SIZE', '50')),
                       help='Maximum file size in MB (default: 50 or MAX_FILE_SIZE env var)')

    return parser.parse_args()


async def main():
    """主函数"""
    args = parse_args()

    # 设置日志级别
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 创建配置字典
    config = {
        'host': args.host,
        'port': args.port,
        'workers_per_gpu': args.workers_per_gpu,
        'model': args.model,  # 添加模型配置
        'max_file_size': args.max_file_size * 1024 * 1024,  # MB to bytes
        'supported_formats': ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm'],
        'log_level': args.log_level,
        # MasterProcess需要的配置参数
        'max_queue_size': int(os.getenv('MAX_QUEUE_SIZE', '100')),
        'health_check_interval': int(os.getenv('HEALTH_CHECK_INTERVAL', '30')),
        'task_timeout': int(os.getenv('TASK_TIMEOUT', '300')),
        'max_retries': int(os.getenv('MAX_RETRIES', '3')),
        'gpu_memory_fraction': float(os.getenv('GPU_MEMORY_FRACTION', '0.8'))
    }

    api = UnifiedAPI(config)
    await api.run()

if __name__ == "__main__":
    asyncio.run(main())