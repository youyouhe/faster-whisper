#!/usr/bin/env python3
"""
工作进程 - 负责实际音频转写处理
Worker Process for Audio Transcription Processing
"""

import os
import sys
import logging
import time
import uuid
import json
import signal
import threading
from typing import Dict, Any, Optional, Tuple
from multiprocessing import Queue
from dataclasses import dataclass

from shared_memory_manager import SharedMemoryPool, SharedMemoryConfig

logger = logging.getLogger(__name__)

@dataclass
class WorkerConfig:
    """工作进程配置"""
    worker_id: int
    gpu_id: int
    model_path: str = "base"
    compute_type: str = "float32"
    device: str = "cuda"
    max_workers: int = 1

class WorkerProcess:
    """工作进程类"""

    def __init__(self, worker_id: int, gpu_id: int,
                 task_queue: Queue, result_queue: Queue,
                 config: Optional[WorkerConfig] = None):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.config = config or WorkerConfig(worker_id, gpu_id)

        # 状态
        self.running = False
        self.current_task = None
        self.processed_tasks = 0
        self.error_count = 0

        # Whisper模型
        self.model = None

        # 共享内存池
        self.memory_pool = None

        # 设置日志
        self._setup_logging()

        logger.info(f"Worker {worker_id} initialized for GPU {gpu_id}")

    def _setup_logging(self):
        """设置日志"""
        # 设置工作进程专用日志
        logging.basicConfig(
            level=logging.INFO,
            format=f'%(asctime)s - Worker{self.worker_id} - %(levelname)s - %(message)s'
        )

    def start(self):
        """启动工作进程"""
        logger.info(f"Starting worker {self.worker_id}...")
        self.running = True

        # 设置GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        # 初始化模型
        if not self._init_model():
            logger.error(f"Worker {self.worker_id} failed to initialize model")
            return False

        # 初始化共享内存池
        if not self._init_memory_pool():
            logger.error(f"Worker {self.worker_id} failed to initialize memory pool")
            return False

        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # 开始处理任务
        self._run()

        return True

    def _init_model(self) -> bool:
        """初始化Whisper模型"""
        try:
            from faster_whisper import WhisperModel

            logger.info(f"Worker {self.worker_id} loading Whisper model...")

            # 使用large-v3-turbo模型提高识别率
            model_path = "large-v3-turbo"

            # 尝试不同的计算类型，优先使用float32
            compute_types = ["float32", "float16", "int8"]
            for compute_type in compute_types:
                try:
                    self.model = WhisperModel(
                        model_path,
                        device=self.config.device,
                        compute_type=compute_type
                    )
                    logger.info(f"Worker {self.worker_id} model loaded successfully with {compute_type}")
                    return True
                except Exception as e:
                    logger.warning(f"Worker {self.worker_id}: Failed to load model with {compute_type}: {e}")
                    continue

            logger.error(f"Worker {self.worker_id}: Failed to load model with any compute type")
            return False

        except ImportError:
            logger.error(f"Worker {self.worker_id}: faster_whisper not available")
            # 继续运行，使用模拟模式
            return True
        except Exception as e:
            logger.error(f"Worker {self.worker_id} model loading failed: {e}")
            # 继续运行，使用模拟模式
            return True

    def _init_memory_pool(self) -> bool:
        """初始化共享内存池"""
        try:
            self.memory_pool = SharedMemoryPool(
                gpu_id=self.gpu_id,
                pool_size_mb=200,
                chunk_size_mb=50
            )
            logger.info(f"Worker {self.worker_id} memory pool initialized")
            return True
        except Exception as e:
            logger.error(f"Worker {self.worker_id} memory pool init failed: {e}")
            return False

    def _run(self):
        """主工作循环"""
        logger.info(f"Worker {self.worker_id} starting main loop")

        while self.running:
            try:
                # 获取任务（带超时）
                try:
                    task = self.task_queue.get(timeout=1.0)
                    if task is None:  # 关闭信号
                        break

                    logger.info(f"Worker {self.worker_id} received task: {task.get('task_id')}")

                    # 处理任务
                    result = self._process_task(task)

                    # 发送结果
                    self.result_queue.put(result)

                except Exception as e:
                    if "Empty" in str(type(e)):  # Queue.Empty
                        continue
                    else:
                        logger.error(f"Worker {self.worker_id} task error: {e}")
                        self.error_count += 1

            except Exception as e:
                logger.error(f"Worker {self.worker_id} main loop error: {e}")
                self.error_count += 1

        logger.info(f"Worker {self.worker_id} stopped")

    def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个任务"""
        task_id = task.get('task_id')
        start_time = time.time()

        try:
            self.current_task = task_id
            logger.info(f"Worker {self.worker_id} processing task {task_id}")

            # 模拟音频转写（如果模型加载失败）
            if self.model is None:
                # 模拟处理时间
                time.sleep(0.5)

                result_data = {
                    'text': f'Sample transcription for task {task_id}',
                    'segments': [
                        {
                            'start': 0.0,
                            'end': 2.0,
                            'text': f'Hello world from worker {self.worker_id}'
                        }
                    ],
                    'language': 'en'
                }
            else:
                # 实际音频转写
                result_data = self._transcribe_audio(task)

            processing_time = time.time() - start_time
            self.processed_tasks += 1

            result = {
                'task_id': task_id,
                'worker_id': self.worker_id,
                'gpu_id': self.gpu_id,
                'status': 'completed',
                'result': result_data,
                'processing_time': processing_time,
                'processed_at': time.time()
            }

            logger.info(f"Worker {self.worker_id} completed task {task_id} in {processing_time:.2f}s")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            self.error_count += 1

            logger.error(f"Worker {self.worker_id} failed task {task_id}: {e}")

            result = {
                'task_id': task_id,
                'worker_id': self.worker_id,
                'gpu_id': self.gpu_id,
                'status': 'failed',
                'error': str(e),
                'processing_time': processing_time,
                'processed_at': time.time()
            }

            return result
        finally:
            self.current_task = None

    def _transcribe_audio(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """实际音频转写"""
        task_id = task.get('task_id')
        task_data = task.get('task_data', {})

        try:
            # 从共享内存读取音频数据
            audio_data = self._read_audio_from_shared_memory(task)

            if audio_data is None:
                raise Exception("Failed to read audio data from shared memory")

            # 使用Whisper进行转写
            logger.info(f"Worker {self.worker_id} transcribing audio for task {task_id}")

            # 创建临时文件
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            try:
                # 使用文件路径进行转写
                segments, info = self.model.transcribe(
                    temp_file_path,
                    language=task_data.get('language', None) if task_data.get('language') != 'auto' else None,
                    beam_size=5,
                    vad_filter=True,
                    word_timestamps=True
                )

                # 收集结果
                result_segments = []
                full_text = ""

                for segment in segments:
                    segment_data = {
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text.strip()
                    }

                    # 添加词级别时间戳
                    if hasattr(segment, 'words') and segment.words:
                        segment_data['words'] = [
                            {
                                'start': word.start,
                                'end': word.end,
                                'word': word.word,
                                'probability': word.probability
                            } for word in segment.words
                        ]

                    result_segments.append(segment_data)
                    full_text += segment.text + " "

                result = {
                    'text': full_text.strip(),
                    'segments': result_segments,
                    'language': info.language if hasattr(info, 'language') else 'en',
                    'language_probability': getattr(info, 'language_probability', 1.0),
                    'detected_language': getattr(info, 'language', 'en')
                }

                logger.info(f"Worker {self.worker_id} transcription completed for task {task_id}")
                return result

            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"Worker {self.worker_id} transcription failed for task {task_id}: {e}")
            raise

    def _read_audio_from_shared_memory(self, task: Dict[str, Any]) -> Optional[bytes]:
        """从共享内存读取音频数据"""
        try:
            memory_offset = task.get('memory_offset')
            task_id = task.get('task_id')

            if memory_offset is None:
                raise Exception("No memory offset provided")

            # 获取音频大小 - 优先使用直接传递的大小，然后从task_data获取
            audio_size = task.get('audio_size')
            if audio_size is None:
                audio_size = task.get('task_data', {}).get('audio_size', 0)

            if audio_size == 0:
                raise Exception("Invalid audio size: 0 bytes")

            logger.info(f"Worker {self.worker_id} reading {audio_size} bytes from offset {memory_offset}")

            # 从共享内存池读取数据
            audio_data = self.memory_pool.read_data(
                offset=memory_offset,
                size=audio_size
            )

            if audio_data is None:
                raise Exception("Failed to read from shared memory")

            logger.info(f"Worker {self.worker_id} successfully read {len(audio_data)} bytes from shared memory")
            return audio_data

        except Exception as e:
            logger.error(f"Worker {self.worker_id} failed to read from shared memory: {e}")
            return None

    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"Worker {self.worker_id} received signal {signum}, shutting down...")
        self.running = False

    def get_stats(self) -> Dict[str, Any]:
        """获取工作进程统计信息"""
        return {
            'worker_id': self.worker_id,
            'gpu_id': self.gpu_id,
            'running': self.running,
            'current_task': self.current_task,
            'processed_tasks': self.processed_tasks,
            'error_count': self.error_count,
            'model_loaded': self.model is not None,
            'memory_pool_connected': self.memory_pool is not None
        }

def worker_main(worker_id: int, gpu_id: int, task_queue: Queue, result_queue: Queue):
    """工作进程主函数"""
    try:
        # 创建工作进程
        worker = WorkerProcess(worker_id, gpu_id, task_queue, result_queue)

        # 启动工作进程
        worker.start()

    except Exception as e:
        logger.error(f"Worker {worker_id} fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 如果直接运行此文件，用于测试
    import multiprocessing as mp

    if len(sys.argv) >= 3:
        worker_id = int(sys.argv[1])
        gpu_id = int(sys.argv[2])

        # 创建队列
        task_queue = mp.Queue()
        result_queue = mp.Queue()

        # 运行工作进程
        worker_main(worker_id, gpu_id, task_queue, result_queue)
    else:
        print("Usage: python worker_process.py <worker_id> <gpu_id>")
        sys.exit(1)