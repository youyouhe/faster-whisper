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
    model_path: str = "large-v3-turbo"
    compute_type: str = "float32"
    device: str = "cuda"
    max_workers: int = 1

class WorkerProcess:
    """工作进程类"""

    def __init__(self, worker_id: int, gpu_id: int,
                 task_queue: Queue, result_queue: Queue,
                 config: Optional[WorkerConfig] = None,
                 model_path: Optional[str] = None):
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.task_queue = task_queue
        self.result_queue = result_queue

        # 如果传入了model_path，使用它；否则使用config中的默认值
        if config is None:
            config = WorkerConfig(worker_id, gpu_id)
        if model_path:
            config.model_path = model_path

        self.config = config

        # 状态
        self.running = False
        self.current_task = None
        self.processed_tasks = 0
        self.error_count = 0

        # Whisper模型
        self.model = None

        # 共享内存池 - 支持多个GPU的内存池
        self.memory_pools = {}  # 存储多个GPU的内存池

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

        # 设置GPU设备 - 只有在没有外部CUDA_VISIBLE_DEVICES时才设置
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            logger.info(f"Worker {self.worker_id} setting CUDA_VISIBLE_DEVICES={self.gpu_id}")
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        else:
            logger.info(f"Worker {self.worker_id} using external CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")

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

            # 使用配置中的模型路径
            model_path = self.config.model_path
            logger.info(f"Worker {self.worker_id} loading Whisper model: {model_path}")

            # 尝试不同的计算类型，优先使用float32
            compute_types = ["float32", "float16", "int8"]
            for compute_type in compute_types:
                try:
                    self.model = WhisperModel(
                        model_path,
                        device=self.config.device,
                        compute_type=compute_type
                    )
                    logger.info(f"Worker {self.worker_id} model loaded successfully with {compute_type} ({model_path})")

                    # 模型加载成功后的验证
                    try:
                        # 创建一个有效的最小WAV文件进行验证
                        import tempfile
                        import struct

                        # 创建一个有效的最小WAV文件 (16-bit PCM, mono, 16kHz, 1秒静音)
                        sample_rate = 16000
                        duration_seconds = 1
                        num_samples = sample_rate * duration_seconds

                        # WAV文件头
                        # RIFF header (12 bytes)
                        riff_header = b'RIFF'
                        chunk_size = struct.pack('<I', 36 + num_samples * 2)  # 文件大小-8
                        wave_format = b'WAVE'

                        # fmt chunk (24 bytes)
                        fmt_header = b'fmt '
                        fmt_size = struct.pack('<I', 16)  # fmt chunk size
                        audio_format = struct.pack('<H', 1)  # PCM
                        channels = struct.pack('<H', 1)  # mono
                        sample_rate_bytes = struct.pack('<I', sample_rate)
                        byte_rate = struct.pack('<I', sample_rate * 2)  # sample_rate * channels * bytes_per_sample
                        block_align = struct.pack('<H', 2)  # channels * bytes_per_sample
                        bits_per_sample = struct.pack('<H', 16)

                        # data chunk header (8 bytes + data)
                        data_header = b'data'
                        data_size = struct.pack('<I', num_samples * 2)  # number of samples * bytes_per_sample
                        silence_data = b'\x00' * (num_samples * 2)  # 静音数据

                        # 组装完整的WAV文件
                        test_audio = (riff_header + chunk_size + wave_format +
                                    fmt_header + fmt_size + audio_format + channels +
                                    sample_rate_bytes + byte_rate + block_align + bits_per_sample +
                                    data_header + data_size + silence_data)

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                            temp_file.write(test_audio)
                            temp_file.flush()
                            temp_file_path = temp_file.name

                        try:
                            info = self.model.transcribe(
                                temp_file_path,
                                language=None,
                                beam_size=1
                            )
                            logger.info(f"Worker {self.worker_id} model validation successful")
                        finally:
                            import os
                            try:
                                os.unlink(temp_file_path)
                            except:
                                pass
                    except Exception as model_error:
                        logger.error(f"Worker {self.worker_id}: Model validation failed: {model_error}")

                    return True
                except Exception as e:
                    logger.error(f"Worker {self.worker_id}: Failed to load model {model_path} with {compute_type}: {e}")
                    continue

            logger.error(f"Worker {self.worker_id}: Failed to load model {model_path} with any compute type")
            return False

        except ImportError:
            logger.error(f"Worker {self.worker_id}: faster_whisper not available")
            # 继续运行，使用模拟模式
            return True
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Model loading failed with unexpected error: {e}")
            # 继续运行，使用模拟模式
            return True

    def _init_memory_pool(self) -> bool:
        """初始化共享内存池"""
        try:
            # 获取共享内存配置
            config = SharedMemoryConfig()

            # 初始化所有可能的GPU内存池
            # 总是初始化所有3个GPU的内存池，不受CUDA_VISIBLE_DEVICES限制
            gpu_count = 3  # 固定为3个GPU
            logger.info(f"Worker {self.worker_id}: Initializing {gpu_count} GPU memory pools (all GPUs) with size {config.pool_size_mb}MB, chunks {config.chunk_size_mb}MB")

            for gpu_id in range(gpu_count):
                try:
                    pool = SharedMemoryPool(
                        gpu_id=gpu_id,
                        pool_size_mb=config.pool_size_mb,
                        chunk_size_mb=config.chunk_size_mb
                    )
                    self.memory_pools[gpu_id] = pool
                    logger.info(f"Worker {self.worker_id} initialized memory pool for GPU {gpu_id}")
                except Exception as e:
                    logger.warning(f"Worker {self.worker_id} failed to init memory pool for GPU {gpu_id}: {e}")

            # 确保初始化自己的GPU内存池
            if self.gpu_id not in self.memory_pools:
                try:
                    pool = SharedMemoryPool(
                        gpu_id=self.gpu_id,
                        pool_size_mb=config.pool_size_mb,
                        chunk_size_mb=config.chunk_size_mb
                    )
                    self.memory_pools[self.gpu_id] = pool
                    logger.info(f"Worker {self.worker_id} initialized own memory pool for GPU {self.gpu_id}")
                except Exception as e:
                    logger.error(f"Worker {self.worker_id} failed to init own memory pool: {e}")
                    return False

            logger.info(f"Worker {self.worker_id} initialized {len(self.memory_pools)} memory pools: {list(self.memory_pools.keys())}")
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
        response_format = task_data.get('response_format', 'json')

        try:
            # 从共享内存读取音频数据
            audio_data = self._read_audio_from_shared_memory(task)

            if audio_data is None:
                raise Exception("Failed to read audio data from shared memory")

            # 使用Whisper进行转写
            logger.info(f"Worker {self.worker_id} transcribing audio for task {task_id} (format: {response_format})")

            # 创建临时文件
            import tempfile
            import os

            # 确保音频数据有效
            if not audio_data or len(audio_data) == 0:
                raise Exception("Empty audio data")

            # 确定文件扩展名
            file_ext = ".wav"
            if audio_data.startswith(b'ID3'):
                file_ext = ".mp3"

            temp_file_path = None
            max_retries = 3
            retry_count = 0

            while retry_count < max_retries:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, mode='wb') as temp_file:
                        # 确保写入完整数据
                        temp_file.write(audio_data)
                        temp_file.flush()
                        os.fsync(temp_file.fileno())  # 强制写入磁盘
                        temp_file_path = temp_file.name

                    # 验证文件是否正确创建
                    if not os.path.exists(temp_file_path):
                        raise Exception("Temporary file not created")

                    file_size = os.path.getsize(temp_file_path)
                    if file_size != len(audio_data):
                        raise Exception(f"File size mismatch: expected {len(audio_data)}, got {file_size}")

                    # 验证文件可读
                    with open(temp_file_path, 'rb') as verify_file:
                        verify_data = verify_file.read(min(1024, len(audio_data)))
                        if not verify_data.startswith(audio_data[:len(verify_data)]):
                            raise Exception("File content verification failed")

                    logger.info(f"Worker {self.worker_id} created temporary file: {temp_file_path} ({file_size} bytes)")
                    break

                except Exception as e:
                    retry_count += 1
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass
                        temp_file_path = None

                    if retry_count >= max_retries:
                        raise Exception(f"Failed to create temporary file after {max_retries} attempts: {e}")

                    logger.warning(f"Worker {self.worker_id}: Temporary file creation attempt {retry_count} failed: {e}")
                    import time
                    time.sleep(0.1)  # 短暂等待后重试

            try:
                # 使用文件路径进行转写
                logger.info(f"Worker {self.worker_id} transcribing file: {temp_file_path}")
                segments, info = self.model.transcribe(
                    temp_file_path,
                    language=task_data.get('language', None) if task_data.get('language') != 'auto' else None,
                    beam_size=5,
                    vad_filter=True,
                    word_timestamps=True
                )

                # 根据response_format生成不同格式的结果
                if response_format == 'srt':
                    result = self._generate_srt_result(segments, info)
                else:
                    # 默认JSON格式
                    result = self._generate_json_result(segments, info)

                logger.info(f"Worker {self.worker_id} transcription completed for task {task_id}")
                return result

            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

        except Exception as e:
            # 提供详细的错误信息用于调试
            error_msg = str(e)
            logger.error(f"Worker {self.worker_id} transcription failed for task {task_id}: {error_msg}")

            # 如果是文件相关错误，提供更多调试信息
            if "Invalid data found when processing input" in error_msg:
                logger.error(f"Worker {self.worker_id}: Audio processing error details:")
                logger.error(f"  - Temp file path: {temp_file_path}")
                logger.error(f"  - Audio data size: {len(audio_data) if audio_data else 0}")
                logger.error(f"  - File exists: {os.path.exists(temp_file_path) if temp_file_path else 'N/A'}")

                if temp_file_path and os.path.exists(temp_file_path):
                    try:
                        file_size = os.path.getsize(temp_file_path)
                        logger.error(f"  - File size: {file_size}")

                        # 检查文件头
                        with open(temp_file_path, 'rb') as f:
                            header = f.read(16)
                            logger.error(f"  - File header: {header}")
                    except Exception as header_error:
                        logger.error(f"  - Header check failed: {header_error}")

                # 提供原始音频数据的头信息
                if audio_data:
                    logger.error(f"  - Original audio header: {audio_data[:16]}")

            raise

    def _generate_json_result(self, segments, info) -> Dict[str, Any]:
        """生成JSON格式结果"""
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

        return result

    def _generate_srt_result(self, segments, info) -> Dict[str, Any]:
        """生成SRT格式结果"""
        try:
            # 生成SRT内容
            srt_lines = []
            segment_index = 1

            # 清理文本函数
            def clean_text(text):
                if not text:
                    return text
                # 移除多余的标点符号空格
                import re
                text = re.sub(r'\s+([,.!?;:，。！？；：])', r'\1', text)
                # 移除中文字符和标点之间的空格
                text = re.sub(r'([^\s])\s+([,.!?;:，。！？；：])', r'\1\2', text)
                # 移除多个连续空格
                text = re.sub(r'\s+', ' ', text)
                # 去除首尾空白
                text = text.strip()
                return text

            # 时间戳格式化函数
            def format_timestamp_srt(seconds):
                """将秒转换为SRT时间戳格式 (HH:MM:SS,mmm)"""
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                millis = int((seconds % 1) * 1000)

                return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

            # 处理每个segment
            for segment in segments:
                if segment.text.strip():
                    # 清理文本
                    cleaned_text = clean_text(segment.text)

                    if cleaned_text.strip():  # 只添加非空段落
                        # 添加段落编号
                        srt_lines.append(str(segment_index))

                        # 添加时间戳
                        srt_lines.append(f"{format_timestamp_srt(segment.start)} --> {format_timestamp_srt(segment.end)}")

                        # 添加文本内容
                        srt_lines.append(cleaned_text)

                        # 添加空行
                        srt_lines.append("")
                        segment_index += 1

            # 合并所有行为SRT内容
            srt_content = "\n".join(srt_lines).strip()

            # 清理BOM和无效字符
            if srt_content.startswith('\ufeff'):
                srt_content = srt_content[1:]

            result = {
                'text': srt_content,
                'format': 'srt',
                'language': info.language if hasattr(info, 'language') else 'en',
                'language_probability': getattr(info, 'language_probability', 1.0),
                'detected_language': getattr(info, 'language', 'en'),
                'segments_count': segment_index - 1
            }

            logger.info(f"Worker {self.worker_id} generated SRT with {segment_index - 1} segments")
            return result

        except Exception as e:
            logger.error(f"Worker {self.worker_id} SRT generation failed: {e}")
            # 如果SRT生成失败，返回JSON格式
            return self._generate_json_result(segments, info)

    def _read_audio_from_shared_memory(self, task: Dict[str, Any]) -> Optional[bytes]:
        """从共享内存读取音频数据"""
        try:
            memory_offset = task.get('memory_offset')
            task_id = task.get('task_id')
            memory_pool_gpu = task.get('memory_pool_gpu')  # 获取数据写入的GPU ID

            if memory_offset is None:
                raise Exception("No memory offset provided")

            # 获取音频大小 - 优先使用直接传递的大小，然后从task_data获取
            audio_size = task.get('audio_size')
            if audio_size is None:
                audio_size = task.get('task_data', {}).get('audio_size', 0)

            if audio_size == 0:
                raise Exception("Invalid audio size: 0 bytes")

            if audio_size > 100 * 1024 * 1024:  # 100MB限制
                raise Exception(f"Audio size too large: {audio_size} bytes")

            logger.info(f"Worker {self.worker_id} reading {audio_size} bytes from offset {memory_offset} (GPU pool: {memory_pool_gpu})")

            # 必须使用数据所在的内存池
            if memory_pool_gpu is not None and memory_pool_gpu in self.memory_pools:
                memory_pool = self.memory_pools[memory_pool_gpu]
                logger.info(f"Worker {self.worker_id} using memory pool from GPU {memory_pool_gpu} (data location)")
            else:
                logger.error(f"Worker {self.worker_id}: Required memory pool GPU {memory_pool_gpu} not available")
                logger.error(f"Worker {self.worker_id}: Available pools: {list(self.memory_pools.keys())}")
                return None

            # 从共享内存池读取数据
            audio_data = memory_pool.read_data(
                offset=memory_offset,
                size=audio_size
            )

            if audio_data is None:
                raise Exception("Failed to read from shared memory")

            if len(audio_data) != audio_size:
                raise Exception(f"Data size mismatch: expected {audio_size}, got {len(audio_data)}")

            # 详细调试：检查数据内容
            logger.debug(f"Worker {self.worker_id} audio data header: {audio_data[:32]}")
            if all(b == 0 for b in audio_data[:32]):
                logger.error(f"Worker {self.worker_id}: CRITICAL - Audio data is all zero!")
                # 尝试读取更多数据来确认
                sample_data = memory_pool.read_data(memory_offset, min(1024, audio_size))
                if sample_data:
                    logger.error(f"Worker {self.worker_id}: Sample data: {sample_data[:64]}")
                    logger.error(f"Worker {self.worker_id}: Non-zero bytes in sample: {sum(1 for b in sample_data if b != 0)}")

            # 验证音频数据完整性
            if not self._validate_audio_data(audio_data):
                raise Exception("Audio data validation failed")

            logger.info(f"Worker {self.worker_id} successfully read and validated {len(audio_data)} bytes from shared memory")
            return audio_data

        except Exception as e:
            logger.error(f"Worker {self.worker_id} failed to read from shared memory: {e}")
            return None

    def _validate_audio_data(self, audio_data: bytes) -> bool:
        """验证音频数据的完整性"""
        try:
            if len(audio_data) < 44:  # WAV文件头最小大小
                logger.error(f"Worker {self.worker_id}: Audio data too small: {len(audio_data)} bytes")
                return False

            # 检查WAV文件头
            if audio_data.startswith(b'RIFF'):
                # 验证RIFF格式
                if len(audio_data) < 12:
                    return False

                # 检查WAVE标识
                if audio_data[8:12] != b'WAVE':
                    logger.error(f"Worker {self.worker_id}: Invalid WAV format")
                    return False

                # 获取文件大小并验证
                riff_size = int.from_bytes(audio_data[4:8], byteorder='little')
                if riff_size != len(audio_data) - 8:
                    logger.warning(f"Worker {self.worker_id}: RIFF size mismatch: {riff_size} vs {len(audio_data) - 8}")
                    # 不返回False，因为可能存在padding

            # 检查MP3文件头 (ID3v1/v2)
            elif audio_data.startswith(b'ID3') or audio_data[:3] == b'ID3':
                # MP3文件，进行基本检查
                pass
            else:
                # 其他格式，进行基本检查
                logger.warning(f"Worker {self.worker_id}: Unknown audio format, proceeding anyway")

            return True

        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Audio validation error: {e}")
            return False

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

def worker_main(worker_id: int, gpu_id: int, task_queue: Queue, result_queue: Queue, model_path: Optional[str] = None):
    """工作进程主函数"""
    try:
        # 创建工作进程
        worker = WorkerProcess(worker_id, gpu_id, task_queue, result_queue, model_path=model_path)

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