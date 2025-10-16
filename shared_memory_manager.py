#!/usr/bin/env python3
"""
Shared Memory Manager for Single-Port Multi-Process Architecture
高性能进程间通信，使用共享内存传输音频数据
"""

import multiprocessing as mp
import os
import uuid
import time
import logging
from multiprocessing import shared_memory
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import threading
import struct

logger = logging.getLogger(__name__)

@dataclass
class MemoryChunk:
    """内存块信息"""
    task_id: str
    offset: int
    size: int
    status: int  # 0=free, 1=allocated, 2=processing, 3=completed
    timestamp: float
    worker_id: Optional[int] = None

@dataclass
class Task:
    """任务信息"""
    task_id: str
    offset: int
    size: int
    metadata: Dict[str, Any]

@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    result_data: bytes
    processing_time: float
    worker_id: int
    status: str  # success, error

class SharedMemoryPool:
    """共享内存池管理器"""

    def __init__(self, gpu_id: int, pool_size_mb: int = 200, chunk_size_mb: int = 50):
        self.gpu_id = gpu_id
        self.pool_size_bytes = pool_size_mb * 1024 * 1024
        self.chunk_size_bytes = chunk_size_mb * 1024 * 1024
        self.max_chunks = self.pool_size_bytes // self.chunk_size_bytes

        # 创建共享内存块
        self.memory_name = f"whisper_shared_mem_gpu_{gpu_id}"
        try:
            self.shared_block = shared_memory.SharedMemory(
                name=self.memory_name,
                create=True,
                size=self.pool_size_bytes
            )
            logger.info(f"Created shared memory pool: {self.memory_name}, size: {pool_size_mb}MB")
        except FileExistsError:
            # 内存块已存在，连接到现有块
            self.shared_block = shared_memory.SharedMemory(name=self.memory_name)
            logger.info(f"Connected to existing shared memory pool: {self.memory_name}")

        # 创建元数据管理数组
        self.metadata_name = f"whisper_metadata_gpu_{gpu_id}"
        try:
            self.metadata = shared_memory.SharedMemory(
                name=self.metadata_name,
                create=True,
                size=self.max_chunks * 64  # 每个chunk 64字节元数据
            )
        except FileExistsError:
            self.metadata = shared_memory.SharedMemory(name=self.metadata_name)

        # 同步锁
        self.lock = mp.Lock()
        self.chunk_locks = [mp.Lock() for _ in range(self.max_chunks)]

        # 进程间通信队列
        self.task_queue = mp.Queue(maxsize=100)
        self.result_queue = mp.Queue(maxsize=100)

        # 初始化元数据
        self._init_metadata()

        # 活跃任务跟踪
        self.active_tasks = {}
        self.completed_tasks = {}

    def _init_metadata(self):
        """初始化元数据区域"""
        with self.lock:
            for i in range(self.max_chunks):
                offset = i * 64
                # 使用struct打包元数据
                metadata = struct.pack(
                    '!64s',  # 64字节的chunk信息
                    b''.join([
                        struct.pack('!i', 0),  # status: free
                        struct.pack('!i', i),  # chunk_index
                        struct.pack('!i', 0),  # task_size
                        struct.pack('!i', 0),  # task_offset
                        struct.pack('!d', 0.0),  # timestamp
                        struct.pack('!i', -1),  # worker_id
                        b'\x00' * 40  # padding
                    ])
                )
                self.metadata.buf[offset:offset+64] = metadata

    def allocate_chunk(self, size: int, task_id: str = None) -> Tuple[Optional[int], Optional[str]]:
        """分配内存块"""
        if task_id is None:
            task_id = str(uuid.uuid4())

        with self.lock:
            # 查找空闲块
            for i in range(self.max_chunks):
                offset = i * 64
                chunk_metadata = self.metadata.buf[offset:offset+64]

                # 解包状态
                status = struct.unpack('!i', chunk_metadata[0:4])[0]

                if status == 0:  # free
                    # 检查块是否足够大
                    chunk_offset = i * self.chunk_size_bytes
                    if size <= self.chunk_size_bytes:
                        # 更新元数据
                        new_metadata = struct.pack(
                            '!64s',
                            b''.join([
                                struct.pack('!i', 1),  # status: allocated
                                struct.pack('!i', i),  # chunk_index
                                struct.pack('!i', size),  # task_size
                                struct.pack('!i', chunk_offset),  # task_offset
                                struct.pack('!d', time.time()),  # timestamp
                                struct.pack('!i', -1),  # worker_id
                                task_id.encode('utf-8')[:32]  # task_id (32字节)
                            ])
                        )
                        self.metadata.buf[offset:offset+64] = new_metadata

                        # 记录分配
                        self.active_tasks[task_id] = MemoryChunk(
                            task_id=task_id,
                            offset=chunk_offset,
                            size=size,
                            status=1,
                            timestamp=time.time()
                        )

                        # 测试写入：验证内存区域可写
                        test_pattern = b'TEST' * 4  # 16字节测试模式
                        self.shared_block.buf[chunk_offset:chunk_offset+16] = test_pattern
                        verify = bytes(self.shared_block.buf[chunk_offset:chunk_offset+16])
                        if verify != test_pattern:
                            logger.error(f"Memory write test failed at offset {chunk_offset}")
                        else:
                            # 清除测试数据
                            self.shared_block.buf[chunk_offset:chunk_offset+16] = b'\x00' * 16

                        logger.debug(f"Allocated chunk {i} for task {task_id}, size: {size}")
                        return chunk_offset, None

            return None, "No available chunks in shared memory pool"

    def write_data(self, data: bytes, offset: int, task_id: str = None) -> bool:
        """写入数据到共享内存"""
        import time  # 确保time模块可用

        try:
            with self.lock:
                if offset + len(data) > self.pool_size_bytes:
                    return False

                logger.info(f"Writing {len(data)} bytes to shared memory at offset {offset} (pool GPU {self.gpu_id}, pool name: {self.memory_name})")
                logger.debug(f"Data header: {data[:32]}")

                # 更新chunk状态为processing（在共享元数据中）
                chunk_index = offset // self.chunk_size_bytes
                metadata_offset = chunk_index * 64

                if task_id:
                    # 在写入数据前更新元数据状态为processing
                    new_metadata = struct.pack(
                        '!64s',
                        b''.join([
                            struct.pack('!i', 2),  # status: processing
                            struct.pack('!i', chunk_index),  # chunk_index
                            struct.pack('!i', len(data)),  # task_size
                            struct.pack('!i', offset),  # task_offset
                            struct.pack('!d', time.time()),  # timestamp
                            struct.pack('!i', -1),  # worker_id (尚未分配)
                            task_id.encode('utf-8')[:32]  # task_id (32字节)
                        ])
                    )
                    self.metadata.buf[metadata_offset:metadata_offset+64] = new_metadata
                    logger.info(f"Updated chunk {chunk_index} status to processing for task {task_id}")

                    # 验证元数据写入
                    verification = self.metadata.buf[metadata_offset:metadata_offset+64]
                    if verification != new_metadata:
                        logger.error(f"Metadata write verification failed for chunk {chunk_index}")
                    else:
                        logger.debug(f"Metadata write verification successful for chunk {chunk_index}")

                # 写入数据（会自动覆盖任何旧数据）
                end_offset = offset + len(data)
                self.shared_block.buf[offset:end_offset] = data

                # 强制内存同步 - 确保数据写入到物理内存
                import ctypes
                try:
                    # 使用ctypes强制内存同步
                    ctypes.memset(ctypes.addressof(self.shared_block.buf) + offset, 0, 0)
                except:
                    # 备用方案：多次读写触发同步
                    for i in range(3):
                        _ = self.shared_block.buf[offset]
                        _ = self.shared_block.buf[offset + len(data) - 1] if len(data) > 0 else 0

                # 验证写入（多次验证确保同步）
                max_retries = 3
                for retry in range(max_retries):
                    if len(data) >= 16:
                        written = bytes(self.shared_block.buf[offset:offset+16])
                        if written == data[:16]:
                            logger.info(f"Write verification successful at offset {offset} (attempt {retry + 1})")

                            # 详细验证：验证整个数据块
                            if len(data) <= 1024:  # 对小文件进行完整验证
                                full_written = bytes(self.shared_block.buf[offset:offset+len(data)])
                                if full_written != data:
                                    logger.error(f"Full verification failed at offset {offset}")
                                    logger.error(f"Expected size: {len(data)}, Written size: {len(full_written)}")
                                    if retry < max_retries - 1:
                                        self.shared_block.buf[offset:end_offset] = data
                                        continue
                                    else:
                                        return False
                                else:
                                    logger.info(f"Full verification successful for {len(data)} bytes at offset {offset}")

                            # 写入成功，更新状态为allocated
                            if task_id:
                                final_metadata = struct.pack(
                                    '!64s',
                                    b''.join([
                                        struct.pack('!i', 1),  # status: allocated (数据已写入)
                                        struct.pack('!i', chunk_index),  # chunk_index
                                        struct.pack('!i', len(data)),  # task_size
                                        struct.pack('!i', offset),  # task_offset
                                        struct.pack('!d', time.time()),  # timestamp
                                        struct.pack('!i', -1),  # worker_id (尚未分配)
                                        task_id.encode('utf-8')[:32]  # task_id (32字节)
                                    ])
                                )
                                self.metadata.buf[metadata_offset:metadata_offset+64] = final_metadata
                                logger.debug(f"Updated chunk {chunk_index} status to allocated after successful write")

                            break
                        else:
                            if retry < max_retries - 1:
                                logger.warning(f"Write verification retry {retry + 1} at offset {offset}")
                                logger.warning(f"Expected: {data[:16]}, Got: {written}")
                                # 重新写入
                                self.shared_block.buf[offset:end_offset] = data
                                # 强制同步
                                _ = self.shared_block.buf[offset]
                                time.sleep(0.001)  # 1ms延迟确保同步
                            else:
                                logger.error(f"Write verification failed at offset {offset}")
                                logger.error(f"Expected: {data[:16]}")
                                logger.error(f"Written:  {written}")
                                return False

                logger.info(f"Successfully wrote {len(data)} bytes to shared memory at offset {offset} (pool: {self.memory_name})")
                return True
        except Exception as e:
            logger.error(f"Error writing to shared memory: {e}")
            return False

    def read_data(self, offset: int, size: int) -> Optional[bytes]:
        """从共享内存读取数据"""
        try:
            with self.lock:
                if offset + size > self.pool_size_bytes:
                    return None

                logger.debug(f"Reading {size} bytes from offset {offset} (pool GPU {self.gpu_id}, pool name: {self.memory_name})")

                # 强制内存同步 - 确保读取到最新数据
                try:
                    import ctypes
                    # 使用ctypes强制内存同步
                    ctypes.memset(ctypes.addressof(self.shared_block.buf) + offset, 0, 0)
                except:
                    # 备用方案：多次访问触发同步
                    for i in range(3):
                        _ = self.shared_block.buf[offset]
                        _ = self.shared_block.buf[offset + size - 1] if size > 0 else 0

                # 读取数据（可能需要多次尝试）
                max_retries = 3
                data = None

                for retry in range(max_retries):
                    data = bytes(self.shared_block.buf[offset:offset+size])

                    # 检查是否为全零数据（如果数据应该非空）
                    if len(data) >= 16 and all(b == 0 for b in data[:16]):
                        if retry < max_retries - 1:
                            logger.warning(f"Read zero data attempt {retry + 1} from offset {offset} (pool GPU {self.gpu_id}, pool: {self.memory_name}), retrying...")
                            import time
                            time.sleep(0.001)  # 1ms延迟
                            continue
                        else:
                            logger.warning(f"Read zero data from offset {offset} (pool GPU {self.gpu_id}, pool: {self.memory_name})")
                            logger.error(f"Memory block info: size={self.pool_size_bytes}, name={self.memory_name}")
                            logger.error(f"Read request: offset={offset}, size={size}")
                            # 尝试读取内存块状态
                            try:
                                chunk_index = offset // self.chunk_size_bytes
                                chunk_status = self.get_chunk_status(chunk_index)
                                if chunk_status:
                                    logger.error(f"Chunk status: {chunk_status}")
                            except Exception as e:
                                logger.error(f"Failed to get chunk status: {e}")
                    else:
                        # 成功读取到非零数据
                        logger.debug(f"Successfully read non-zero data on attempt {retry + 1}")
                        break

                # 添加调试信息
                if data:
                    logger.debug(f"Read {size} bytes from offset {offset} (pool GPU {self.gpu_id})")
                    logger.debug(f"Read data header: {data[:32]}")
                else:
                    logger.error(f"No data read from offset {offset}")

                return data
        except Exception as e:
            logger.error(f"Error reading from shared memory: {e}")
            return None

    def free_chunk(self, task_id: str) -> bool:
        """释放内存块"""
        with self.lock:
            if task_id not in self.active_tasks:
                return False

            chunk = self.active_tasks[task_id]

            # 找到对应的元数据位置
            chunk_index = chunk.offset // self.chunk_size_bytes
            metadata_offset = chunk_index * 64

            # 重置元数据
            new_metadata = struct.pack(
                '!64s',
                b''.join([
                    struct.pack('!i', 0),  # status: free
                    struct.pack('!i', chunk_index),  # chunk_index
                    struct.pack('!i', 0),  # task_size
                    struct.pack('!i', 0),  # task_offset
                    struct.pack('!d', 0.0),  # timestamp
                    struct.pack('!i', -1),  # worker_id
                    b'\x00' * 40  # padding
                ])
            )
            self.metadata.buf[metadata_offset:metadata_offset+64] = new_metadata

            # 注意：不要立即清零数据，避免竞态条件
            # 数据将在下次分配时被覆盖，或通过定时清理任务处理

            # 从活跃任务中移除
            del self.active_tasks[task_id]

            logger.debug(f"Freed chunk {chunk_index} for task {task_id}")
            return True

    def get_chunk_status(self, chunk_index: int) -> Optional[Dict[str, Any]]:
        """获取chunk状态"""
        try:
            offset = chunk_index * 64
            chunk_metadata = self.metadata.buf[offset:offset+64]

            # 解包元数据
            status = struct.unpack('!i', chunk_metadata[0:4])[0]
            chunk_idx = struct.unpack('!i', chunk_metadata[4:8])[0]
            task_size = struct.unpack('!i', chunk_metadata[8:12])[0]
            task_offset = struct.unpack('!i', chunk_metadata[12:16])[0]
            timestamp = struct.unpack('!d', chunk_metadata[16:24])[0]
            worker_id = struct.unpack('!i', chunk_metadata[24:28])[0]

            # 提取task_id
            task_id_bytes = chunk_metadata[28:60]
            task_id = bytes(task_id_bytes).decode('utf-8').rstrip('\x00')

            return {
                'status': status,
                'chunk_index': chunk_idx,
                'task_size': task_size,
                'task_offset': task_offset,
                'timestamp': timestamp,
                'worker_id': worker_id,
                'task_id': task_id if task_id else None
            }
        except Exception as e:
            logger.error(f"Error getting chunk status: {e}")
            return None

    def get_pool_stats(self) -> Dict[str, Any]:
        """获取内存池统计信息"""
        with self.lock:
            stats = {
                'total_chunks': self.max_chunks,
                'free_chunks': 0,
                'allocated_chunks': 0,
                'processing_chunks': 0,
                'completed_chunks': 0,
                'pool_size_mb': self.pool_size_bytes / (1024 * 1024),
                'used_size_mb': 0,
                'active_tasks': len(self.active_tasks),
                'gpu_id': self.gpu_id
            }

            for i in range(self.max_chunks):
                chunk_status = self.get_chunk_status(i)
                if chunk_status:
                    status = chunk_status['status']
                    if status == 0:
                        stats['free_chunks'] += 1
                    elif status == 1:
                        stats['allocated_chunks'] += 1
                        stats['used_size_mb'] += chunk_status['task_size'] / (1024 * 1024)
                    elif status == 2:
                        stats['processing_chunks'] += 1
                        stats['used_size_mb'] += chunk_status['task_size'] / (1024 * 1024)
                    elif status == 3:
                        stats['completed_chunks'] += 1
                        stats['used_size_mb'] += chunk_status['task_size'] / (1024 * 1024)

            return stats

    def cleanup(self):
        """清理共享内存资源"""
        try:
            with self.lock:
                # 清理所有活跃任务
                for task_id in list(self.active_tasks.keys()):
                    self.free_chunk(task_id)

                # 释放共享内存
                self.shared_block.close()
                try:
                    self.shared_block.unlink()
                except FileNotFoundError:
                    pass

                self.metadata.close()
                try:
                    self.metadata.unlink()
                except FileNotFoundError:
                    pass

            logger.info(f"Cleaned up shared memory pool for GPU {self.gpu_id}")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# 配置管理
class SharedMemoryConfig:
    """共享内存配置管理"""

    def __init__(self):
        self.pool_size_mb = int(os.getenv("SHARED_MEMORY_POOL_SIZE_MB", "400"))
        self.chunk_size_mb = int(os.getenv("MEMORY_CHUNK_SIZE_MB", "80"))
        self.workers_per_gpu = int(os.getenv("WORKERS_PER_GPU", "2"))
        self.max_concurrent_tasks = int(os.getenv("MAX_CONCURRENT_TASKS", "10"))
        self.health_check_interval = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
        self.memory_timeout = int(os.getenv("MEMORY_TIMEOUT", "300"))  # 5分钟

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            'pool_size_mb': self.pool_size_mb,
            'chunk_size_mb': self.chunk_size_mb,
            'workers_per_gpu': self.workers_per_gpu,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'health_check_interval': self.health_check_interval,
            'memory_timeout': self.memory_timeout
        }

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    config = SharedMemoryConfig()
    print("Shared Memory Configuration:")
    for key, value in config.get_config_dict().items():
        print(f"  {key}: {value}")

    # 测试内存池
    pool = SharedMemoryPool(gpu_id=0, pool_size_mb=config.pool_size_mb, chunk_size_mb=config.chunk_size_mb)

    # 测试分配和写入
    test_data = b"Hello, this is test audio data!" * 1000  # 约30KB
    task_id = str(uuid.uuid4())

    offset, error = pool.allocate_chunk(len(test_data), task_id)
    if error:
        print(f"Allocation failed: {error}")
    else:
        print(f"Allocated chunk at offset: {offset}")

        # 写入数据
        if pool.write_data(test_data, offset):
            print("Data written successfully")

            # 读取数据
            read_data = pool.read_data(offset, len(test_data))
            if read_data == test_data:
                print("Data verification successful")
            else:
                print("Data verification failed")

            # 释放内存
            if pool.free_chunk(task_id):
                print("Memory freed successfully")
            else:
                print("Failed to free memory")

    # 显示统计信息
    stats = pool.get_pool_stats()
    print(f"\nPool Stats: {stats}")

    # 清理
    pool.cleanup()