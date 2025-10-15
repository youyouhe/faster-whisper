#!/usr/bin/env python3
"""
主进程管理器 - 单端口多进程架构的核心调度组件
Master Process Manager for Single-Port Multi-Process Architecture
"""

import asyncio
import logging
import multiprocessing as mp
import os
import signal
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from pathlib import Path

from shared_memory_manager import SharedMemoryPool, SharedMemoryConfig

logger = logging.getLogger(__name__)

@dataclass
class WorkerInfo:
    """工作进程信息"""
    worker_id: int
    pid: int
    gpu_id: int
    process: mp.Process
    task_queue: mp.Queue
    result_queue: mp.Queue
    status: str  # 'idle', 'busy', 'dead', 'starting'
    last_heartbeat: float
    task_count: int = 0
    error_count: int = 0
    memory_usage: float = 0.0
    queue_size: int = 0

@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    client_id: str
    audio_size: int
    status: str  # 'pending', 'processing', 'completed', 'failed'
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    worker_id: Optional[int] = None
    memory_offset: Optional[int] = None
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0

class MasterProcess:
    """主进程管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()

        # 进程管理
        self.workers: Dict[int, WorkerInfo] = {}
        self.next_worker_id = 0

        # 任务管理
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_queue = asyncio.Queue(maxsize=self.config['max_queue_size'])
        self.result_queue = asyncio.Queue(maxsize=self.config['max_queue_size'])

        # 共享内存池
        self.memory_pools: Dict[int, SharedMemoryPool] = {}
        self.memory_config = SharedMemoryConfig()

        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'active_workers': 0,
            'uptime': time.time(),
            'avg_processing_time': 0.0,
            'memory_usage': 0.0
        }

        # 控制标志
        self.running = False
        self.shutdown_event = asyncio.Event()

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 初始化共享内存池
        self._init_memory_pools()

        # 设置信号处理
        self._setup_signal_handlers()

        logger.info("Master process initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'workers_per_gpu': int(os.getenv('WORKERS_PER_GPU', '2')),
            'max_queue_size': int(os.getenv('MAX_QUEUE_SIZE', '100')),
            'health_check_interval': int(os.getenv('HEALTH_CHECK_INTERVAL', '30')),
            'task_timeout': int(os.getenv('TASK_TIMEOUT', '300')),
            'max_retries': int(os.getenv('MAX_RETRIES', '3')),
            'gpu_memory_fraction': float(os.getenv('GPU_MEMORY_FRACTION', '0.8')),
            'log_level': os.getenv('LOG_LEVEL', 'INFO')
        }

    def _init_memory_pools(self):
        """初始化共享内存池"""
        gpu_count = self._get_gpu_count()
        for gpu_id in range(gpu_count):
            try:
                pool = SharedMemoryPool(
                    gpu_id=gpu_id,
                    pool_size_mb=self.memory_config.pool_size_mb,
                    chunk_size_mb=self.memory_config.chunk_size_mb
                )
                self.memory_pools[gpu_id] = pool
                logger.info(f"Initialized memory pool for GPU {gpu_id}")
            except Exception as e:
                logger.error(f"Failed to initialize memory pool for GPU {gpu_id}: {e}")

    def _get_gpu_count(self) -> int:
        """获取GPU数量"""
        # 先专注于单GPU测试
        return 1
        # try:
        #     import torch
        #     return torch.cuda.device_count()
        # except ImportError:
        #     logger.warning("PyTorch not available, assuming single GPU")
        #     return 1

    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def start(self):
        """启动主进程"""
        logger.info("Starting master process...")
        self.running = True

        # 启动工作进程
        await self._start_workers()

        # 启动后台任务
        tasks = [
            asyncio.create_task(self._task_dispatcher()),
            asyncio.create_task(self._result_handler()),
            asyncio.create_task(self._health_checker()),
            asyncio.create_task(self._stats_reporter()),
            asyncio.create_task(self._memory_cleaner())
        ]

        try:
            # 等待关闭信号
            await self.shutdown_event.wait()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            # 停止所有任务
            for task in tasks:
                task.cancel()

            # 关闭主进程
            await self._shutdown()

    async def _start_workers(self):
        """启动工作进程"""
        gpu_count = self._get_gpu_count()
        workers_per_gpu = self.config['workers_per_gpu']

        for gpu_id in range(gpu_count):
            for worker_idx in range(workers_per_gpu):
                worker_id = self.next_worker_id
                self.next_worker_id += 1

                success = await self._start_worker(worker_id, gpu_id)
                if success:
                    logger.info(f"Started worker {worker_id} on GPU {gpu_id}")
                else:
                    logger.error(f"Failed to start worker {worker_id} on GPU {gpu_id}")

    async def _start_worker(self, worker_id: int, gpu_id: int) -> bool:
        """启动单个工作进程"""
        try:
            # 创建工作进程通信队列
            worker_queue = mp.Queue(maxsize=50)
            result_queue = mp.Queue(maxsize=50)

            # 启动工作进程
            process = mp.Process(
                target=self._worker_process_main,
                args=(worker_id, gpu_id, worker_queue, result_queue)
            )
            process.start()

            # 记录工作进程信息，保存队列引用
            worker_info = WorkerInfo(
                worker_id=worker_id,
                pid=process.pid,
                gpu_id=gpu_id,
                process=process,
                task_queue=worker_queue,
                result_queue=result_queue,
                status='starting',
                last_heartbeat=time.time()
            )

            self.workers[worker_id] = worker_info

            # 等待工作进程启动
            await asyncio.sleep(2)  # 增加等待时间，确保模型加载完成

            # 检查进程是否正常运行
            if process.is_alive():
                worker_info.status = 'idle'
                self.stats['active_workers'] += 1
                logger.info(f"Worker {worker_id} started successfully (PID: {process.pid})")
                return True
            else:
                logger.error(f"Worker {worker_id} failed to start")
                return False

        except Exception as e:
            logger.error(f"Error starting worker {worker_id}: {e}")
            return False

    def _worker_process_main(self, worker_id: int, gpu_id: int,
                            task_queue: mp.Queue, result_queue: mp.Queue):
        """工作进程主函数"""
        try:
            from worker_process import worker_main
            worker_main(worker_id, gpu_id, task_queue, result_queue)
        except Exception as e:
            logger.error(f"Worker {worker_id} fatal error: {e}")

    async def submit_task(self, task_data: Dict[str, Any]) -> Tuple[bool, str]:
        """提交任务"""
        try:
            task_id = str(uuid.uuid4())

            # 创建任务信息
            task_info = TaskInfo(
                task_id=task_id,
                client_id=task_data.get('client_id', 'unknown'),
                audio_size=task_data.get('audio_size', 0),
                status='pending',
                created_at=time.time()
            )

            self.tasks[task_id] = task_info

            # 分配共享内存
            memory_pool = self._get_best_memory_pool()
            if memory_pool:
                offset, error = memory_pool.allocate_chunk(
                    task_info.audio_size, task_id
                )
                if error:
                    task_info.status = 'failed'
                    task_info.error_message = f"Memory allocation failed: {error}"
                    return False, error
                task_info.memory_offset = offset

            # 添加到任务队列
            await self.task_queue.put({
                'task_id': task_id,
                'task_data': task_data,
                'memory_pool_gpu': memory_pool.gpu_id if memory_pool else None,
                'memory_offset': offset,
                'audio_size': task_info.audio_size
            })

            self.stats['total_tasks'] += 1
            logger.info(f"Task {task_id} submitted successfully")

            return True, task_id

        except Exception as e:
            logger.error(f"Error submitting task: {e}")
            return False, str(e)

    async def _task_dispatcher(self):
        """任务分发器"""
        while self.running:
            try:
                # 获取任务
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )

                # 找到空闲工作进程
                worker = self._find_idle_worker()
                if worker:
                    await self._assign_task_to_worker(task, worker)
                else:
                    # 没有空闲工作进程，重新放回队列
                    await self.task_queue.put(task)
                    await asyncio.sleep(0.1)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in task dispatcher: {e}")

    async def _assign_task_to_worker(self, task: Dict[str, Any], worker: WorkerInfo):
        """分配任务给工作进程"""
        try:
            task_id = task['task_id']

            # 更新任务状态
            if task_id in self.tasks:
                self.tasks[task_id].status = 'processing'
                self.tasks[task_id].started_at = time.time()
                self.tasks[task_id].worker_id = worker.worker_id

            # 更新工作进程状态
            worker.status = 'busy'
            worker.last_heartbeat = time.time()

            # 真实分配任务给工作进程
            logger.info(f"Assigning task {task_id} to worker {worker.worker_id}")

            # 将任务放入工作进程的队列
            try:
                worker.task_queue.put(task, timeout=5.0)
                logger.info(f"Task {task_id} successfully queued for worker {worker.worker_id}")
            except Exception as e:
                logger.error(f"Failed to queue task {task_id} for worker {worker.worker_id}: {e}")
                # 任务分配失败，恢复状态
                worker.status = 'idle'
                if task_id in self.tasks:
                    self.tasks[task_id].status = 'pending'
                    self.tasks[task_id].worker_id = None
                return

            # 启动结果监听任务（为这个特定任务）
            asyncio.create_task(self._listen_for_worker_result(worker, task_id))

        except Exception as e:
            logger.error(f"Error assigning task to worker: {e}")
            # 恢复状态
            worker.status = 'idle'
            if task_id in self.tasks:
                self.tasks[task_id].status = 'pending'
                self.tasks[task_id].worker_id = None

    async def _listen_for_worker_result(self, worker: WorkerInfo, task_id: str):
        """监听特定工作进程的任务结果"""
        try:
            timeout = 300  # 5分钟超时
            start_time = time.time()

            while time.time() - start_time < timeout:
                try:
                    # 非阻塞检查结果队列
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self._get_result_from_queue, worker.result_queue, 0.1
                    )

                    if result and result.get('task_id') == task_id:
                        # 收到结果，放入主结果队列
                        await self.result_queue.put(result)
                        logger.info(f"Received result for task {task_id} from worker {worker.worker_id}")
                        return

                except Exception as e:
                    if "Empty" not in str(e):
                        logger.error(f"Error checking result from worker {worker.worker_id}: {e}")

                await asyncio.sleep(0.1)

            # 超时处理
            logger.warning(f"Task {task_id} timeout for worker {worker.worker_id}")
            timeout_result = {
                'task_id': task_id,
                'worker_id': worker.worker_id,
                'status': 'failed',
                'error': 'Task timeout',
                'processing_time': timeout
            }
            await self.result_queue.put(timeout_result)

        except Exception as e:
            logger.error(f"Error listening for result from worker {worker.worker_id}: {e}")
            error_result = {
                'task_id': task_id,
                'worker_id': worker.worker_id,
                'status': 'failed',
                'error': f'Result listening error: {str(e)}',
                'processing_time': time.time() - self.tasks[task_id].started_at if task_id in self.tasks else 0
            }
            await self.result_queue.put(error_result)

    def _get_result_from_queue(self, queue: mp.Queue, timeout: float):
        """从队列获取结果"""
        try:
            return queue.get(timeout=timeout)
        except:
            return None

    async def _result_handler(self):
        """结果处理器"""
        while self.running:
            try:
                result = await asyncio.wait_for(
                    self.result_queue.get(),
                    timeout=1.0
                )

                await self._process_result(result)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in result handler: {e}")

    async def _process_result(self, result: Dict[str, Any]):
        """处理任务结果"""
        try:
            task_id = result['task_id']

            if task_id in self.tasks:
                task_info = self.tasks[task_id]
                task_info.status = result['status']
                task_info.completed_at = time.time()
                task_info.result_data = result.get('result')

                if result['status'] == 'completed':
                    self.stats['completed_tasks'] += 1

                    # 释放共享内存
                    if task_info.memory_offset is not None:
                        pool = self._get_memory_pool_by_gpu(result.get('gpu_id', 0))
                        if pool:
                            pool.free_chunk(task_id)

                elif result['status'] == 'failed':
                    self.stats['failed_tasks'] += 1
                    task_info.error_message = result.get('error', 'Unknown error')

                # 更新工作进程状态
                worker_id = result.get('worker_id')
                if worker_id in self.workers:
                    self.workers[worker_id].status = 'idle'
                    # 增加Worker的任务计数
                    if result['status'] == 'completed':
                        self.workers[worker_id].task_count += 1
                    elif result['status'] == 'failed':
                        self.workers[worker_id].error_count += 1

                logger.info(f"Processed result for task {task_id}")

        except Exception as e:
            logger.error(f"Error processing result: {e}")

    def _find_idle_worker(self) -> Optional[WorkerInfo]:
        """找到空闲工作进程 - 使用负载均衡算法"""
        idle_workers = [worker for worker in self.workers.values()
                       if worker.status == 'idle' and worker.process.is_alive()]

        if not idle_workers:
            return None

        # 选择任务数最少的工作进程（负载均衡）
        return min(idle_workers, key=lambda w: w.task_count)

    def _get_best_memory_pool(self) -> Optional[SharedMemoryPool]:
        """获取最佳共享内存池"""
        if not self.memory_pools:
            return None

        # 简单轮询策略，后续可以实现更智能的分配
        available_pools = [pool for pool in self.memory_pools.values()
                          if pool.get_pool_stats()['free_chunks'] > 0]

        if available_pools:
            # 选择GPU ID最小的可用池
            return min(available_pools, key=lambda p: p.gpu_id)

        return None

    def _get_memory_pool_by_gpu(self, gpu_id: int) -> Optional[SharedMemoryPool]:
        """根据GPU ID获取内存池"""
        return self.memory_pools.get(gpu_id)

    async def _health_checker(self):
        """健康检查器"""
        while self.running:
            try:
                await asyncio.sleep(self.config['health_check_interval'])

                dead_workers = []
                for worker_id, worker in self.workers.items():
                    if not worker.process.is_alive():
                        dead_workers.append(worker_id)
                        logger.warning(f"Worker {worker_id} died")

                # 重启死亡的工作进程
                for worker_id in dead_workers:
                    await self._restart_worker(worker_id)

            except Exception as e:
                logger.error(f"Error in health checker: {e}")

    async def _restart_worker(self, worker_id: int):
        """重启工作进程"""
        try:
            if worker_id in self.workers:
                old_worker = self.workers[worker_id]
                gpu_id = old_worker.gpu_id

                # 清理旧进程
                if old_worker.process.is_alive():
                    old_worker.process.terminate()
                    old_worker.process.join(timeout=5)

                # 启动新进程
                success = await self._start_worker(worker_id, gpu_id)
                if success:
                    logger.info(f"Successfully restarted worker {worker_id}")
                else:
                    logger.error(f"Failed to restart worker {worker_id}")

        except Exception as e:
            logger.error(f"Error restarting worker {worker_id}: {e}")

    async def _stats_reporter(self):
        """统计报告器"""
        while self.running:
            try:
                await asyncio.sleep(60)  # 每分钟报告一次

                stats = self.get_stats()
                logger.info(f"Master process stats: {json.dumps(stats, indent=2)}")

            except Exception as e:
                logger.error(f"Error in stats reporter: {e}")

    async def _memory_cleaner(self):
        """内存清理器"""
        while self.running:
            try:
                await asyncio.sleep(300)  # 每5分钟清理一次

                current_time = time.time()
                timeout = self.memory_config.memory_timeout

                # 清理超时的任务内存
                for task_id, task_info in list(self.tasks.items()):
                    if (task_info.status in ['processing', 'pending'] and
                        current_time - task_info.created_at > timeout):

                        # 释放内存
                        if task_info.memory_offset is not None:
                            for pool in self.memory_pools.values():
                                if pool.free_chunk(task_id):
                                    break

                        # 标记任务失败
                        task_info.status = 'failed'
                        task_info.error_message = 'Task timeout'
                        self.stats['failed_tasks'] += 1

                        logger.warning(f"Cleaned up timeout task {task_id}")

            except Exception as e:
                logger.error(f"Error in memory cleaner: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        current_time = time.time()
        uptime = current_time - self.stats['uptime']

        # 计算平均处理时间
        completed_tasks = [t for t in self.tasks.values() if t.status == 'completed']
        avg_time = 0.0
        if completed_tasks:
            total_time = sum(t.completed_at - t.started_at for t in completed_tasks
                           if t.started_at and t.completed_at)
            avg_time = total_time / len(completed_tasks)

        # 内存使用统计
        total_memory_mb = 0
        used_memory_mb = 0
        for pool in self.memory_pools.values():
            pool_stats = pool.get_pool_stats()
            total_memory_mb += pool_stats['pool_size_mb']
            used_memory_mb += pool_stats['used_size_mb']

        return {
            'uptime_seconds': uptime,
            'total_tasks': self.stats['total_tasks'],
            'completed_tasks': self.stats['completed_tasks'],
            'failed_tasks': self.stats['failed_tasks'],
            'active_workers': self.stats['active_workers'],
            'pending_tasks': len([t for t in self.tasks.values() if t.status == 'pending']),
            'processing_tasks': len([t for t in self.tasks.values() if t.status == 'processing']),
            'avg_processing_time': avg_time,
            'total_memory_mb': total_memory_mb,
            'used_memory_mb': used_memory_mb,
            'memory_utilization': used_memory_mb / total_memory_mb if total_memory_mb > 0 else 0,
            'workers': {
                str(w.worker_id): {
                    'status': w.status,
                    'pid': w.pid,
                    'gpu_id': w.gpu_id,
                    'task_count': w.task_count,
                    'error_count': w.error_count,
                    'last_heartbeat': w.last_heartbeat
                } for w in self.workers.values()
            }
        }

    async def _shutdown(self):
        """关闭主进程"""
        logger.info("Shutting down master process...")
        self.running = False

        # 通知所有工作进程停止
        for worker_id, worker in self.workers.items():
            try:
                if worker.process.is_alive():
                    # 向工作进程发送停止信号
                    worker.task_queue.put(None, timeout=1.0)
                    logger.info(f"Sent stop signal to worker {worker_id}")
            except Exception as e:
                logger.error(f"Error sending stop signal to worker {worker_id}: {e}")

        # 等待一段时间让工作进程处理完当前任务
        await asyncio.sleep(1)

        # 关闭所有工作进程
        for worker in self.workers.values():
            try:
                if worker.process.is_alive():
                    worker.process.terminate()
                    worker.process.join(timeout=5)
                    if worker.process.is_alive():
                        worker.process.kill()
                        worker.process.join(timeout=2)
            except Exception as e:
                logger.error(f"Error stopping worker {worker.worker_id}: {e}")

        # 清理共享内存
        for pool in self.memory_pools.values():
            try:
                pool.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up memory pool: {e}")

        # 关闭线程池
        self.executor.shutdown(wait=True)

        logger.info("Master process shutdown complete")

async def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    master = MasterProcess()
    await master.start()

if __name__ == "__main__":
    asyncio.run(main())