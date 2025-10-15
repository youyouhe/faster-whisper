#!/usr/bin/env python3
"""
测试工作进程模型加载
Test Worker Process Model Loading
"""

import sys
import os
import logging
sys.path.append('.')

from worker_process import WorkerProcess, WorkerConfig
from multiprocessing import Queue

logging.basicConfig(level=logging.INFO)

def test_worker_model():
    """测试工作进程模型加载"""
    print("=== 测试工作进程模型加载 ===")

    # 创建队列
    task_queue = Queue()
    result_queue = Queue()

    # 创建工作进程配置
    config = WorkerConfig(
        worker_id=0,
        gpu_id=0,
        model_path="tiny",
        compute_type="float32"
    )

    # 创建工作进程
    worker = WorkerProcess(
        worker_id=0,
        gpu_id=0,
        task_queue=task_queue,
        result_queue=result_queue,
        config=config
    )

    # 测试模型初始化
    print("1. 测试模型初始化...")
    model_loaded = worker._init_model()
    print(f"   模型加载结果: {model_loaded}")

    if worker.model:
        print(f"   模型对象: {type(worker.model)}")
        print(f"   模型路径: {config.model_path}")
        print(f"   计算类型: {config.compute_type}")
    else:
        print("   模型对象为空")

    # 测试共享内存初始化
    print("\n2. 测试共享内存初始化...")
    memory_loaded = worker._init_memory_pool()
    print(f"   共享内存初始化结果: {memory_loaded}")

    # 清理
    if worker.memory_pool:
        worker.memory_pool.cleanup()

    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_worker_model()