#!/usr/bin/env python3
"""
测试共享内存架构
Test Shared Memory Architecture
"""

import asyncio
import logging
import time
import uuid
from master_process import MasterProcess

logger = logging.getLogger(__name__)

async def test_master_process():
    """测试主进程"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Starting shared memory architecture test...")

    # 创建主进程
    master = MasterProcess()

    # 启动主进程（在后台）
    master_task = asyncio.create_task(master.start())

    # 等待工作进程启动
    await asyncio.sleep(5)

    # 测试提交任务
    logger.info("Testing task submission...")

    test_tasks = []
    for i in range(3):
        task_data = {
            'client_id': f'test_client_{i}',
            'audio_size': 1024 * 100,  # 100KB
            'metadata': {
                'language': 'auto',
                'response_format': 'json'
            }
        }

        success, task_id = await master.submit_task(task_data)
        if success:
            logger.info(f"Task {i+1} submitted successfully: {task_id}")
            test_tasks.append(task_id)
        else:
            logger.error(f"Task {i+1} submission failed: {task_id}")

    # 等待任务处理
    await asyncio.sleep(10)

    # 检查结果
    logger.info("Checking task results...")
    for task_id in test_tasks:
        if task_id in master.tasks:
            task = master.tasks[task_id]
            logger.info(f"Task {task_id}: status={task.status}, result={task.result_data}")
        else:
            logger.warning(f"Task {task_id} not found in master tasks")

    # 获取统计信息
    stats = master.get_stats()
    logger.info(f"Final stats: {stats}")

    # 关闭主进程
    logger.info("Shutting down master process...")
    master.shutdown_event.set()

    # 等待主进程关闭
    await master_task

    logger.info("Test completed!")

if __name__ == "__main__":
    asyncio.run(test_master_process())