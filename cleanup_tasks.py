#!/usr/bin/env python3
"""
定时清理任务脚本
每天清理过期、僵死的任务，保持系统健康运行
"""

import os
import sqlite3
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 配置
DATABASE_PATH = os.getenv("DATABASE_PATH", "/home/cat/faster-whisper/data/tasks.db")
CLEANUP_INTERVAL_HOURS = int(os.getenv("CLEANUP_INTERVAL_HOURS", "24"))  # 清理间隔
MAX_AGE_DAYS = int(os.getenv("MAX_AGE_DAYS", "7"))  # 任务最大保留天数
STALE_TASK_HOURS = int(os.getenv("STALE_TASK_HOURS", "2"))  # 僵死任务阈值（处理中超过这个时间的任务）

class TaskCleaner:
    """任务清理器"""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = db_path

    def _get_db_connection(self):
        """获取数据库连接"""
        return sqlite3.connect(self.db_path)

    def cleanup_old_tasks(self):
        """清理超过最大保留天数的已完成和失败任务"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)
            cutoff_str = cutoff_date.strftime('%Y-%m-%dT%H:%M:%S+00:00')

            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                # 清理已完成的旧任务
                cursor.execute("""
                    DELETE FROM tasks
                    WHERE status IN ('completed', 'failed')
                    AND updated_at < ?
                """, (cutoff_str,))

                completed_deleted = cursor.rowcount

                # 删除相关的SRT文件
                cursor.execute("""
                    SELECT srt_file_path FROM tasks
                    WHERE status IN ('completed', 'failed')
                    AND updated_at < ?
                    AND srt_file_path IS NOT NULL
                """, (cutoff_str,))

                deleted_files = 0
                for (file_path,) in cursor.fetchall():
                    try:
                        if Path(file_path).exists():
                            os.remove(file_path)
                            deleted_files += 1
                            logger.debug(f"删除SRT文件: {file_path}")
                    except Exception as e:
                        logger.warning(f"删除SRT文件失败 {file_path}: {e}")

                conn.commit()

                logger.info(f"清理完成: 删除了 {completed_deleted} 个旧任务记录, {deleted_files} 个SRT文件")
                return completed_deleted

        except Exception as e:
            logger.error(f"清理旧任务失败: {e}")
            return 0

    def cleanup_stale_tasks(self):
        """清理僵死的处理中任务"""
        try:
            stale_time = datetime.now(timezone.utc) - timedelta(hours=STALE_TASK_HOURS)
            stale_str = stale_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')

            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                # 查找僵死的处理中任务
                cursor.execute("""
                    SELECT task_id, filename, created_at, updated_at
                    FROM tasks
                    WHERE status = 'processing'
                    AND updated_at < ?
                """, (stale_str,))

                stale_tasks = cursor.fetchall()

                if not stale_tasks:
                    logger.info("没有发现僵死的处理任务")
                    return 0

                # 将僵死任务标记为失败
                cursor.execute("""
                    UPDATE tasks
                    SET status = 'failed',
                        error_message = 'Task marked as failed due to timeout (stale task cleanup)',
                        updated_at = ?
                    WHERE status = 'processing'
                    AND updated_at < ?
                """, (datetime.now(timezone.utc).isoformat(), stale_str))

                updated_count = cursor.rowcount
                conn.commit()

                logger.warning(f"发现并处理了 {updated_count} 个僵死任务:")
                for task_id, filename, created_at, updated_at in stale_tasks:
                    logger.warning(f"  - 任务 {task_id} ({filename}) 创建于 {created_at}, 最后更新 {updated_at}")

                return updated_count

        except Exception as e:
            logger.error(f"清理僵死任务失败: {e}")
            return 0

    def cleanup_pending_upload_tasks(self):
        """清理长时间等待上传的任务"""
        try:
            stale_time = datetime.now(timezone.utc) - timedelta(hours=STALE_TASK_HOURS)
            stale_str = stale_time.strftime('%Y-%m-%dT%H:%M:%S+00:00')

            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                # 查找长时间等待上传的任务
                cursor.execute("""
                    SELECT COUNT(*) FROM tasks
                    WHERE status = 'pending_upload'
                    AND created_at < ?
                """, (stale_str,))

                count = cursor.fetchone()[0]

                if count > 0:
                    # 删除长时间等待上传的任务
                    cursor.execute("""
                        DELETE FROM tasks
                        WHERE status = 'pending_upload'
                        AND created_at < ?
                    """, (stale_str,))

                    deleted_count = cursor.rowcount
                    conn.commit()

                    logger.warning(f"删除了 {deleted_count} 个长时间等待上传的任务")
                    return deleted_count

                return 0

        except Exception as e:
            logger.error(f"清理等待上传任务失败: {e}")
            return 0

    def get_task_statistics(self):
        """获取任务统计信息"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()

                # 统计各状态任务数量
                cursor.execute("""
                    SELECT status, COUNT(*)
                    FROM tasks
                    GROUP BY status
                    ORDER BY COUNT(*) DESC
                """)

                status_counts = dict(cursor.fetchall())

                # 获取总任务数
                cursor.execute("SELECT COUNT(*) FROM tasks")
                total_tasks = cursor.fetchone()[0]

                # 获取最近24小时的任务数
                yesterday = datetime.now(timezone.utc) - timedelta(hours=24)
                yesterday_str = yesterday.strftime('%Y-%m-%dT%H:%M:%S+00:00')

                cursor.execute("SELECT COUNT(*) FROM tasks WHERE created_at > ?", (yesterday_str,))
                recent_tasks = cursor.fetchone()[0]

                return {
                    'total_tasks': total_tasks,
                    'recent_tasks_24h': recent_tasks,
                    'status_counts': status_counts
                }

        except Exception as e:
            logger.error(f"获取任务统计失败: {e}")
            return {}

    def run_cleanup(self):
        """运行完整的清理流程"""
        logger.info("=" * 50)
        logger.info(f"开始执行任务清理 - {datetime.now().isoformat()}")
        logger.info("=" * 50)

        # 获取清理前统计
        stats_before = self.get_task_statistics()
        logger.info(f"清理前统计: 总任务 {stats_before.get('total_tasks', 0)}, 状态分布: {stats_before.get('status_counts', {})}")

        # 执行清理操作
        old_tasks_deleted = self.cleanup_old_tasks()
        stale_tasks_handled = self.cleanup_stale_tasks()
        pending_upload_deleted = self.cleanup_pending_upload_tasks()

        # 获取清理后统计
        stats_after = self.get_task_statistics()
        logger.info(f"清理后统计: 总任务 {stats_after.get('total_tasks', 0)}, 状态分布: {stats_after.get('status_counts', {})}")

        # 总结
        total_cleaned = old_tasks_deleted + stale_tasks_handled + pending_upload_deleted
        logger.info("=" * 50)
        logger.info(f"清理完成! 总共处理了 {total_cleaned} 个任务:")
        logger.info(f"  - 删除旧任务: {old_tasks_deleted}")
        logger.info(f"  - 处理僵死任务: {stale_tasks_handled}")
        logger.info(f"  - 删除等待上传任务: {pending_upload_deleted}")
        logger.info("=" * 50)

        return total_cleaned

def main():
    """主函数"""
    db_path = DATABASE_PATH

    # 检查数据库文件是否存在
    if not Path(db_path).exists():
        logger.error(f"数据库文件不存在: {db_path}")
        return

    # 创建清理器并运行清理
    cleaner = TaskCleaner(db_path)
    cleaner.run_cleanup()

if __name__ == "__main__":
    main()