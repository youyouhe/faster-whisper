#!/usr/bin/env python3
"""
异步API测试脚本
Test script for async transcription API
"""

import asyncio
import aiohttp
import json
import time
import os
from pathlib import Path

class AsyncAPITester:
    def __init__(self, base_url="http://localhost:5020"):
        self.base_url = base_url
        self.results = {}

    async def submit_async_task(self, session, file_path, language="auto", response_format="json", callback_url=None):
        """提交异步转录任务"""
        url = f"{self.base_url}/transcribe_async"

        # 准备文件和数据
        data = aiohttp.FormData()
        data.add_field('file', open(file_path, 'rb'),
                      filename=Path(file_path).name,
                      content_type='audio/wav')
        data.add_field('language', language)
        data.add_field('response_format', response_format)
        if callback_url:
            data.add_field('callback_url', callback_url)

        try:
            async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Task submitted: {result.get('task_id')}")
                    return result
                else:
                    error_text = await response.text()
                    print(f"❌ Submission failed: {response.status} - {error_text}")
                    return None
        except Exception as e:
            print(f"❌ Submission error: {e}")
            return None

    async def check_task_status(self, session, task_id):
        """检查任务状态"""
        url = f"{self.base_url}/task/{task_id}"

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    print(f"❌ Status check failed: {response.status}")
                    return None
        except Exception as e:
            print(f"❌ Status check error: {e}")
            return None

    async def wait_for_completion(self, session, task_id, max_wait_time=300, check_interval=2):
        """等待任务完成"""
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            status_result = await self.check_task_status(session, task_id)

            if status_result:
                status = status_result.get('status')
                print(f"📊 Task {task_id[:8]}... status: {status}")

                if status == 'completed':
                    print(f"✅ Task {task_id[:8]}... completed!")
                    self.results[task_id] = {
                        'status': 'completed',
                        'result': status_result.get('result'),
                        'processing_time': time.time() - start_time
                    }
                    return True

                elif status == 'failed':
                    error = status_result.get('error', 'Unknown error')
                    print(f"❌ Task {task_id[:8]}... failed: {error}")
                    self.results[task_id] = {
                        'status': 'failed',
                        'error': error,
                        'processing_time': time.time() - start_time
                    }
                    return False

            await asyncio.sleep(check_interval)

        print(f"⏰ Task {task_id[:8]}... timeout after {max_wait_time}s")
        self.results[task_id] = {
            'status': 'timeout',
            'processing_time': time.time() - start_time
        }
        return False

    async def test_single_async_task(self, file_path, response_format="json"):
        """测试单个异步任务"""
        print(f"\n🚀 Testing single async task: {Path(file_path).name} (format: {response_format})")

        async with aiohttp.ClientSession() as session:
            # 提交任务
            submit_result = await self.submit_async_task(session, file_path, response_format=response_format)
            if not submit_result:
                return False

            task_id = submit_result.get('task_id')
            if not task_id:
                print("❌ No task ID in response")
                return False

            # 等待完成
            success = await self.wait_for_completion(session, task_id)
            return success

    async def test_multiple_async_tasks(self, file_paths, max_concurrent=5):
        """测试多个异步任务并发"""
        print(f"\n🚀 Testing {len(file_paths)} async tasks (max concurrent: {max_concurrent})")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def submit_and_wait(file_path):
            async with semaphore:
                return await self.test_single_async_task(file_path)

        # 并发执行
        start_time = time.time()
        tasks = [submit_and_wait(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # 统计结果
        successful = sum(1 for r in results if r is True)
        failed = sum(1 for r in results if r is False)
        exceptions = sum(1 for r in results if isinstance(r, Exception))

        print(f"\n📈 Results Summary:")
        print(f"   Total tasks: {len(file_paths)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Exceptions: {exceptions}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Avg time per task: {total_time/len(file_paths):.2f}s")

        return results

    def print_detailed_results(self):
        """打印详细结果"""
        print(f"\n📋 Detailed Results:")
        for task_id, result in self.results.items():
            status = result['status']
            time_taken = result['processing_time']

            print(f"\nTask {task_id[:8]}...:")
            print(f"  Status: {status}")
            print(f"  Time: {time_taken:.2f}s")

            if status == 'completed' and 'result' in result:
                result_data = result['result']
                if isinstance(result_data, dict):
                    # 检查格式类型
                    format_type = result_data.get('format', 'json')
                    text = result_data.get('text', '')

                    if format_type == 'srt':
                        print(f"  Format: SRT")
                        print(f"  Segments: {result_data.get('segments_count', 'N/A')}")
                        # 显示SRT前几行
                        lines = text.split('\n')[:6]  # 显示前2个subtitle段落
                        preview = '\n'.join(lines)
                        print(f"  SRT preview:\n    {preview}")
                    else:
                        print(f"  Format: JSON")
                        print(f"  Text preview: {text[:100]}...")

async def main():
    """主测试函数"""
    # 检查服务是否运行
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:5020/health", timeout=5) as response:
                if response.status != 200:
                    print("❌ Service not healthy")
                    return
                print("✅ Service is running")
    except Exception as e:
        print(f"❌ Cannot connect to service: {e}")
        return

    tester = AsyncAPITester()

    # 寻找测试文件
    test_files = []
    current_dir = Path(__file__).parent

    # 查找WAV文件
    for wav_file in current_dir.glob("*.wav"):
        if wav_file.stat().st_size > 0:  # 非空文件
            test_files.append(str(wav_file))

    if not test_files:
        print("❌ No test WAV files found")
        print("💡 Please place some WAV files in the current directory")
        return

    print(f"📁 Found {len(test_files)} test files:")
    for f in test_files[:5]:  # 只显示前5个
        print(f"   - {Path(f).name} ({Path(f).stat().st_size} bytes)")

    # 测试1: 单个异步任务 (JSON格式)
    if test_files:
        print("\n" + "="*50)
        print("TEST 1: Single Async Task (JSON Format)")
        print("="*50)

        await tester.test_single_async_task(test_files[0], response_format="json")
        tester.print_detailed_results()

    # 测试2: 单个异步任务 (SRT格式)
    if test_files:
        print("\n" + "="*50)
        print("TEST 2: Single Async Task (SRT Format)")
        print("="*50)

        await tester.test_single_async_task(test_files[0], response_format="srt")
        tester.print_detailed_results()

    # 测试3: 多个异步任务并发
    if len(test_files) >= 2:
        print("\n" + "="*50)
        print("TEST 3: Multiple Async Tasks (Concurrent)")
        print("="*50)

        # 使用前几个文件进行并发测试
        concurrent_files = test_files[:min(5, len(test_files))]
        await tester.test_multiple_async_tasks(concurrent_files, max_concurrent=3)
        tester.print_detailed_results()

if __name__ == "__main__":
    print("🧪 Async API Test Script")
    print("Make sure the service is running on http://localhost:5020")
    print()

    asyncio.run(main())