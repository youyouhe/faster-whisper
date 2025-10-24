#!/usr/bin/env python3
"""
负载均衡器测试脚本
Load Balancer Test Script
"""

import asyncio
import aiohttp
import json
import time
import os
from pathlib import Path

class LoadBalancerTester:
    def __init__(self, base_url="http://localhost:5001", api_key=None):
        self.base_url = base_url
        self.api_key = api_key or os.getenv('API_KEY', 'a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456')
        self.results = {}

        # 设置请求头
        self.headers = {}
        if self.api_key:
            self.headers['X-API-Key'] = self.api_key
            print(f"🔐 Using API key authentication: {self.api_key[:8]}...{self.api_key[-8:]}")
        else:
            print("⚠️  No API key configured - requests may fail")

    async def test_load_balancer_direct(self, session, file_path):
        """直接测试负载均衡器的同步接口（仅支持SRT格式）"""
        file_size = Path(file_path).stat().st_size
        print(f"\n🚀 Testing load balancer: {Path(file_path).name} ({file_size:,} bytes, SRT format)")

        start_time = time.time()
        try:
            # 准备文件和数据
            with open(file_path, 'rb') as f:
                file_data = f.read()

            # 创建multipart数据（仅SRT格式）
            data = aiohttp.FormData()
            data.add_field('file', file_data,
                          filename=Path(file_path).name,
                          content_type='audio/wav')
            data.add_field('response_format', 'srt')

            # 根据文件大小动态调整超时时间
            # 大文件需要更长时间处理
            if file_size > 10 * 1024 * 1024:  # > 10MB
                timeout_total = 900  # 15分钟
            elif file_size > 5 * 1024 * 1024:  # > 5MB
                timeout_total = 600  # 10分钟
            else:
                timeout_total = 300  # 5分钟

            print(f"  ⏱️  Timeout set to {timeout_total}s for {file_size:,} bytes file")

            # 直接调用负载均衡器的inference接口
            async with session.post(f"{self.base_url}/inference",
                                         data=data,
                                         headers=self.headers,
                                         timeout=aiohttp.ClientTimeout(total=timeout_total, connect=60)) as response:
                processing_time = time.time() - start_time

                if response.status == 200:
                    result = await response.json()
                    if result.get('code') == 0:
                        task_id = f"task_{int(time.time())}"
                        self.results[task_id] = {
                            'status': 'completed',
                            'result': result.get('data'),
                            'processing_time': processing_time
                        }
                        print(f"✅ Task {task_id[:8]}... completed in {processing_time:.1f}s!")
                        print(f"  📝 Format: SRT | Response code: {result.get('code')}")
                        srt_content = result.get('data', '')
                        print(f"  📄 SRT preview: {srt_content[:80]}...")
                        return True
                    else:
                        error_msg = result.get('msg', 'Unknown error')
                        print(f"❌ Task failed after {processing_time:.1f}s: {error_msg}")
                        return False
                else:
                    error_text = await response.text()
                    print(f"❌ HTTP error {response.status} after {processing_time:.1f}s: {error_text}")
                    return False
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            print(f"❌ Request timeout after {processing_time:.1f}s (file: {file_size:,} bytes)")
            return False
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"❌ Request error after {processing_time:.1f}s: {e}")
            return False

    async def test_health(self):
        """测试负载均衡器健康状态"""
        print("🔍 Checking load balancer health...")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health",
                                     headers=self.headers,
                                     timeout=5) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        print("✅ Load balancer is healthy")
                        print(f"  Status: {health_data.get('status')}")
                        print(f"  Healthy backends: {health_data.get('healthy_backends', 0)}")
                        print(f"  Total backends: {health_data.get('total_backends', 0)}")
                        return True
                    else:
                        print(f"❌ Health check failed: HTTP {response.status}")
                        return False
        except Exception as e:
            print(f"❌ Cannot connect to load balancer: {e}")
            return False

    async def test_stats(self):
        """获取负载均衡器统计信息"""
        print("📊 Getting load balancer stats...")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/stats",
                                     headers=self.headers,
                                     timeout=10) as response:
                    if response.status == 200:
                        stats_data = await response.json()
                        print("✅ Stats retrieved successfully")

                        # 显示关键统计信息
                        data = stats_data.get('data', stats_data)

                        print(f"  Load balancer status: {data.get('load_balancer', {}).get('status')}")
                        print(f"  Active requests: {data.get('load_balancer', {}).get('active_requests', 0)}")
                        print(f"  Queue length: {data.get('load_balancer', {}).get('queue_length', 0)}")

                        # 显示后端服务状态
                        instance_details = data.get('instance_details', [])
                        if instance_details:
                            print(f"  Backend instances:")
                            for instance in instance_details[:3]:  # 只显示前3个
                                instance_id = instance.get('instance_id', 'unknown')
                                status = instance.get('status', 'unknown')
                                port = instance.get('port', 0)
                                gpu_device = instance.get('gpu_device', 'unknown')
                                print(f"    - {instance_id}: {status} (GPU: {gpu_device}, Port: {port})")

                        return True
                    else:
                        print(f"❌ Stats request failed: HTTP {response.status}")
                        return False
        except Exception as e:
            print(f"❌ Stats error: {e}")
            return False

    async def test_concurrent_requests(self, file_paths, max_concurrent=None):
        """测试并发请求（仅SRT格式）"""
        if max_concurrent is None:
            max_concurrent = len(file_paths)  # 使用所有文件

        total_size = sum(Path(f).stat().st_size for f in file_paths)
        print(f"\n🚀 Testing {len(file_paths)} concurrent SRT requests (max: {max_concurrent})")
        print(f"📁 Total data size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")

        # 创建一个共享的session用于所有并发请求
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def test_single(file_path):
                async with semaphore:
                    return await self.test_load_balancer_direct(session, file_path)

            # 并发执行 - 使用所有文件
            start_time = time.time()
            tasks = [test_single(path) for path in file_paths]

            # 添加进度显示
            print(f"  ⏳ Waiting for {len(tasks)} requests to complete...")

            # 添加进度监控任务
            completed_count = 0
            async def monitor_progress():
                nonlocal completed_count
                while completed_count < len(tasks):
                    await asyncio.sleep(10)  # 每10秒显示一次进度
                    if completed_count > 0:
                        elapsed = time.time() - start_time
                        progress = completed_count / len(tasks) * 100
                        print(f"  📊 Progress: {completed_count}/{len(tasks)} ({progress:.1f}%) - {elapsed:.1f}s elapsed")

            progress_task = asyncio.create_task(monitor_progress())

            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                total_time = time.time() - start_time
                progress_task.cancel()  # 停止进度监控
            except KeyboardInterrupt:
                total_time = time.time() - start_time
                progress_task.cancel()
                print(f"\n  ⚠️ Test interrupted by user after {total_time:.1f}s")
                completed = len([r for r in results if r is True])
                print(f"  📊 Partial results: {completed}/{len(file_paths)} completed so far")
                raise

        # 统计结果
        successful = sum(1 for r in results if r is True)
        failed = sum(1 for r in results if r is False)
        exceptions = sum(1 for r in results if isinstance(r, Exception))

        # 计算处理速度
        throughput = total_size / total_time / 1024 / 1024 if total_time > 0 else 0

        print(f"\n📈 Concurrent SRT Test Results:")
        print(f"  Total requests: {len(file_paths)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Exceptions: {exceptions}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg time per request: {total_time/len(file_paths):.2f}s")
        print(f"  Throughput: {throughput:.2f} MB/s")

        return results

    def print_summary(self):
        """打印测试总结"""
        print(f"\n📋 Final Test Summary:")

        total = len(self.results)
        completed = sum(1 for r in self.results.values() if r['status'] == 'completed')
        failed = total - completed

        print(f"  Total tests: {total}")
        print(f"  Successful: {completed}")
        print(f"  Failed: {failed}")

        if completed > 0:
            print(f"  Success rate: {completed/total*100:.1f}%")

            # 计算处理时间统计
            processing_times = [r['processing_time'] for r in self.results.values() if r['processing_time'] > 0]
            if processing_times:
                avg_time = sum(processing_times) / len(processing_times)
                min_time = min(processing_times)
                max_time = max(processing_times)
                print(f"  Processing time: avg {avg_time:.1f}s, min {min_time:.1f}s, max {max_time:.1f}s")
        else:
            print("  ❌ All tests failed")

async def main():
    """主测试函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Load Balancer Test Script')
    parser.add_argument('--base-url', default='http://localhost:5001',
                       help='Load balancer base URL (default: http://localhost:5001)')
    parser.add_argument('--api-key',
                       help='API key for authentication (default: from env or hardcoded)')
    parser.add_argument('--url', help='Load balancer URL (alias for --base-url)')

    args = parser.parse_args()

    # 支持旧的--url参数
    base_url = args.url if args.url else args.base_url

    print("🧪 Load Balancer Test Script")
    print(f"Testing load balancer at {base_url}")
    print()

    tester = LoadBalancerTester(base_url=base_url, api_key=args.api_key)

    # 验证API key是否配置
    if not tester.api_key:
        print("⚠️  Warning: No API key configured. Set API_KEY environment variable or use --api-key")
        print("   Docker services may require authentication.")
        print()

    # 1. 健康检查
    health_ok = await tester.test_health()
    if not health_ok:
        print("❌ Load balancer is not responding. Exiting.")
        return

    # 2. 获取统计信息
    await tester.test_stats()

    # 3. 查找测试文件
    test_files = []
    current_dir = Path(__file__).parent

    # 查找WAV文件
    for wav_file in current_dir.glob("*.wav"):
        if wav_file.stat().st_size > 0:  # 非空文件
            test_files.append(str(wav_file))

    # 查找MP3文件作为备选
    if not test_files:
        for mp3_file in current_dir.glob("*.mp3"):
            if mp3_file.stat().st_size > 0:
                test_files.append(str(mp3_file))

    if not test_files:
        print("❌ No test audio files found")
        print("💡 Please place some .wav or .mp3 files in the current directory")
        return

    print(f"\n📁 Found {len(test_files)} test files:")
    for f in test_files[:3]:  # 只显示前3个
        print(f"  - {Path(f).name} ({Path(f).stat().st_size} bytes)")

    # 4. 测试单个SRT请求
    if test_files:
        print(f"\n" + "="*50)
        print("TEST 1: Single SRT Request")
        print("="*50)

        async with aiohttp.ClientSession() as session:
            success = await tester.test_load_balancer_direct(session, test_files[0])
            if success:
                tester.print_summary()

    # 5. 测试并发SRT请求 - 使用所有找到的wav文件
    if test_files:
        print(f"\n" + "="*50)
        print("TEST 2: Concurrent SRT Requests (All Files)")
        print("="*50)

        await tester.test_concurrent_requests(test_files)  # 使用所有文件，不限制并发数
        tester.print_summary()

if __name__ == "__main__":
    print("Starting load balancer test...")
    asyncio.run(main())
