#!/usr/bin/env python3
"""
统计功能测试脚本
用于测试多实例部署的统计数据收集和API接口
"""

import asyncio
import aiohttp
import json
import time
import argparse
from typing import Dict, Any, List

# 配置
LOAD_BALANCER_URL = "http://localhost:5001"
API_KEY = "your-secret-api-key-here"

# 颜色输出
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

def log_info(msg: str):
    print(f"{GREEN}ℹ️  {msg}{NC}")

def log_warn(msg: str):
    print(f"{YELLOW}⚠️  {msg}{NC}")

def log_error(msg: str):
    print(f"{RED}❌ {msg}{NC}")

def log_debug(msg: str):
    print(f"{BLUE}🔍 {msg}{NC}")

async def test_load_balancer_stats(session: aiohttp.ClientSession) -> Dict[str, Any]:
    """测试负载均衡器统计接口"""
    log_info("Testing load balancer stats API...")

    try:
        async with session.get(f"{LOAD_BALANCER_URL}/stats", timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                stats_data = await response.json()
                log_info("✅ Load balancer stats API working!")
                return stats_data
            else:
                log_error(f"❌ Load balancer stats API failed: HTTP {response.status}")
                return {}
    except Exception as e:
        log_error(f"❌ Error testing load balancer stats: {e}")
        return {}

async def test_individual_instance_stats(session: aiohttp.ClientSession, service: str) -> Dict[str, Any]:
    """测试单个实例统计接口"""
    try:
        async with session.get(f"{service}/stats", timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status == 200:
                stats_data = await response.json()
                return stats_data
            else:
                return {"error": f"HTTP {response.status}"}
    except Exception as e:
        return {"error": str(e)}

async def test_inference_request(session: aiohttp.ClientSession, test_file: str = None) -> bool:
    """发送一个推理请求测试统计更新"""
    log_info("Testing inference request to update stats...")

    if not test_file:
        log_warn("No test file provided, skipping inference test")
        return False

    try:
        # 读取测试文件
        with open(test_file, 'rb') as f:
            file_data = f.read()

        # 创建 multipart data
        from aiohttp import FormData
        data = FormData()
        data.add_field('file', file_data,
                      filename=test_file,
                      content_type='application/octet-stream')
        data.add_field('response_format', 'srt')
        data.add_field('language', 'auto')

        # 发送请求
        start_time = time.time()
        async with session.post(f"{LOAD_BALANCER_URL}/inference",
                              data=data,
                              headers={'X-API-Key': API_KEY},
                              timeout=aiohttp.ClientTimeout(total=300)) as response:

            if response.status == 200:
                result = await response.json()
                processing_time = time.time() - start_time
                log_info(f"✅ Inference request completed in {processing_time:.2f}s")
                return True
            else:
                log_error(f"❌ Inference request failed: HTTP {response.status}")
                error_text = await response.text()
                log_error(f"Error response: {error_text[:200]}")
                return False

    except Exception as e:
        log_error(f"❌ Error during inference request: {e}")
        return False

def format_stats_display(stats_data: Dict[str, Any]) -> str:
    """格式化显示统计数据"""
    if not stats_data:
        return "No stats data available"

    output = []

    # 负载均衡器状态
    lb = stats_data.get("load_balancer", {})
    output.append(f"📊 Load Balancer Status:")
    output.append(f"   Status: {lb.get('status', 'unknown')}")
    output.append(f"   Healthy Backends: {lb.get('healthy_backends', 0)}/{lb.get('total_backends', 0)}")
    output.append(f"   Queue Length: {lb.get('queue_length', 0)}")
    output.append("")

    # 汇总统计
    aggregated = stats_data.get("aggregated_stats", {})
    output.append(f"📈 Aggregated Statistics:")
    output.append(f"   Total Requests: {aggregated.get('total_requests', 0)}")
    output.append(f"   Successful: {aggregated.get('successful_requests', 0)}")
    output.append(f"   Failed: {aggregated.get('failed_requests', 0)}")
    output.append(f"   Success Rate: {aggregated.get('success_rate_percent', 0):.1f}%")
    output.append(f"   Files Processed: {aggregated.get('total_files_processed', 0)}")
    output.append(f"   Total File Size: {aggregated.get('total_file_size_mb', 0):.1f} MB")
    output.append(f"   Total Chunks: {aggregated.get('total_chunks_processed', 0)}")
    output.append(f"   Avg File Size: {aggregated.get('average_file_size_mb', 0):.1f} MB")
    output.append(f"   Avg Processing Time: {aggregated.get('average_processing_time_seconds', 0):.1f}s")
    output.append("")

    # 实例详情
    instances = stats_data.get("instance_details", [])
    if instances:
        output.append(f"🖥️  Instance Details:")
        for instance in instances:
            output.append(f"   Instance {instance.get('instance_id', 'unknown')}:")
            output.append(f"     Port: {instance.get('port', 0)}, GPU: {instance.get('gpu_device', 'unknown')}")
            output.append(f"     Status: {instance.get('status', 'unknown')}")

            req_stats = instance.get('request_stats', {})
            if req_stats:
                output.append(f"     Requests: {req_stats.get('total_requests', 0)} "
                             f"(Success: {req_stats.get('successful_requests', 0)}, "
                             f"Failed: {req_stats.get('failed_requests', 0)})")

            file_stats = instance.get('file_stats', {})
            if file_stats and file_stats.get('total_files_processed', 0) > 0:
                output.append(f"     Files: {file_stats.get('total_files_processed', 0)}, "
                             f"Size: {file_stats.get('total_file_size_mb', 0):.1f}MB")

            perf_stats = instance.get('performance_stats', {})
            if perf_stats:
                output.append(f"     Avg Processing: {perf_stats.get('average_processing_time_seconds', 0):.1f}s")

            output.append("")

    return "\n".join(output)

async def run_stats_test(test_file: str = None, iterations: int = 1):
    """运行统计测试"""
    log_info("🚀 Starting statistics test...")
    log_info(f"Load Balancer URL: {LOAD_BALANCER_URL}")
    log_info(f"Test file: {test_file or 'None'}")
    log_info(f"Iterations: {iterations}")
    print("")

    async with aiohttp.ClientSession() as session:
        # 1. 测试初始统计
        log_info("📊 Getting initial statistics...")
        initial_stats = await test_load_balancer_stats(session)
        if initial_stats:
            log_info("Initial Statistics:")
            print(format_stats_display(initial_stats))

        print("=" * 60)

        # 2. 发送推理请求（如果有测试文件）
        if test_file:
            log_info(f"🔄 Sending {iterations} inference request(s) to update stats...")
            success_count = 0

            for i in range(iterations):
                log_info(f"Sending inference request {i+1}/{iterations}...")
                success = await test_inference_request(session, test_file)
                if success:
                    success_count += 1

                # 请求间隔
                if i < iterations - 1:
                    await asyncio.sleep(2)

            log_info(f"✅ Completed {success_count}/{iterations} inference requests")

            # 等待统计更新
            log_info("⏳ Waiting for stats to update...")
            await asyncio.sleep(5)

        print("=" * 60)

        # 3. 获取更新后的统计
        log_info("📊 Getting updated statistics...")
        final_stats = await test_load_balancer_stats(session)
        if final_stats:
            log_info("Updated Statistics:")
            print(format_stats_display(final_stats))

        # 4. 显示对比
        if initial_stats and final_stats and test_file:
            print("=" * 60)
            log_info("📈 Statistics Comparison:")

            initial_agg = initial_stats.get("aggregated_stats", {})
            final_agg = final_stats.get("aggregated_stats", {})

            requests_diff = final_agg.get("total_requests", 0) - initial_agg.get("total_requests", 0)
            success_diff = final_agg.get("successful_requests", 0) - initial_agg.get("successful_requests", 0)
            files_diff = final_agg.get("total_files_processed", 0) - initial_agg.get("total_files_processed", 0)

            log_info(f"   New Requests: +{requests_diff}")
            log_info(f"   New Successes: +{success_diff}")
            log_info(f"   New Files Processed: +{files_diff}")

        # 5. 测试各个实例的统计接口
        print("=" * 60)
        log_info("🔍 Testing individual instance stats APIs...")

        # 从负载均衡器获取后端服务列表
        backend_status = final_stats.get("backend_status", {})
        services = list(backend_status.keys())

        if services:
            log_info(f"Found {len(services)} backend services")

            # 并发测试所有实例
            tasks = [test_individual_instance_stats(session, service) for service in services]
            instance_results = await asyncio.gather(*tasks, return_exceptions=True)

            healthy_instances = 0
            for i, (service, result) in enumerate(zip(services, instance_results)):
                if isinstance(result, dict) and "error" not in result:
                    healthy_instances += 1
                    instance_info = result.get("stats", {}).get("instance_info", {})
                    log_info(f"   ✅ {service}: Instance {instance_info.get('instance_id', 'unknown')}, "
                           f"Port {instance_info.get('port', 0)}, "
                           f"GPU {instance_info.get('gpu_device', 'unknown')}")
                else:
                    log_error(f"   ❌ {service}: {result}")

            log_info(f"Individual instance stats: {healthy_instances}/{len(services)} healthy")
        else:
            log_warn("No backend services found")

        print("=" * 60)
        log_info("🎉 Statistics test completed!")

def main():
    parser = argparse.ArgumentParser(description="Test statistics functionality for multi-instance faster-whisper deployment")
    parser.add_argument("--test-file", "-f", help="Path to test audio file for inference requests")
    parser.add_argument("--iterations", "-n", type=int, default=1, help="Number of inference requests to send")
    parser.add_argument("--load-balancer", "-l", default=LOAD_BALANCER_URL, help="Load balancer URL")
    parser.add_argument("--api-key", "-k", default=API_KEY, help="API key for authentication")

    args = parser.parse_args()

    global LOAD_BALANCER_URL, API_KEY
    LOAD_BALANCER_URL = args.load_balancer
    API_KEY = args.api_key

    # 检查测试文件
    if args.test_file:
        try:
            import os
            if not os.path.exists(args.test_file):
                log_error(f"Test file not found: {args.test_file}")
                return
            log_info(f"Using test file: {args.test_file}")
        except Exception as e:
            log_error(f"Error checking test file: {e}")
            return

    # 运行测试
    asyncio.run(run_stats_test(args.test_file, args.iterations))

if __name__ == "__main__":
    main()