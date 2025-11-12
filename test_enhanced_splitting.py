#!/usr/bin/env python3
"""
Enhanced Audio Splitting Test Script
增强音频分割测试脚本

测试VAD引导的智能音频分割功能
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_splitter_enhanced import EnhancedAudioSplitter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_single_file(file_path: str, splitter: EnhancedAudioSplitter) -> Dict[str, Any]:
    """测试单个文件的分割"""
    logger.info(f"\n{'='*60}")
    logger.info(f"测试文件: {file_path}")
    logger.info(f"{'='*60}")

    try:
        # 获取文件信息
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        logger.info(f"文件大小: {file_size:.2f} MB")

        # 测试音频文件验证
        logger.info("步骤1: 验证音频文件...")
        is_valid, validation_result = splitter.validate_audio_file(file_path, "test_validation")
        if not is_valid:
            logger.error(f"音频验证失败: {validation_result['issues']}")
            return {'file': file_path, 'success': False, 'error': '; '.join(validation_result['issues'])}

        logger.info(f"✅ 音频验证通过")
        logger.info(f"   - 数据大小: {validation_result['data_size_bytes']} bytes")

        # 获取音频详细信息
        try:
            # 尝试使用不同的方法获取音频信息
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_format', '-show_streams',
                file_path
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                # 解析ffprobe输出获取时长
                for line in result.stdout.split('\n'):
                    if line.startswith('duration='):
                        duration = float(line.split('=')[1])
                        break
                else:
                    duration = 60.0  # 默认时长
                logger.info(f"   - 时长: {duration:.2f} 秒")
            else:
                duration = 60.0  # 默认时长
                logger.info(f"   - 时长: {duration:.2f} 秒 (估算)")

        except Exception as e:
            logger.warning(f"无法获取详细音频信息: {e}")
            # 基于文件大小估算时长 (假设平均比特率)
            estimated_bitrate = 128000  # 128 kbps
            duration = (file_size * 1024 * 1024 * 8) / estimated_bitrate
            logger.info(f"   - 估算时长: {duration:.2f} 秒")

        # 测试智能分片计算
        logger.info("\n步骤2: 计算智能分片...")
        chunk_count = splitter.calculate_smart_chunks(
            duration,
            available_workers=8,
            file_size_mb=file_size
        )

        chunk_info = {
            'chunk_count': chunk_count,
            'splitting_strategy': 'VAD-guided',
            'estimated_chunk_size_mb': file_size / max(chunk_count, 1),
            'total_overlap_seconds': 0.0
        }

        logger.info(f"✅ 分片计算完成:")
        logger.info(f"   - 分片数量: {chunk_info['chunk_count']}")
        logger.info(f"   - 分片策略: {chunk_info['splitting_strategy']}")
        logger.info(f"   - 预估分片大小: {chunk_info['estimated_chunk_size_mb']:.2f} MB")
        logger.info(f"   - 总重叠时间: {chunk_info['total_overlap_seconds']:.2f} 秒")

        # 测试VAD引导分割
        logger.info("\n步骤3: 执行VAD引导分割...")
        segments = splitter.split_with_vad_guidance_enhanced(file_path, chunk_info['chunk_count'])

        logger.info(f"✅ 分割完成，生成 {len(segments)} 个分片:")

        segment_info = []
        for i, (chunk_data, start_time, end_time) in enumerate(segments):
            # 保存分片到临时文件以检查大小
            temp_path = f"/tmp/test_chunk_{i}_{os.path.basename(file_path)}.wav"
            with open(temp_path, 'wb') as f:
                f.write(chunk_data)

            segment_size = os.path.getsize(temp_path) / (1024 * 1024)  # MB
            duration = end_time - start_time
            segment_info.append({
                'index': i,
                'path': temp_path,
                'size_mb': segment_size,
                'duration': duration,
                'start_time': start_time,
                'end_time': end_time
            })

            logger.info(f"   分片 {i+1}: {segment_size:.2f} MB, {duration:.2f}秒 "
                       f"({start_time:.2f}s - {end_time:.2f}s)")

        # 清理测试分片
        logger.info("\n步骤4: 清理测试分片...")
        for segment in segments:
            _, _, _ = segment  # Unpack tuple
        for info in segment_info:
            if os.path.exists(info['path']):
                os.unlink(info['path'])

        logger.info("✅ 清理完成")

        return {
            'file': file_path,
            'success': True,
            'file_size_mb': file_size,
            'duration': duration,
            'chunk_count': len(segments),
            'segments': segment_info
        }

    except Exception as e:
        logger.error(f"❌ 测试失败: {str(e)}")
        return {'file': file_path, 'success': False, 'error': str(e)}

def run_splitting_test():
    """运行音频分割测试"""
    logger.info("🚀 启动增强音频分割测试")

    # 初始化增强分割器
    splitter = EnhancedAudioSplitter()

    # 测试文件列表
    test_files = [
        'money.wav',
        '189.wav',
        'changan.wav',
        'changan_converted.wav'
    ]

    # 过滤存在的文件
    existing_files = []
    for file_path in test_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            logger.warning(f"测试文件不存在: {file_path}")

    if not existing_files:
        logger.error("没有找到可用的测试文件")
        return

    logger.info(f"找到 {len(existing_files)} 个测试文件")

    # 运行测试
    results = []
    for file_path in existing_files:
        result = test_single_file(file_path, splitter)
        results.append(result)

    # 生成测试报告
    logger.info(f"\n{'='*60}")
    logger.info("📊 测试报告")
    logger.info(f"{'='*60}")

    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]

    logger.info(f"总测试数: {len(results)}")
    logger.info(f"成功: {len(successful_tests)}")
    logger.info(f"失败: {len(failed_tests)}")

    if successful_tests:
        logger.info(f"\n✅ 成功测试详情:")
        for result in successful_tests:
            logger.info(f"   📁 {result['file']}")
            logger.info(f"      - 大小: {result['file_size_mb']:.2f} MB")
            logger.info(f"      - 时长: {result['duration']:.2f} 秒")
            logger.info(f"      - 分片数: {result['chunk_count']}")

            # 显示分片大小分布
            sizes = [s['size_mb'] for s in result['segments']]
            if sizes:
                logger.info(f"      - 分片大小范围: {min(sizes):.2f} - {max(sizes):.2f} MB")
                logger.info(f"      - 平均分片大小: {sum(sizes)/len(sizes):.2f} MB")

    if failed_tests:
        logger.info(f"\n❌ 失败测试详情:")
        for result in failed_tests:
            logger.info(f"   📁 {result['file']}")
            logger.info(f"      - 错误: {result['error']}")

    logger.info(f"\n🎉 测试完成！")

def main():
    """主函数"""
    try:
        run_splitting_test()
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()