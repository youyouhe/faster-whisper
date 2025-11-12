#!/usr/bin/env python3
"""
VAD参数测试和对比脚本
用于测试不同参数组合的检测效果
"""

import subprocess
import json
import os
from datetime import datetime

def run_vad_test(params, test_name):
    """运行VAD测试并返回结果"""
    cmd = f"python vad_detector.py 66.wav -o {test_name}.json {' '.join(params)}"
    print(f"\n测试: {test_name}")
    print(f"命令: {cmd}")
    print("-" * 50)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        # 读取结果文件
        try:
            with open(f"{test_name}.json", 'r', encoding='utf-8') as f:
                data = json.load(f)

            segments_count = len(data['silence_segments'])
            total_silence = data['total_silence_time']
            silence_ratio = data['silence_ratio'] * 100

            print(f"✅ 成功")
            print(f"静音段数量: {segments_count}")
            print(f"总静音时间: {total_silence:.2f}秒")
            print(f"静音占比: {silence_ratio:.1f}%")

            return {
                'name': test_name,
                'params': ' '.join(params),
                'segments': segments_count,
                'total_silence': total_silence,
                'silence_ratio': silence_ratio
            }
        except Exception as e:
            print(f"❌ 读取结果失败: {e}")
            return None
    else:
        print(f"❌ 执行失败: {result.stderr}")
        return None

def main():
    print("VAD参数对比测试")
    print("=" * 60)

    # 检查音频文件
    if not os.path.exists('66.wav'):
        print("错误: 66.wav 文件不存在")
        return

    # 测试配置
    test_configs = [
        # 基础配置
        ([], "baseline", "基础配置"),

        # 阈值测试
        (["-t", "0.005"], "sensitive_threshold", "低阈值-敏感检测"),
        (["-t", "0.02"], "relaxed_threshold", "高阈值-宽松检测"),
        (["-t", "0.05"], "very_relaxed", "极高阈值-只检测明显静音"),

        # 静音时长测试
        (["-d", "0.2"], "short_silence", "短静音段-0.2秒"),
        (["-d", "1.0"], "long_silence", "长静音段-1.0秒"),
        (["-d", "2.0"], "very_long", "很长静音段-2.0秒"),

        # 窗口大小测试
        (["-f", "1024"], "small_window", "小窗口-高时间精度"),
        (["-f", "4096"], "large_window", "大窗口-高稳定性"),

        # 帧移测试
        (["-l", "256"], "small_hop", "小帧移-高精度"),
        (["-l", "1024"], "large_hop", "大帧移-快速处理"),

        # 组合测试
        (["-t", "0.005", "-d", "0.3"], "fine_detection", "精细检测"),
        (["-t", "0.02", "-d", "1.0", "-f", "4096"], "stable_detection", "稳定检测"),
        (["-t", "0.01", "-f", "1024", "-l", "256"], "precise_detection", "精确检测"),

        # 极端配置
        (["-t", "0.003", "-d", "0.1", "-f", "1024", "-l", "128"], "ultra_sensitive", "超敏感"),
        (["-t", "0.05", "-d", "3.0", "-f", "4096"], "ultra_conservative", "超保守"),
    ]

    results = []
    print("开始参数测试...")
    print(f"音频文件: 66.wav")
    print(f"测试配置数量: {len(test_configs)}")

    # 运行所有测试
    for i, (params, name, desc) in enumerate(test_configs, 1):
        print(f"\n[{i}/{len(test_configs)}] {desc}")
        result = run_vad_test(params, name)
        if result:
            result['description'] = desc
            results.append(result)

    # 生成对比报告
    if results:
        print("\n" + "=" * 80)
        print("参数对比报告")
        print("=" * 80)

        # 按静音段数量排序
        results_sorted = sorted(results, key=lambda x: x['segments'], reverse=True)

        print(f"{'配置名称':<20} {'描述':<20} {'静音段数':<8} {'静音时长(s)':<12} {'静音占比(%)':<12}")
        print("-" * 80)

        for result in results_sorted:
            print(f"{result['name']:<20} {result['description']:<20} "
                  f"{result['segments']:<8} {result['total_silence']:<12.2f} "
                  f"{result['silence_ratio']:<12.1f}")

        # 保存对比报告
        with open('parameter_comparison.json', 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'audio_file': '66.wav',
                'total_tests': len(test_configs),
                'successful_tests': len(results),
                'results': results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n对比报告已保存到: parameter_comparison.json")

        # 推荐配置
        print("\n" + "=" * 50)
        print("推荐配置")
        print("=" * 50)

        print("1. 精确检测（推荐播客等清晰语音）:")
        print("   python vad_detector.py 66.wav -t 0.005 -d 0.3 -f 1024 -l 256")

        print("\n2. 稳定检测（推荐有噪音的录音）:")
        print("   python vad_detector.py 66.wav -t 0.02 -d 1.0 -f 4096")

        print("\n3. 快速检测（推荐大文件处理）:")
        print("   python vad_detector.py 66.wav -f 4096 -l 1024")

        print("\n4. 通用检测（默认配置）:")
        print("   python vad_detector.py 66.wav")

    print(f"\n测试完成！生成了 {len(results)} 个有效结果。")

if __name__ == "__main__":
    main()