#!/usr/bin/env python3
"""
VAD检测程序使用示例
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"示例: {description}")
    print(f"命令: {cmd}")
    print('='*60)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"错误: {result.stderr}")

    return result.returncode == 0

def main():
    print("VAD检测程序使用示例")
    print("="*60)

    # 检查音频文件是否存在
    if not os.path.exists('66.wav'):
        print("错误: 66.wav 文件不存在")
        sys.exit(1)

    examples = [
        # 基本使用
        ('python vad_detector.py 66.wav', '基本使用 - 检测默认参数下的静音段'),

        # 自定义输出文件
        ('python vad_detector.py 66.wav -o custom_output.json',
         '自定义输出文件名'),

        # 调整能量阈值（更敏感）
        ('python vad_detector.py 66.wav -t 0.005 -o sensitive_detection.json',
         '降低能量阈值 - 检测更多静音段'),

        # 调整能量阈值（更宽松）
        ('python vad_detector.py 66.wav -t 0.02 -o relaxed_detection.json',
         '提高能量阈值 - 只检测明显的静音段'),

        # 调整最小静音持续时间
        ('python vad_detector.py 66.wav -d 1.0 -o long_silence.json',
         '只检测持续1秒以上的静音段'),

        # 检测短静音
        ('python vad_detector.py 66.wav -d 0.2 -o short_silence.json',
         '检测0.2秒以上的静音段'),

        # 组合参数
        ('python vad_detector.py 66.wav -t 0.005 -d 0.3 -o fine_detection.json',
         '精细检测 - 低阈值+短静音段'),

        # 调整音频处理参数
        ('python vad_detector.py 66.wav -f 1024 -l 256 -o high_res.json',
         '高分辨率检测 - 更小的帧窗口'),
    ]

    # 询问用户是否运行所有示例
    response = input("是否运行所有示例？这可能需要一些时间 (y/n): ")

    if response.lower() == 'y':
        success_count = 0
        for cmd, desc in examples:
            if run_command(cmd, desc):
                success_count += 1

        print(f"\n{'='*60}")
        print(f"示例运行完成: {success_count}/{len(examples)} 成功")
        print("="*60)
    else:
        # 让用户选择要运行的示例
        print("\n可用的示例:")
        for i, (_, desc) in enumerate(examples, 1):
            print(f"{i}. {desc}")

        try:
            choice = input("请选择要运行的示例编号 (1-8): ")
            choice = int(choice) - 1
            if 0 <= choice < len(examples):
                cmd, desc = examples[choice]
                run_command(cmd, desc)
            else:
                print("无效的选择")
        except ValueError:
            print("请输入有效的数字")

    # 显示结果文件
    print(f"\n{'='*60}")
    print("生成的结果文件:")
    print("="*60)

    result_files = [f for f in os.listdir('.') if f.endswith('.json') or f.endswith('.txt')]
    for file in result_files:
        size = os.path.getsize(file)
        print(f"{file:30} ({size} bytes)")

if __name__ == "__main__":
    main()