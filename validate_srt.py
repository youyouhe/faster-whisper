#!/usr/bin/env python3
"""
SRT格式验证工具
SRT Format Validation Tool
"""

import re
import sys
from typing import List, Dict, Any


def validate_srt_format(srt_content: str) -> Dict[str, Any]:
    """
    验证SRT格式是否正确

    Args:
        srt_content: SRT内容字符串

    Returns:
        包含验证结果的字典
    """
    if not srt_content or not srt_content.strip():
        return {
            'valid': False,
            'error': 'SRT内容为空',
            'segments_count': 0
        }

    try:
        lines = srt_content.strip().split('\n')
        segments = []
        current_segment = []
        line_number = 0

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line:  # 空行，段落分隔符
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
                i += 1
                continue

            if current_segment:
                current_segment.append(line)
            else:
                # 新段落开始，应该是序号
                if not line.isdigit():
                    return {
                        'valid': False,
                        'error': f'第{line_number+1}行：段落序号应该是数字，但找到: "{line}"',
                        'segments_count': len(segments)
                    }
                current_segment = [line]

            line_number += 1
            i += 1

        # 处理最后一个段落
        if current_segment:
            segments.append(current_segment)

        # 验证每个段落的格式
        for seg_idx, segment in enumerate(segments):
            if len(segment) < 3:
                return {
                    'valid': False,
                    'error': f'第{seg_idx+1}个段落：至少需要3行（序号、时间戳、文本）',
                    'segments_count': len(segments)
                }

            # 检查时间戳格式
            timestamp_line = segment[1]
            timestamp_pattern = r'^\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}$'
            if not re.match(timestamp_pattern, timestamp_line):
                return {
                    'valid': False,
                    'error': f'第{seg_idx+1}个段落：时间戳格式错误，应为 HH:MM:SS,mmm --> HH:MM:SS,mmm，但找到: "{timestamp_line}"',
                    'segments_count': len(segments)
                }

            # 检查文本内容
            if len(segment) < 3 or not segment[2].strip():
                return {
                    'valid': False,
                    'error': f'第{seg_idx+1}个段落：缺少文本内容',
                    'segments_count': len(segments)
                }

        return {
            'valid': True,
            'error': None,
            'segments_count': len(segments),
            'segments': segments
        }

    except Exception as e:
        return {
            'valid': False,
            'error': f'解析SRT内容时发生错误: {str(e)}',
            'segments_count': 0
        }


def analyze_srt_content(srt_content: str) -> Dict[str, Any]:
    """
    分析SRT内容的详细信息

    Args:
        srt_content: SRT内容字符串

    Returns:
        包含分析结果的字典
    """
    validation_result = validate_srt_format(srt_content)

    if not validation_result['valid']:
        return validation_result

    segments = validation_result['segments']

    # 分析时间戳
    timestamps = []
    total_duration = 0

    for segment in segments:
        timestamp_line = segment[1]
        # 解析开始和结束时间
        start_str, end_str = timestamp_line.split(' --> ')

        def time_to_seconds(time_str):
            """将时间字符串转换为秒"""
            time_part, ms_part = time_str.split(',')
            hours, minutes, seconds = map(int, time_part.split(':'))
            return hours * 3600 + minutes * 60 + seconds + int(ms_part) / 1000

        start_time = time_to_seconds(start_str)
        end_time = time_to_seconds(end_str)
        duration = end_time - start_time

        timestamps.append({
            'start': start_time,
            'end': end_time,
            'duration': duration,
            'text': segment[2]
        })

        total_duration = max(total_duration, end_time)

    # 统计文本信息
    total_characters = sum(len(seg['text']) for seg in timestamps)
    total_words = sum(len(seg['text'].split()) for seg in timestamps)

    # 计算平均段落时长
    segment_durations = [seg['duration'] for seg in timestamps]
    avg_duration = sum(segment_durations) / len(segment_durations) if segment_durations else 0

    return {
        'valid': True,
        'error': None,
        'segments_count': len(segments),
        'total_duration_seconds': total_duration,
        'total_characters': total_characters,
        'total_words': total_words,
        'average_segment_duration': avg_duration,
        'timestamps': timestamps[:5]  # 只返回前5个时间戳作为预览
    }


def format_duration(seconds: float) -> str:
    """格式化时长为可读格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python validate_srt.py <srt文件或内容>")
        print("示例: python validate_srt.py result.srt")
        print("      python validate_srt.py '1\\n00:00:00,000 --> 00:00:02,000\\nHello world'")
        sys.exit(1)

    input_arg = sys.argv[1]

    # 判断是文件路径还是内容字符串
    try:
        import os
        if os.path.isfile(input_arg):
            # 从文件读取
            with open(input_arg, 'r', encoding='utf-8') as f:
                srt_content = f.read()
            source = f"文件 {input_arg}"
        else:
            # 直接作为内容处理
            srt_content = input_arg
            source = "命令行内容"
    except Exception as e:
        print(f"❌ 读取输入失败: {e}")
        sys.exit(1)

    print(f"🔍 验证SRT格式 - 来源: {source}")
    print("=" * 50)

    # 验证格式
    validation_result = validate_srt_format(srt_content)

    if not validation_result['valid']:
        print(f"❌ SRT格式验证失败")
        print(f"错误: {validation_result['error']}")
        print(f"段落数: {validation_result['segments_count']}")
        sys.exit(1)

    print(f"✅ SRT格式验证通过")
    print(f"段落数: {validation_result['segments_count']}")

    # 分析内容
    analysis = analyze_srt_content(srt_content)

    print(f"\n📊 内容分析:")
    print(f"   总时长: {format_duration(analysis['total_duration_seconds'])}")
    print(f"   总字符数: {analysis['total_characters']}")
    print(f"   总词数: {analysis['total_words']}")
    print(f"   平均段落时长: {format_duration(analysis['average_segment_duration'])}")

    print(f"\n📝 时间戳预览 (前5个):")
    for i, ts in enumerate(analysis['timestamps']):
        print(f"   {i+1}. {format_duration(ts['start'])} --> {format_duration(ts['end'])} "
              f"({format_duration(ts['duration'])})")
        print(f"      文本: {ts['text'][:50]}{'...' if len(ts['text']) > 50 else ''}")

    print(f"\n✨ 验证完成！SRT格式正确。")


if __name__ == "__main__":
    main()