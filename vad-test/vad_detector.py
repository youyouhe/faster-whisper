#!/usr/bin/env python3
"""
VAD (Voice Activity Detection) 静音检测程序
用于检测WAV音频文件中的静音段（非语音活动部分）

作者: AI Assistant
"""

import argparse
import librosa
import numpy as np
import json
from datetime import timedelta
import os
import sys


class VADDetector:
    def __init__(self, audio_file, frame_length=2048, hop_length=512,
                 energy_threshold=0.01, silence_duration=0.5):
        """
        初始化VAD检测器

        Args:
            audio_file: 音频文件路径
            frame_length: FFT窗口大小
            hop_length: 帧移
            energy_threshold: 能量阈值，低于此值视为静音
            silence_duration: 静音段最小持续时间（秒）
        """
        self.audio_file = audio_file
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.sample_rate = None
        self.audio_data = None

    def load_audio(self):
        """加载音频文件"""
        try:
            self.audio_data, self.sample_rate = librosa.load(self.audio_file, sr=None)
            print(f"已加载音频文件: {self.audio_file}")
            print(f"采样率: {self.sample_rate} Hz")
            print(f"音频时长: {len(self.audio_data)/self.sample_rate:.2f} 秒")
            return True
        except Exception as e:
            print(f"加载音频文件失败: {e}")
            return False

    def compute_energy(self):
        """计算音频信号的短时能量"""
        # 计算短时能量
        stft = librosa.stft(self.audio_data,
                           n_fft=self.frame_length,
                           hop_length=self.hop_length)
        magnitude = np.abs(stft)
        energy = np.sum(magnitude**2, axis=0)
        return energy

    def detect_silence(self):
        """检测静音段"""
        # 计算能量
        energy = self.compute_energy()

        # 归一化能量
        energy = energy / np.max(energy)

        # 计算时间轴
        time_frames = librosa.frames_to_time(
            range(len(energy)),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )

        # 检测静音帧
        silence_frames = energy < self.energy_threshold

        # 将连续的静音帧合并成静音段
        silence_segments = []
        in_silence = False
        start_time = 0

        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_silence:
                # 开始新的静音段
                start_time = time_frames[i]
                in_silence = True
            elif not is_silent and in_silence:
                # 结束静音段
                end_time = time_frames[i]
                duration = end_time - start_time
                if duration >= self.silence_duration:
                    silence_segments.append({
                        'start': start_time,
                        'end': end_time,
                        'duration': duration
                    })
                in_silence = False

        # 处理音频结尾的静音段
        if in_silence:
            end_time = time_frames[-1]
            duration = end_time - start_time
            if duration >= self.silence_duration:
                silence_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'duration': duration
                })

        return silence_segments

    def format_time(self, seconds):
        """格式化时间为可读格式"""
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)

        if hours > 0:
            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{milliseconds:03d}"
        else:
            return f"{int(minutes):02d}:{int(seconds):02d}.{milliseconds:03d}"

    def save_results(self, silence_segments, output_file):
        """保存检测结果到文件"""
        results = {
            'audio_file': self.audio_file,
            'parameters': {
                'frame_length': self.frame_length,
                'hop_length': self.hop_length,
                'energy_threshold': self.energy_threshold,
                'silence_duration': self.silence_duration
            },
            'silence_segments': silence_segments,
            'total_silence_time': sum(seg['duration'] for seg in silence_segments),
            'silence_ratio': sum(seg['duration'] for seg in silence_segments) / (len(self.audio_data) / self.sample_rate)
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results

    def print_results(self, silence_segments):
        """打印检测结果"""
        print("\n" + "="*60)
        print("VAD 静音检测结果")
        print("="*60)

        if not silence_segments:
            print("未检测到静音段")
            return

        print(f"检测到 {len(silence_segments)} 个静音段:")
        print("-"*60)
        print(f"{'序号':<4} {'开始时间':<12} {'结束时间':<12} {'持续时间(秒)':<12}")
        print("-"*60)

        for i, segment in enumerate(silence_segments, 1):
            start_time = self.format_time(segment['start'])
            end_time = self.format_time(segment['end'])
            duration = f"{segment['duration']:.3f}"
            print(f"{i:<4} {start_time:<12} {end_time:<12} {duration:<12}")

        total_silence = sum(seg['duration'] for seg in silence_segments)
        total_duration = len(self.audio_data) / self.sample_rate
        silence_ratio = total_silence / total_duration * 100

        print("-"*60)
        print(f"总静音时间: {total_silence:.3f} 秒")
        print(f"音频总时长: {total_duration:.3f} 秒")
        print(f"静音占比: {silence_ratio:.1f}%")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='VAD 静音检测程序')
    parser.add_argument('input_file', nargs='?', default='66.wav',
                       help='输入音频文件路径 (默认: 66.wav)')
    parser.add_argument('-o', '--output', default='silence_detection.json',
                       help='输出结果文件路径 (默认: silence_detection.json)')
    parser.add_argument('-t', '--threshold', type=float, default=0.01,
                       help='能量阈值 (0-1, 默认: 0.01)')
    parser.add_argument('-d', '--duration', type=float, default=0.5,
                       help='最小静音持续时间(秒, 默认: 0.5)')
    parser.add_argument('-f', '--frame-length', type=int, default=2048,
                       help='FFT窗口大小 (默认: 2048)')
    parser.add_argument('-l', '--hop-length', type=int, default=512,
                       help='帧移 (默认: 512)')

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 音频文件 '{args.input_file}' 不存在")
        sys.exit(1)

    # 创建VAD检测器
    detector = VADDetector(
        audio_file=args.input_file,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        energy_threshold=args.threshold,
        silence_duration=args.duration
    )

    # 加载音频文件
    if not detector.load_audio():
        sys.exit(1)

    # 检测静音段
    print(f"\n正在检测静音段 (能量阈值: {args.threshold}, 最小静音时长: {args.duration}秒)...")
    silence_segments = detector.detect_silence()

    # 打印结果
    detector.print_results(silence_segments)

    # 保存结果
    results = detector.save_results(silence_segments, args.output)
    print(f"\n检测结果已保存到: {args.output}")

    # 输出时间戳列表（便于其他程序使用）
    timestamp_file = args.output.replace('.json', '_timestamps.txt')
    with open(timestamp_file, 'w', encoding='utf-8') as f:
        for seg in silence_segments:
            f.write(f"{seg['start']:.3f}\t{seg['end']:.3f}\t{seg['duration']:.3f}\n")

    print(f"时间戳列表已保存到: {timestamp_file}")


if __name__ == "__main__":
    main()