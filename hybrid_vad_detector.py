#!/usr/bin/env python3
"""
Hybrid VAD Detector - 结合silero-vad和librosa能量VAD
用于更准确和鲁棒的音频静音检测
"""

import numpy as np
import librosa
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
import io
import wave
import struct

try:
    from faster_whisper.vad import VadOptions, get_speech_timestamps
except ImportError:
    get_speech_timestamps = None
    VadOptions = None

logger = logging.getLogger(__name__)

@dataclass
class SilenceSegment:
    """静音段数据结构"""
    start_time: float
    end_time: float
    duration: float
    confidence: float = 1.0
    source: str = "hybrid"  # 'energy', 'silero', 'hybrid'

class HybridVADDetector:
    """
    混合VAD检测器，结合两种VAD方法：
    1. 基于能量的VAD（librosa）- 高灵敏度
    2. 基于机器学习的VAD（silero-vad）- 高精度
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 # 能量VAD参数
                 energy_threshold: float = 0.01,
                 min_silence_duration_energy: float = 0.5,
                 frame_length: int = 2048,
                 hop_length: int = 512,

                 # Silero VAD参数
                 min_silence_duration_silero: float = 0.3,
                 max_speech_duration_silero: float = 60.0,

                 # 混合策略参数
                 hybrid_mode: str = "intersection",  # 'union', 'intersection', 'energy_primary'
                 confidence_threshold: float = 0.6):

        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.min_silence_duration_energy = min_silence_duration_energy
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.min_silence_duration_silero = min_silence_duration_silero
        self.max_speech_duration_silero = max_speech_duration_silero
        self.hybrid_mode = hybrid_mode
        self.confidence_threshold = confidence_threshold

        logger.info(f"[HybridVAD] Initialized with mode={hybrid_mode}, "
                   f"energy_threshold={energy_threshold}, "
                   f"min_silence_energy={min_silence_duration_energy}s, "
                   f"min_silence_silero={min_silence_duration_silero}s")

        # 检查silero-vad可用性
        self.silero_available = get_speech_timestamps is not None
        if not self.silero_available:
            logger.warning("[HybridVAD] Silero VAD not available, using energy-based VAD only")

    def detect_silence(self, audio: np.ndarray,
                      min_silence_duration: Optional[float] = None) -> List[SilenceSegment]:
        """
        检测音频中的静音段

        Args:
            audio: 音频数据 (float32 mono)
            min_silence_duration: 最小静音持续时间，如果为None则使用初始化参数

        Returns:
            静音段列表
        """
        if min_silence_duration is None:
            min_silence_duration = self.min_silence_duration_energy

        logger.info(f"[HybridVAD] Detecting silence in {len(audio)/self.sample_rate:.2f}s audio")

        # 方法1：基于能量的VAD检测
        energy_segments = self._detect_energy_based_silence(audio, min_silence_duration)
        logger.info(f"[HybridVAD] Energy VAD found {len(energy_segments)} silence segments")

        # 方法2：silero-vad检测（如果可用）
        silero_segments = []
        if self.silero_available:
            silero_segments = self._detect_silero_based_silence(audio, min_silence_duration)
            logger.info(f"[HybridVAD] Silero VAD found {len(silero_segments)} silence segments")

        # 混合策略
        hybrid_segments = self._merge_detection_results(energy_segments, silero_segments)
        logger.info(f"[HybridVAD] Hybrid approach found {len(hybrid_segments)} final segments")

        return hybrid_segments

    def _detect_energy_based_silence(self, audio: np.ndarray,
                                   min_silence_duration: float) -> List[SilenceSegment]:
        """基于能量的静音检测"""
        try:
            # 计算短时能量
            stft = librosa.stft(audio,
                               n_fft=self.frame_length,
                               hop_length=self.hop_length)
            magnitude = np.abs(stft)
            energy = np.sum(magnitude**2, axis=0)

            # 归一化能量
            if np.max(energy) > 0:
                energy = energy / np.max(energy)
            else:
                return []

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
                    if duration >= min_silence_duration:
                        silence_segments.append(SilenceSegment(
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration,
                            confidence=min(1.0, self.energy_threshold / energy[i]) if i < len(energy) else 1.0,
                            source="energy"
                        ))
                    in_silence = False

            # 处理音频结尾的静音段
            if in_silence and len(time_frames) > 0:
                end_time = time_frames[-1]
                duration = end_time - start_time
                if duration >= min_silence_duration:
                    silence_segments.append(SilenceSegment(
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        confidence=0.8,
                        source="energy"
                    ))

            return silence_segments

        except Exception as e:
            logger.error(f"[HybridVAD] Energy-based detection failed: {e}")
            return []

    def _detect_silero_based_silence(self, audio: np.ndarray,
                                   min_silence_duration: float) -> List[SilenceSegment]:
        """基于silero-vad的静音检测"""
        try:
            # 创建VAD选项
            vad_options = VadOptions(
                min_silence_duration_ms=int(min_silence_duration * 1000),
                max_speech_duration_s=self.max_speech_duration_silero
            )

            # 获取语音时间戳
            speech_timestamps = get_speech_timestamps(
                audio=audio,
                vad_options=vad_options,
                sampling_rate=self.sample_rate
            )

            # 转换语音时间戳为静音段
            silence_segments = []
            total_duration = len(audio) / self.sample_rate

            if not speech_timestamps:
                # 没有检测到语音，整个音频都是静音
                silence_segments.append(SilenceSegment(
                    start_time=0.0,
                    end_time=total_duration,
                    duration=total_duration,
                    confidence=0.9,
                    source="silero"
                ))
                return silence_segments

            # 开头的静音段
            if speech_timestamps[0]['start'] > 0:
                start_time = 0.0
                end_time = speech_timestamps[0]['start'] / self.sample_rate
                duration = end_time - start_time
                if duration >= min_silence_duration:
                    silence_segments.append(SilenceSegment(
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        confidence=0.9,
                        source="silero"
                    ))

            # 语音段之间的静音
            for i in range(len(speech_timestamps) - 1):
                current_speech_end = speech_timestamps[i]['end'] / self.sample_rate
                next_speech_start = speech_timestamps[i + 1]['start'] / self.sample_rate

                duration = next_speech_start - current_speech_end
                if duration >= min_silence_duration:
                    silence_segments.append(SilenceSegment(
                        start_time=current_speech_end,
                        end_time=next_speech_start,
                        duration=duration,
                        confidence=0.85,
                        source="silero"
                    ))

            # 结尾的静音段
            last_speech_end = speech_timestamps[-1]['end'] / self.sample_rate
            if last_speech_end < total_duration:
                duration = total_duration - last_speech_end
                if duration >= min_silence_duration:
                    silence_segments.append(SilenceSegment(
                        start_time=last_speech_end,
                        end_time=total_duration,
                        duration=duration,
                        confidence=0.8,
                        source="silero"
                    ))

            return silence_segments

        except Exception as e:
            logger.error(f"[HybridVAD] Silero-based detection failed: {e}")
            return []

    def _merge_detection_results(self,
                                energy_segments: List[SilenceSegment],
                                silero_segments: List[SilenceSegment]) -> List[SilenceSegment]:
        """合并两种VAD检测结果"""

        if not silero_segments:
            # 只有能量VAD结果
            return energy_segments

        if not energy_segments:
            # 只有silero VAD结果
            return silero_segments

        if self.hybrid_mode == "union":
            # 并集：两种方法检测到的静音段都包括
            return self._union_segments(energy_segments, silero_segments)
        elif self.hybrid_mode == "intersection":
            # 交集：只包括两种方法都检测到的静音段
            return self._intersection_segments(energy_segments, silero_segments)
        elif self.hybrid_mode == "energy_primary":
            # 能量优先：以能量VAD为主，用silero VAD验证
            return self._energy_primary_segments(energy_segments, silero_segments)
        else:
            # 默认返回能量VAD结果
            logger.warning(f"[HybridVAD] Unknown hybrid mode {self.hybrid_mode}, using energy-only")
            return energy_segments

    def _union_segments(self, energy_segments: List[SilenceSegment],
                       silero_segments: List[SilenceSegment]) -> List[SilenceSegment]:
        """合并两种检测结果的并集"""
        all_segments = energy_segments + silero_segments
        all_segments.sort(key=lambda x: x.start_time)

        # 合并重叠或相邻的段落
        merged_segments = []
        for segment in all_segments:
            if not merged_segments:
                merged_segments.append(segment)
            else:
                last_segment = merged_segments[-1]
                # 如果当前段落与上一段落重叠或相邻（间隔<0.1秒），则合并
                if segment.start_time <= last_segment.end_time + 0.1:
                    # 合并段落
                    merged_segment = SilenceSegment(
                        start_time=min(last_segment.start_time, segment.start_time),
                        end_time=max(last_segment.end_time, segment.end_time),
                        duration=max(last_segment.end_time, segment.end_time) -
                                min(last_segment.start_time, segment.start_time),
                        confidence=max(last_segment.confidence, segment.confidence),
                        source="hybrid_union"
                    )
                    merged_segments[-1] = merged_segment
                else:
                    merged_segments.append(segment)

        return merged_segments

    def _intersection_segments(self, energy_segments: List[SilenceSegment],
                              silero_segments: List[SilenceSegment]) -> List[SilenceSegment]:
        """取两种检测结果的交集"""
        intersection_segments = []

        for energy_seg in energy_segments:
            for silero_seg in silero_segments:
                # 计算重叠区域
                overlap_start = max(energy_seg.start_time, silero_seg.start_time)
                overlap_end = min(energy_seg.end_time, silero_seg.end_time)

                if overlap_start < overlap_end:
                    # 有重叠，计算重叠持续时间
                    overlap_duration = overlap_end - overlap_start
                    min_duration = min(energy_seg.duration, silero_seg.duration)

                    # 如果重叠超过原段落的50%，则认为是有效静音段
                    if overlap_duration >= min_duration * 0.5:
                        intersection_segments.append(SilenceSegment(
                            start_time=overlap_start,
                            end_time=overlap_end,
                            duration=overlap_duration,
                            confidence=(energy_seg.confidence + silero_seg.confidence) / 2,
                            source="hybrid_intersection"
                        ))
                        break  # 避免重复添加

        # 去重和排序
        intersection_segments.sort(key=lambda x: x.start_time)
        return self._remove_duplicate_segments(intersection_segments)

    def _energy_primary_segments(self, energy_segments: List[SilenceSegment],
                                silero_segments: List[SilenceSegment]) -> List[SilenceSegment]:
        """以能量VAD为主，用silero VAD验证和增强置信度"""
        validated_segments = []

        for energy_seg in energy_segments:
            # 检查是否有silero VAD支持
            silero_support = any(
                self._segments_overlap(energy_seg, silero_seg)
                for silero_seg in silero_segments
            )

            # 调整置信度
            confidence = energy_seg.confidence
            if silero_support:
                confidence = min(1.0, confidence + 0.2)  # 有silero支持时增加置信度
                source = "hybrid_validated"
            else:
                confidence = max(0.3, confidence - 0.1)  # 无silero支持时降低置信度
                source = "hybrid_energy_only"

            # 只有置信度超过阈值才保留
            if confidence >= self.confidence_threshold:
                validated_segment = SilenceSegment(
                    start_time=energy_seg.start_time,
                    end_time=energy_seg.end_time,
                    duration=energy_seg.duration,
                    confidence=confidence,
                    source=source
                )
                validated_segments.append(validated_segment)

        return validated_segments

    def _segments_overlap(self, seg1: SilenceSegment, seg2: SilenceSegment,
                         threshold: float = 0.3) -> bool:
        """检查两个段落是否重叠"""
        overlap_start = max(seg1.start_time, seg2.start_time)
        overlap_end = min(seg1.end_time, seg2.end_time)

        if overlap_start >= overlap_end:
            return False

        overlap_duration = overlap_end - overlap_start
        min_duration = min(seg1.duration, seg2.duration)

        return overlap_duration >= min_duration * threshold

    def _remove_duplicate_segments(self, segments: List[SilenceSegment]) -> List[SilenceSegment]:
        """移除重复的静音段"""
        if not segments:
            return segments

        # 按开始时间排序
        segments.sort(key=lambda x: x.start_time)

        # 去重
        deduped_segments = [segments[0]]
        for i in range(1, len(segments)):
            current = segments[i]
            last = deduped_segments[-1]

            # 如果当前段落与上一段落高度重叠，跳过
            if (abs(current.start_time - last.start_time) < 0.1 and
                abs(current.end_time - last.end_time) < 0.1):
                # 保留置信度更高的
                if current.confidence > last.confidence:
                    deduped_segments[-1] = current
            else:
                deduped_segments.append(current)

        return deduped_segments

    def get_optimal_split_points(self, audio: np.ndarray, num_splits: int,
                               min_silence_duration: Optional[float] = None) -> List[float]:
        """
        获取最优的音频分割点（静音段的中点）

        Args:
            audio: 音频数据
            num_splits: 需要的分割段数
            min_silence_duration: 最小静音持续时间

        Returns:
            分割点时间列表（按时间排序）
        """
        if min_silence_duration is None:
            min_silence_duration = self.min_silence_duration_energy

        # 检测所有静音段
        silence_segments = self.detect_silence(audio, min_silence_duration)

        if not silence_segments:
            logger.warning("[HybridVAD] No silence segments detected")
            return []

        # 计算理想的目标分割时间
        total_duration = len(audio) / self.sample_rate
        target_interval = total_duration / num_splits

        # 选择最优分割点
        split_points = []
        target_times = [target_interval * i for i in range(1, num_splits)]  # 理想分割时间

        for target_time in target_times:
            # 找到最接近目标时间的静音段中点
            best_segment = None
            min_distance = float('inf')

            for segment in silence_segments:
                segment_midpoint = (segment.start_time + segment.end_time) / 2
                distance = abs(segment_midpoint - target_time)

                # 检查是否已经被使用过（避免重复选择）
                if (distance < min_distance and
                    distance > target_interval * 0.1 and  # 避免选择太接近的点
                    segment_midpoint not in split_points):
                    min_distance = distance
                    best_segment = segment

            if best_segment:
                split_point = (best_segment.start_time + best_segment.end_time) / 2
                split_points.append(split_point)
                logger.info(f"[HybridVAD] Selected split point at {split_point:.2f}s "
                          f"(target: {target_time:.2f}s, confidence: {best_segment.confidence:.2f})")

        # 按时间排序
        split_points.sort()
        return split_points

    def get_detection_stats(self, audio: np.ndarray) -> Dict[str, any]:
        """获取VAD检测统计信息"""
        total_duration = len(audio) / self.sample_rate

        # 分别用两种方法检测
        energy_segments = self._detect_energy_based_silence(audio, 0.1)  # 最小阈值来统计
        silero_segments = self._detect_silero_based_silence(audio, 0.1) if self.silero_available else []
        hybrid_segments = self.detect_silence(audio)

        stats = {
            'audio_duration': total_duration,
            'energy_segments_count': len(energy_segments),
            'energy_total_silence': sum(seg.duration for seg in energy_segments),
            'energy_silence_ratio': sum(seg.duration for seg in energy_segments) / total_duration,
            'silero_segments_count': len(silero_segments),
            'silero_total_silence': sum(seg.duration for seg in silero_segments),
            'silero_silence_ratio': sum(seg.duration for seg in silero_segments) / total_duration,
            'hybrid_segments_count': len(hybrid_segments),
            'hybrid_total_silence': sum(seg.duration for seg in hybrid_segments),
            'hybrid_silence_ratio': sum(seg.duration for seg in hybrid_segments) / total_duration,
            'hybrid_mode': self.hybrid_mode,
            'silero_available': self.silero_available
        }

        return stats


# 使用示例和测试函数
def test_hybrid_vad(audio_file: str, num_splits: int = 8):
    """测试混合VAD检测器"""
    import os

    if not os.path.exists(audio_file):
        print(f"Error: Audio file {audio_file} not found")
        return

    # 加载音频
    try:
        audio, sr = librosa.load(audio_file, sr=16000)  # 确保采样率为16kHz
        print(f"Loaded {audio_file}: {len(audio)/sr:.2f}s, sample rate: {sr}Hz")
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    # 创建混合VAD检测器
    detector = HybridVADDetector(
        sample_rate=sr,
        energy_threshold=0.01,  # 根据你的测试参数
        min_silence_duration_energy=0.5,
        hybrid_mode="energy_primary",  # 以能量为主
        confidence_threshold=0.6
    )

    # 获取检测统计
    stats = detector.get_detection_stats(audio)
    print("\n=== VAD Detection Statistics ===")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    # 获取分割点
    split_points = detector.get_optimal_split_points(audio, num_splits)
    print(f"\n=== Optimal Split Points ({len(split_points)} found) ===")
    for i, point in enumerate(split_points, 1):
        print(f"Split {i}: {point:.2f}s")

    # 显示检测到的静音段
    silence_segments = detector.detect_silence(audio)
    print(f"\n=== Silence Segments ({len(silence_segments)} found) ===")
    for i, seg in enumerate(silence_segments[:10], 1):  # 只显示前10个
        print(f"{i:2d}: {seg.start_time:6.2f}s - {seg.end_time:6.2f}s "
              f"({seg.duration:4.2f}s, confidence: {seg.confidence:.2f}, source: {seg.source})")

    if len(silence_segments) > 10:
        print(f"... and {len(silence_segments) - 10} more segments")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hybrid_vad_detector.py <audio_file> [num_splits]")
        sys.exit(1)

    audio_file = sys.argv[1]
    num_splits = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    test_hybrid_vad(audio_file, num_splits)