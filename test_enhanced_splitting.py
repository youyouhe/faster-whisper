#!/usr/bin/env python3
"""
测试增强的音频分割器
"""

import os
import sys
import librosa
import logging
from audio_splitter_enhanced import EnhancedAudioSplitter

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_enhanced_splitting():
    """测试增强的音频分割器"""
    audio_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "66.wav")

    if not os.path.exists(audio_file):
        print(f"Error: Audio file {audio_file} not found")
        return

    try:
        # 创建增强音频分割器
        splitter = EnhancedAudioSplitter(sampling_rate=16000)

        # 设置分割参数
        chunk_size_sec = 150  # 2.5分钟每个chunk
        overlap_seconds = 0.5  # 0.5秒重叠

        print(f"Testing enhanced audio splitter with:")
        print(f"  Audio: {audio_file}")
        print(f"  Target chunk size: {chunk_size_sec}s")
        print(f"  Overlap: {overlap_seconds}s")

        # 计算预期的chunk数量
        audio_duration = librosa.get_duration(path=audio_file)
        expected_chunks = max(2, int(audio_duration / chunk_size_sec))
        print(f"  Audio duration: {audio_duration:.2f}s")
        print(f"  Expected chunks: {expected_chunks}")

        print("\n" + "="*60)
        print("TESTING ENHANCED AUDIO SPLITTING")
        print("="*60)

        # 执行增强分割
        chunks = splitter.split_with_vad_guidance_enhanced(
            input_file=audio_file,
            num_chunks=expected_chunks,
            overlap_seconds=overlap_seconds
        )

        if chunks:
            print(f"\n✅ SUCCESS: Created {len(chunks)} chunks")
            print("-" * 60)
            print(f"{'Chunk':<6} {'Size (bytes)':<12} {'Start':<10} {'End':<10} {'Duration':<10}")
            print("-" * 60)

            total_duration = 0.0
            total_size = 0

            for i, (chunk_data, start_time, end_time) in enumerate(chunks, 1):
                duration = end_time - start_time
                size_mb = len(chunk_data) / (1024 * 1024)

                print(f"{i:<6} {len(chunk_data):<12,} {start_time:<10.2f} {end_time:<10.2f} {duration:<10.2f}")

                total_duration += duration
                total_size += len(chunk_data)

            print("-" * 60)
            print(f"Total duration: {total_duration:.2f}s (original: {audio_duration:.2f}s)")
            print(f"Total size: {total_size / (1024 * 1024):.2f}MB")
            print(f"Average chunk size: {total_size / len(chunks) / (1024 * 1024):.2f}MB")

            # 验证覆盖
            coverage = total_duration / audio_duration
            print(f"Coverage: {coverage:.1%}")

            if abs(coverage - 1.0) < 0.05:  # 5% tolerance
                print("✅ Coverage check passed")
            else:
                print(f"⚠️ Coverage warning: expected ~100%, got {coverage:.1%}")

        else:
            print("❌ FAILED: No chunks created")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_splitting()