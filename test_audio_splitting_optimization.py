#!/usr/bin/env python3
"""
Test script for audio splitting and SRT merging improvements
"""

import os
import sys
import logging
from io import BytesIO
from audio_splitter import AudioSplitter
from srt_merger import SRTMerger

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_srt_content(start_offset: float, segments_data: list) -> str:
    """Create mock SRT content for testing"""
    lines = []
    for i, (start_time, end_time, text) in enumerate(segments_data):
        # Add offset to simulate chunk timing
        actual_start = start_offset + start_time
        actual_end = start_offset + end_time

        # Convert to SRT format
        start_srt = seconds_to_srt_time(actual_start)
        end_srt = seconds_to_srt_time(actual_end)

        lines.append(f"{i+1}")
        lines.append(f"{start_srt} --> {end_srt}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)

def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_part = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"

def test_srt_merging():
    """Test the improved SRT merging logic"""
    logger.info("=== Testing SRT Merging Improvements ===")

    # Create test data simulating overlapping chunks
    merger = SRTMerger(overlap_seconds=2.0)

    # Mock chunk timings: (actual_start, actual_end, theoretical_start, theoretical_end)
    chunk_timings = [
        (0.0, 12.0, 0.0, 10.0),      # Chunk 0: 0-10s theoretical, 0-12s actual (2s overlap at end)
        (8.0, 22.0, 10.0, 20.0),     # Chunk 1: 10-20s theoretical, 8-22s actual (2s overlap both sides)
        (18.0, 30.0, 20.0, 30.0)     # Chunk 2: 20-30s theoretical, 18-30s actual (2s overlap at start)
    ]

    # Create mock SRT contents with overlapping and problematic content
    chunk_results = [
        # Chunk 0: Original content
        create_mock_srt_content(0.0, [
            (0.5, 2.0, "这是第一段话的开始"),
            (2.5, 4.0, "这是正常的中间内容"),
            (4.5, 6.0, "那时签下的是功名利禄"),
            (8.0, 10.0, "却不知那是一纸催命符")  # This will overlap with chunk 1
        ]),

        # Chunk 1: Overlapping content, should have duplicates
        create_mock_srt_content(10.0, [
            (0.0, 2.0, "却不知那是一纸催命符"),  # Duplicate from chunk 0
            (1.5, 3.0, "酒醒之后我才看清了"),   # Overlapping with previous
            (4.0, 6.0, "那要命的真相"),        # Overlapping
            (8.0, 10.0, "赤令上的一枚贴黄")     # This will overlap with chunk 2
        ]),

        # Chunk 2: More overlapping content with timing issues
        create_mock_srt_content(20.0, [
            (0.0, 1.5, "赤令上的一枚贴黄"),      # Duplicate from chunk 1
            (2.0, 4.0, "不知何时脱落"),         # Normal content
            (5.0, 7.0, "露出的不是尖"),         # Normal content
            (8.0, 9.5, "而是一个清晰的仙字")     # Normal content
        ])
    ]

    logger.info("Created test data:")
    for i, (content, timing) in enumerate(zip(chunk_results, chunk_timings)):
        actual_start, actual_end, theoretical_start, theoretical_end = timing
        logger.info(f"Chunk {i}: theoretical {theoretical_start}-{theoretical_end}s, actual {actual_start}-{actual_end}s")
        logger.info(f"Content preview: {content[:100]}...")

    # Test merging
    try:
        result_srt = merger.merge_chunk_results(chunk_results, chunk_timings)

        logger.info("\n=== Merged SRT Result ===")
        logger.info(result_srt)

        # Analyze results
        lines = result_srt.strip().split('\n')
        segment_count = len([line for line in lines if line.strip().isdigit()])

        logger.info(f"\n=== Analysis ===")
        logger.info(f"Total segments merged: {segment_count}")

        # Check for timing issues
        time_pattern = r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})'
        matches = re.findall(time_pattern, result_srt)

        timing_issues = 0
        for start_str, end_str in matches:
            start_sec = srt_time_to_seconds(start_str)
            end_sec = srt_time_to_seconds(end_str)
            if end_sec <= start_sec:
                timing_issues += 1
                logger.warning(f"Timing issue found: {start_str} --> {end_str}")

        if timing_issues == 0:
            logger.info("✅ No timing issues detected!")
        else:
            logger.error(f"❌ Found {timing_issues} timing issues!")

        return result_srt

    except Exception as e:
        logger.error(f"Error during merging: {e}")
        import traceback
        traceback.print_exc()
        return None

def srt_time_to_seconds(time_str: str) -> float:
    """Convert SRT time format to seconds"""
    time_part, ms_part = time_str.split(',')
    hours, minutes, seconds = time_part.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(ms_part) / 1000

def test_audio_splitting():
    """Test the improved audio splitting logic"""
    logger.info("\n=== Testing Audio Splitting Improvements ===")

    # Create a mock audio data (would need real audio for full testing)
    splitter = AudioSplitter(sampling_rate=16000)

    # Test the timing calculation logic
    total_duration = 30.0  # 30 seconds
    num_chunks = 3
    overlap_seconds = 2.0

    chunk_duration = total_duration / num_chunks
    overlap_samples = int(overlap_seconds * splitter.sampling_rate)

    logger.info(f"Audio splitting test:")
    logger.info(f"Total duration: {total_duration}s")
    logger.info(f"Number of chunks: {num_chunks}")
    logger.info(f"Chunk duration: {chunk_duration}s")
    logger.info(f"Overlap: {overlap_seconds}s ({overlap_samples} samples)")

    # Calculate theoretical vs actual timing
    for i in range(num_chunks):
        theoretical_start = i * chunk_duration
        theoretical_end = (i + 1) * chunk_duration

        start_sample = max(0, int(theoretical_start * splitter.sampling_rate) - overlap_samples)
        end_sample = min(total_duration * splitter.sampling_rate, int(theoretical_end * splitter.sampling_rate) + overlap_samples)

        actual_start = start_sample / splitter.sampling_rate
        actual_end = end_sample / splitter.sampling_rate

        logger.info(f"Chunk {i}: theoretical {theoretical_start:.1f}-{theoretical_end:.1f}s, "
                   f"actual {actual_start:.1f}-{actual_end:.1f}s")

def main():
    """Main test function"""
    logger.info("Starting audio splitting and SRT merging optimization tests")

    # Test audio splitting logic
    test_audio_splitting()

    # Test SRT merging logic
    test_srt_merging()

    logger.info("=== Test completed ===")

if __name__ == "__main__":
    import re
    main()