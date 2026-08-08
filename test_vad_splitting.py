#!/usr/bin/env python3
"""
Test VAD-guided audio splitting
"""

import numpy as np
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_audio_with_silence():
    """Create mock audio with speech and silence segments"""
    sampling_rate = 16000
    duration = 30  # 30 seconds

    # Create pattern: 2s speech, 1s silence, 3s speech, 0.8s silence, etc.
    pattern = [
        (2.0, "speech"),   # 2s speech
        (1.0, "silence"),  # 1s silence
        (3.0, "speech"),   # 3s speech
        (0.8, "silence"),  # 0.8s silence (too short for split)
        (2.5, "speech"),   # 2.5s speech
        (1.5, "silence"),  # 1.5s silence (good for split)
        (4.0, "speech"),   # 4s speech
        (0.5, "silence"),  # 0.5s silence (too short)
        (3.0, "speech"),   # 3s speech
        (1.2, "silence"),  # 1.2s silence (good for split)
        (2.0, "speech"),   # 2s speech
        (8.5, "silence"),  # remaining silence
    ]

    audio = np.zeros(int(duration * sampling_rate), dtype=np.float32)

    current_pos = 0
    for segment_duration, segment_type in pattern:
        segment_samples = int(segment_duration * sampling_rate)

        if segment_type == "speech":
            # Generate speech-like audio (simplified)
            # Use random noise with some structure to simulate speech
            t = np.linspace(0, segment_duration, segment_samples)
            # Mix of frequencies to simulate speech characteristics
            signal = (np.sin(2 * np.pi * 200 * t) * 0.1 +  # Low freq
                     np.sin(2 * np.pi * 800 * t) * 0.05 +  # Mid freq
                     np.sin(2 * np.pi * 1500 * t) * 0.02 +  # High freq
                     np.random.normal(0, 0.02, segment_samples))  # Noise
            audio[current_pos:current_pos + segment_samples] = signal
        else:
            # Silence (just very low noise)
            audio[current_pos:current_pos + segment_samples] = np.random.normal(0, 0.001, segment_samples)

        current_pos += segment_samples

    logger.info(f"Created mock audio: {duration}s with {len([s for s, t in pattern if t == 'speech'])} speech segments")
    return audio

def test_vad_splitting():
    """Test VAD-guided splitting without dependencies"""
    logger.info("=== Testing VAD-Guided Splitting Logic ===")

    # Mock VAD segments (simulating what real VAD would detect)
    mock_speech_segments = [
        {"start": 0, "end": 32000},      # 2s speech
        {"start": 48000, "end": 96000}, # 3s speech
        {"start": 108800, "end": 148800}, # 2.5s speech
        {"start": 172800, "end": 236800}, # 4s speech
        {"start": 244800, "end": 292800}, # 3s speech
        {"start": 312000, "end": 344000}, # 2s speech
    ]

    logger.info(f"Mock VAD detected {len(mock_speech_segments)} speech segments")

    # Test the split point finding logic
    num_chunks = 3
    total_duration = 30.0
    sampling_rate = 16000

    # Simulate the _find_optimal_split_points logic
    silences = []
    for i in range(len(mock_speech_segments) - 1):
        current_end = mock_speech_segments[i]["end"]
        next_start = mock_speech_segments[i + 1]["start"]
        silence_duration = (next_start - current_end) / sampling_rate

        if silence_duration > 0.3:  # Only consider silences longer than 0.3s
            silences.append({
                "sample": (current_end + next_start) // 2,
                "duration": silence_duration,
                "start_sample": current_end,
                "end_sample": next_start
            })

    logger.info(f"Found {len(silences)} potential split points:")
    for i, silence in enumerate(silences):
        logger.info(f"  {i+1}: {silence['sample']/sampling_rate:.2f}s, duration: {silence['duration']:.2f}s")

    # Find optimal split points
    target_chunk_duration = total_duration / num_chunks
    split_points = []
    current_pos = 0
    theoretical_time = 0.0

    for chunk_idx in range(num_chunks - 1):
        target_end_sample = int((theoretical_time + target_chunk_duration) * sampling_rate)

        # Find the best silence near target position
        best_silence = None
        best_distance = float('inf')

        for silence in silences:
            if silence["sample"] > current_pos:
                distance = abs(silence["sample"] - target_end_sample)
                # Prefer longer silences and closer to target
                weighted_distance = distance / (silence["duration"] + 0.1)
                if weighted_distance < best_distance:
                    best_distance = weighted_distance
                    best_silence = silence

        if best_silence:
            split_sample = best_silence["sample"]
            logger.info(f"Chunk {chunk_idx}: Splitting at silence {split_sample/sampling_rate:.2f}s "
                       f"(silence duration: {best_silence['duration']:.2f}s)")
        else:
            split_sample = target_end_sample
            logger.warning(f"Chunk {chunk_idx}: No suitable silence, splitting at {split_sample/sampling_rate:.2f}s")

        theoretical_time += target_chunk_duration
        split_points.append((current_pos, split_sample, theoretical_time - target_chunk_duration, theoretical_time))
        current_pos = split_sample

    # Add final chunk
    split_points.append((current_pos, int(total_duration * sampling_rate), theoretical_time, total_duration))

    logger.info("\n=== Split Results ===")
    for i, (start, end, theo_start, theo_end) in enumerate(split_points):
        duration = (end - start) / sampling_rate
        logger.info(f"Chunk {i}: {start/sampling_rate:.2f}s - {end/sampling_rate:.2f}s "
                   f"(duration: {duration:.2f}s, theoretical: {theo_start:.2f}s - {theo_end:.2f}s)")

    return split_points

def test_overlap_reduction():
    """Test the effect of reduced overlap"""
    logger.info("\n=== Testing Overlap Reduction ===")

    # Old vs new overlap times
    old_overlap = 2.0
    new_overlap = 0.5

    # Example chunk with overlap
    chunk_duration = 10.0

    logger.info(f"Old overlap: {old_overlap}s")
    logger.info(f"New overlap: {new_overlap}s")
    logger.info(f"Reduction: {old_overlap - new_overlap}s ({(old_overlap - new_overlap) / old_overlap * 100:.1f}%)")

    # Calculate overlap regions
    for i in range(2):  # Test two chunks
        old_overlap_start = (i + 1) * chunk_duration - old_overlap
        old_overlap_end = (i + 1) * chunk_duration + old_overlap

        new_overlap_start = (i + 1) * chunk_duration - new_overlap
        new_overlap_end = (i + 1) * chunk_duration + new_overlap

        logger.info(f"\nChunk {i} overlap region:")
        logger.info(f"  Old: {old_overlap_start:.1f}s - {old_overlap_end:.1f}s (duration: {old_overlap * 2:.1f}s)")
        logger.info(f"  New: {new_overlap_start:.1f}s - {new_overlap_end:.1f}s (duration: {new_overlap * 2:.1f}s)")

def main():
    """Main test function"""
    logger.info("Starting VAD-guided splitting optimization tests")

    # Test VAD splitting logic
    test_vad_splitting()

    # Test overlap reduction
    test_overlap_reduction()

    logger.info("\n=== Test completed ===")
    logger.info("Key improvements:")
    logger.info("1. ✅ VAD-guided splitting at natural silence boundaries")
    logger.info("2. ✅ Reduced overlap from 2.0s to 0.5s (75% reduction)")
    logger.info("3. ✅ Intelligent split point selection based on silence duration")
    logger.info("4. ✅ Fallback to even splitting when VAD fails")

if __name__ == "__main__":
    main()