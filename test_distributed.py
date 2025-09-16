#!/usr/bin/env python3
"""
Test script for distributed processing functionality
"""

import asyncio
import aiohttp
import logging
import io
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_audio_splitting():
    """Test audio splitting functionality"""
    logger.info("Testing audio splitting functionality")

    # Create a dummy large WAV file for testing
    # For now, we'll simulate with a smaller file
    test_audio_path = "test_audio.wav"

    if not os.path.exists(test_audio_path):
        logger.warning(f"Test audio file {test_audio_path} not found. Creating a dummy file for testing.")
        # Create a dummy WAV file (simplified)
        import wave
        import numpy as np

        # Create a 10-second dummy WAV file at 16kHz
        duration = 10  # seconds
        sample_rate = 16000
        samples = int(duration * sample_rate)

        # Generate sine wave
        frequency = 440  # A4 note
        t = np.linspace(0, duration, samples, False)
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3

        # Convert to int16
        audio_data_int16 = (audio_data * 32767).astype(np.int16)

        # Write WAV file
        with wave.open(test_audio_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data_int16.tobytes())

        logger.info(f"Created dummy WAV file: {test_audio_path}")

    try:
        from audio_splitter import AudioSplitter

        splitter = AudioSplitter()

        # Test splitting into 4 chunks
        with open(test_audio_path, 'rb') as f:
            chunks = splitter.split_audio_file(f, 4)

        logger.info(f"Split into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            logger.info(f"Chunk {i}: {len(chunk)} bytes")

        # Test splitting with overlap
        with open(test_audio_path, 'rb') as f:
            chunks_with_overlap = splitter.split_with_overlap(f, 4, 2.0)

        logger.info(f"Split with overlap into {len(chunks_with_overlap)} chunks")
        for i, (chunk, start, end) in enumerate(chunks_with_overlap):
            logger.info(f"Chunk {i}: {start:.2f}s - {end:.2f}s, {len(chunk)} bytes")

        return True

    except Exception as e:
        logger.error(f"Audio splitting test failed: {e}")
        return False

async def test_srt_merging():
    """Test SRT merging functionality"""
    logger.info("Testing SRT merging functionality")

    try:
        from srt_merger import SRTMerger

        merger = SRTMerger(overlap_seconds=2.0)

        # Create sample SRT content for testing
        chunk1_srt = """1
00:00:00,000 --> 00:00:03,000
Hello world

2
00:00:03,000 --> 00:00:06,000
This is a test

3
00:00:06,000 --> 00:00:09,000
Of the system
"""

        chunk2_srt = """1
00:00:07,000 --> 00:00:10,000
This is chunk two

2
00:00:10,000 --> 00:00:13,000
With overlapping content

3
00:00:13,000 --> 00:00:16,000
That should be merged
"""

        chunk3_srt = """1
00:00:14,000 --> 00:00:17,000
This is chunk three

2
00:00:17,000 --> 00:00:20,000
Final part of test

3
00:00:20,000 --> 00:00:23,000
Merging complete
"""

        # Test parsing
        segments1 = merger.parse_srt_content(chunk1_srt, 0, 0.0)
        segments2 = merger.parse_srt_content(chunk2_srt, 1, 7.0)
        segments3 = merger.parse_srt_content(chunk3_srt, 2, 14.0)

        logger.info(f"Parsed segments: chunk1={len(segments1)}, chunk2={len(segments2)}, chunk3={len(segments3)}")

        # Test merging
        all_segments = [segments1, segments2, segments3]
        chunk_timings = [(0.0, 9.0), (7.0, 16.0), (14.0, 23.0)]

        merged_segments = merger.merge_segments(all_segments, chunk_timings)
        logger.info(f"Merged to {len(merged_segments)} segments")

        # Test SRT generation
        final_srt = merger.generate_srt_content(merged_segments)
        logger.info("Generated SRT content:")
        logger.info(final_srt)

        # Test complete merging workflow
        chunk_results = [chunk1_srt, chunk2_srt, chunk3_srt]
        final_merged_srt = merger.merge_chunk_results(chunk_results, chunk_timings)
        logger.info("Complete merged SRT:")
        logger.info(final_merged_srt)

        return True

    except Exception as e:
        logger.error(f"SRT merging test failed: {e}")
        return False

async def test_distributed_processor():
    """Test distributed processor"""
    logger.info("Testing distributed processor")

    try:
        from distributed_processor import DistributedProcessor

        processor = DistributedProcessor()

        # Test distribution decision logic
        should_distribute = await processor.should_distribute(60 * 1024 * 1024, 4)  # 60MB, 4 workers
        logger.info(f"60MB file with 4 workers should distribute: {should_distribute}")

        should_distribute = await processor.should_distribute(30 * 1024 * 1024, 4)  # 30MB, 4 workers
        logger.info(f"30MB file with 4 workers should distribute: {should_distribute}")

        should_distribute = await processor.should_distribute(60 * 1024 * 1024, 1)  # 60MB, 1 worker
        logger.info(f"60MB file with 1 worker should distribute: {should_distribute}")

        # Get processing stats
        stats = processor.get_processing_stats()
        logger.info(f"Processing stats: {stats}")

        return True

    except Exception as e:
        logger.error(f"Distributed processor test failed: {e}")
        return False

async def test_load_balancer_integration():
    """Test load balancer integration"""
    logger.info("Testing load balancer integration")

    try:
        # Import the load balancer modules
        import sys
        sys.path.append('.')

        # Test that the load balancer can be imported with the new components
        from load_balancer import distributed_processor, get_available_backends, get_healthy_backends

        logger.info("Load balancer modules imported successfully")

        # Test distributed processor
        stats = distributed_processor.get_processing_stats()
        logger.info(f"Distributed processor stats: {stats}")

        return True

    except Exception as e:
        logger.error(f"Load balancer integration test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("Starting distributed processing tests")

    tests = [
        ("Audio Splitting", test_audio_splitting),
        ("SRT Merging", test_srt_merging),
        ("Distributed Processor", test_distributed_processor),
        ("Load Balancer Integration", test_load_balancer_integration)
    ]

    results = {}

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")

        try:
            result = await test_func()
            results[test_name] = result
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"Test {test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            logger.error(f"Test {test_name} failed with exception: {e}")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    asyncio.run(main())