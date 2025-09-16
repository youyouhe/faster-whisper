#!/usr/bin/env python3
"""
Audio splitting utilities for distributed processing
"""

import io
import logging
import numpy as np
import av
from typing import List, Tuple, BinaryIO, Union
from faster_whisper.audio import decode_audio

logger = logging.getLogger(__name__)

class AudioSplitter:
    """Split audio files into chunks for distributed processing"""

    def __init__(self, sampling_rate: int = 16000):
        self.sampling_rate = sampling_rate

    def split_audio_file(self, input_file: Union[str, BinaryIO, bytes], num_chunks: int) -> List[bytes]:
        """
        Split audio file into specified number of chunks

        Args:
            input_file: Path to input file, file-like object, or bytes data
            num_chunks: Number of chunks to split into

        Returns:
            List of audio chunk data as bytes
        """
        logger.info(f"Splitting audio into {num_chunks} chunks")

        # Handle bytes input
        if isinstance(input_file, bytes):
            input_file = io.BytesIO(input_file)

        # Decode audio to get raw samples
        audio = decode_audio(input_file, sampling_rate=self.sampling_rate)

        # Calculate chunk sizes
        total_samples = len(audio)
        chunk_size = total_samples // num_chunks

        logger.info(f"Total samples: {total_samples}, chunk size: {chunk_size}")

        chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else total_samples

            chunk_samples = audio[start_idx:end_idx]
            chunk_data = self._encode_audio_chunk(chunk_samples, i)
            chunks.append(chunk_data)

            logger.info(f"Chunk {i}: {len(chunk_samples)} samples ({len(chunk_data)} bytes)")

        return chunks

    def _encode_audio_chunk(self, audio_chunk: np.ndarray, chunk_index: int) -> bytes:
        """
        Encode audio chunk back to WAV format with proper mono channel handling

        Args:
            audio_chunk: Numpy array of audio samples
            chunk_index: Index of the chunk for metadata

        Returns:
            WAV audio data as bytes
        """
        import wave
        import struct

        # Convert float32 to int16
        audio_int16 = (audio_chunk * 32767).astype(np.int16)

        # Create output buffer
        output_buffer = io.BytesIO()

        # Manually write WAV header for mono audio
        num_channels = 1
        sample_width = 2  # 16-bit = 2 bytes
        num_samples = len(audio_int16)
        sample_rate = self.sampling_rate

        # WAV header
        output_buffer.write(b'RIFF')
        output_buffer.write(struct.pack('<L', 36 + num_samples * sample_width))  # File size - 8
        output_buffer.write(b'WAVE')
        output_buffer.write(b'fmt ')
        output_buffer.write(struct.pack('<L', 16))  # Format chunk size
        output_buffer.write(struct.pack('<H', 1))   # Format (PCM)
        output_buffer.write(struct.pack('<H', num_channels))  # Channels
        output_buffer.write(struct.pack('<L', sample_rate))   # Sample rate
        output_buffer.write(struct.pack('<L', sample_rate * num_channels * sample_width))  # Byte rate
        output_buffer.write(struct.pack('<H', num_channels * sample_width))  # Block align
        output_buffer.write(struct.pack('<H', sample_width * 8))  # Bits per sample
        output_buffer.write(b'data')
        output_buffer.write(struct.pack('<L', num_samples * sample_width))  # Data size

        # Write audio data
        for sample in audio_int16:
            output_buffer.write(struct.pack('<h', sample))

        encoded_data = output_buffer.getvalue()

        # Log detailed information for debugging
        raw_samples = len(audio_chunk)
        sample_size = 2  # 16-bit = 2 bytes per sample
        expected_size = raw_samples * sample_size + 44  # WAV header is 44 bytes
        actual_size = len(encoded_data)

        print(f"DEBUG: Chunk {chunk_index}:")
        print(f"  Raw samples: {raw_samples}")
        print(f"  Expected size: {expected_size} bytes ({expected_size/1024/1024:.2f}MB)")
        print(f"  Actual size: {actual_size} bytes ({actual_size/1024/1024:.2f}MB)")
        print(f"  Overhead: {actual_size - expected_size} bytes ({((actual_size - expected_size)/expected_size)*100:.1f}%)")

        return encoded_data

    def calculate_chunk_overlap(self, audio_duration: float, num_chunks: int, overlap_seconds: float = 2.0) -> Tuple[int, float]:
        """
        Calculate overlap between chunks to avoid cutting words

        Args:
            audio_duration: Total audio duration in seconds
            num_chunks: Number of chunks
            overlap_seconds: Overlap duration in seconds

        Returns:
            Tuple of (overlap_samples, actual_overlap_seconds)
        """
        overlap_samples = int(overlap_seconds * self.sampling_rate)
        actual_overlap_seconds = overlap_samples / self.sampling_rate

        logger.info(f"Calculated overlap: {overlap_samples} samples ({actual_overlap_seconds:.2f}s)")
        return overlap_samples, actual_overlap_seconds

    def split_with_overlap(self, input_file: Union[str, BinaryIO, bytes], num_chunks: int, overlap_seconds: float = 2.0) -> List[Tuple[bytes, float, float]]:
        """
        Split audio with overlap for better transcription accuracy

        Args:
            input_file: Path to input file, file-like object, or bytes data
            num_chunks: Number of chunks
            overlap_seconds: Overlap duration in seconds

        Returns:
            List of tuples: (chunk_data, start_time, end_time)
        """
        logger.info(f"Splitting audio with {overlap_seconds}s overlap")

        # Handle bytes input
        if isinstance(input_file, bytes):
            input_file = io.BytesIO(input_file)

        # Decode audio to get raw samples
        audio = decode_audio(input_file, sampling_rate=self.sampling_rate)

        # Calculate timing
        total_duration = len(audio) / self.sampling_rate
        chunk_duration = total_duration / num_chunks

        # Calculate overlap
        overlap_samples, _ = self.calculate_chunk_overlap(total_duration, num_chunks, overlap_seconds)

        chunks = []
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = (i + 1) * chunk_duration

            # Calculate sample indices with overlap
            start_sample = max(0, int(start_time * self.sampling_rate) - overlap_samples)
            end_sample = min(len(audio), int(end_time * self.sampling_rate) + overlap_samples)

            chunk_samples = audio[start_sample:end_sample]
            chunk_data = self._encode_audio_chunk(chunk_samples, i)

            chunks.append((chunk_data, start_time, end_time))

            logger.info(f"Chunk {i}: {start_time:.2f}s - {end_time:.2f}s, actual samples: {len(chunk_samples)}")

        return chunks