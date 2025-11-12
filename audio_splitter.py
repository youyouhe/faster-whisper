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
from faster_whisper.vad import get_speech_timestamps, VadOptions

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

    def split_with_overlap(self, input_file: Union[str, BinaryIO, bytes], num_chunks: int, overlap_seconds: float = 1.0) -> List[Tuple[bytes, float, float]]:
        """
        Split audio with overlap using VAD-guided boundaries (when available)

        Args:
            input_file: Path to input file, file-like object, or bytes data
            num_chunks: Number of chunks
            overlap_seconds: Overlap duration in seconds (reduced to 1.0s)

        Returns:
            List of tuples: (chunk_data, start_time, end_time)
        """
        logger.info(f"Splitting audio with VAD-guided boundaries and {overlap_seconds}s overlap")

        # Handle bytes input
        if isinstance(input_file, bytes):
            input_file = io.BytesIO(input_file)

        # Try VAD-guided splitting first
        try:
            chunks = self._split_with_vad_guidance(input_file, num_chunks, overlap_seconds)
            if chunks:
                logger.info(f"VAD-guided splitting successful: {len(chunks)} chunks")
                return chunks
            else:
                logger.warning("VAD-guided splitting returned no chunks, falling back to even splitting")
        except Exception as e:
            logger.warning(f"VAD-guided splitting failed: {e}, falling back to even splitting")

        # Fallback to even splitting
        return self._split_evenly_with_overlap(input_file, num_chunks, overlap_seconds)

    def _split_with_vad_guidance(self, input_file: Union[str, BinaryIO, bytes], num_chunks: int, overlap_seconds: float) -> List[Tuple[bytes, float, float]]:
        """
        Split audio using VAD-detected speech boundaries
        """
        # Decode audio to get raw samples
        audio = decode_audio(input_file, sampling_rate=self.sampling_rate)
        total_duration = len(audio) / self.sampling_rate

        logger.info(f"Processing audio: {total_duration:.2f}s, target {num_chunks} chunks")

        # Get speech segments using VAD
        vad_options = VadOptions(
            min_silence_duration_ms=800,    # 0.8s minimum silence for splitting
            speech_pad_ms=200,              # 0.2s padding
            min_speech_duration_ms=1000,    # 1.0s minimum speech duration
            threshold=0.5
        )

        speech_segments = get_speech_timestamps(audio, vad_options, self.sampling_rate)
        logger.info(f"VAD detected {len(speech_segments)} speech segments")

        if not speech_segments:
            logger.warning("No speech segments detected by VAD")
            return None

        # Find optimal split points based on VAD segments
        split_points = self._find_vad_split_points(speech_segments, num_chunks, total_duration)

        # Create chunks with minimal overlap
        chunks = []
        overlap_samples = int(overlap_seconds * self.sampling_rate)

        for i, (start_sample, end_sample, start_time, end_time) in enumerate(split_points):
            # Add overlap at boundaries (more conservative than before)
            actual_start_sample = max(0, start_sample - overlap_samples // 2)
            actual_end_sample = min(len(audio), end_sample + overlap_samples // 2)

            # Calculate actual timing
            actual_start_time = actual_start_sample / self.sampling_rate
            actual_end_time = actual_end_sample / self.sampling_rate

            # Extract chunk
            chunk_samples = audio[actual_start_sample:actual_end_sample]
            chunk_data = self._encode_audio_chunk(chunk_samples, i)

            chunks.append((chunk_data, actual_start_time, actual_end_time))

            logger.info(f"VAD Chunk {i}: {actual_start_time:.2f}s - {actual_end_time:.2f}s "
                       f"(target: {start_time:.2f}s - {end_time:.2f}s)")

        return chunks

    def _find_vad_split_points(self, speech_segments: List[dict], num_chunks: int, total_duration: float) -> List[Tuple[int, int, float, float]]:
        """
        Find optimal split points based on VAD segments
        Returns: List of (start_sample, end_sample, start_time, end_time)
        """
        # Find silences between speech segments
        silences = []
        for i in range(len(speech_segments) - 1):
            current_end = speech_segments[i]["end"]
            next_start = speech_segments[i + 1]["start"]
            silence_duration = (next_start - current_end) / self.sampling_rate

            if silence_duration > 0.5:  # Only consider silences longer than 0.5s
                silences.append({
                    "sample": (current_end + next_start) // 2,  # Middle of silence
                    "duration": silence_duration,
                    "start_sample": current_end,
                    "end_sample": next_start
                })

        logger.info(f"Found {len(silences)} suitable silences for splitting")
        for i, silence in enumerate(silences):
            logger.info(f"  Silence {i+1}: {silence['sample']/self.sampling_rate:.2f}s, duration: {silence['duration']:.2f}s")

        if not silences or len(silences) < num_chunks - 1:
            logger.warning(f"Not enough suitable silences ({len(silences)} < {num_chunks-1}), falling back to even splitting")
            return None

        # Sort silences by duration (prefer longer silences)
        silences.sort(key=lambda x: x['duration'], reverse=True)

        # Calculate target chunk duration
        target_chunk_duration = total_duration / num_chunks
        split_points = []

        # Always start from 0
        current_pos = 0
        current_time = 0.0

        # Select split points
        for chunk_idx in range(num_chunks - 1):
            target_end_sample = int((current_time + target_chunk_duration) * self.sampling_rate)

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
                logger.info(f"Chunk {chunk_idx}: Splitting at silence {split_sample/self.sampling_rate:.2f}s "
                           f"(silence duration: {best_silence['duration']:.2f}s)")
            else:
                split_sample = target_end_sample
                logger.warning(f"Chunk {chunk_idx}: No suitable silence, splitting at {split_sample/self.sampling_rate:.2f}s")

            chunk_end_time = current_time + target_chunk_duration
            split_points.append((current_pos, split_sample, current_time, chunk_end_time))

            current_pos = split_sample
            current_time = chunk_end_time

        # Add final chunk
        split_points.append((current_pos, int(total_duration * self.sampling_rate), current_time, total_duration))

        return split_points

    def _split_evenly_with_overlap(self, input_file: Union[str, BinaryIO, bytes], num_chunks: int, overlap_seconds: float) -> List[Tuple[bytes, float, float]]:
        """
        Fallback method: split audio evenly with overlap
        """
        logger.info("Using even splitting with overlap as fallback")

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

            logger.info(f"Even Chunk {i}: {start_time:.2f}s - {end_time:.2f}s, samples: {len(chunk_samples)}")

        return chunks