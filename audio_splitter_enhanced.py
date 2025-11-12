#!/usr/bin/env python3
"""
Enhanced Audio Splitter with Data Validation and Smart Split Point Selection
增强版音频分割器，包含数据验证和智能分割点选择
"""

import io
import logging
import numpy as np
import av
from typing import List, Tuple, BinaryIO, Union
from faster_whisper.audio import decode_audio
from faster_whisper.vad import get_speech_timestamps, VadOptions

logger = logging.getLogger(__name__)

class EnhancedAudioSplitter:
    """Enhanced audio splitter with comprehensive data validation and smart VAD-guided splitting"""

    def __init__(self, sampling_rate: int = 16000):
        self.sampling_rate = sampling_rate
        self.vad_options = VadOptions(min_silence_duration_ms=500, max_speech_duration_s=30)

    def validate_audio_data(self, audio_data: bytes, context: str = "unknown") -> Tuple[bool, dict]:
        """
        Comprehensive audio data validation

        Args:
            audio_data: Raw audio bytes
            context: Context for validation (e.g., "after upload", "after preprocessing")

        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {
            "context": context,
            "data_size_bytes": len(audio_data) if audio_data else 0,
            "is_empty": len(audio_data) == 0,
            "is_valid": False,
            "issues": [],
            "warnings": []
        }

        # Check for empty data
        if len(audio_data) == 0:
            validation_info["issues"].append("Audio data is empty")
            logger.error(f"[{context}] Audio validation failed: Empty data")
            return False, validation_info

        # Check minimum size for WAV processing
        min_wav_size = 44  # WAV header size
        if len(audio_data) < min_wav_size:
            validation_info["issues"].append(f"Audio data too small ({len(audio_data)} bytes < {min_wav_size} bytes)")
            logger.error(f"[{context}] Audio validation failed: Data too small for WAV format")
            return False, validation_info

        # Check WAV header
        try:
            header = audio_data[:44]
            if not (header.startswith(b'RIFF') and b'WAVE' in header[:12]):
                validation_info["issues"].append("Invalid WAV header format")
                logger.error(f"[{context}] Audio validation failed: Invalid WAV header")
                return False, validation_info

            # Extract file size from header
            import struct
            data_size = struct.unpack('<I', header[4:8])[0]
            expected_size = data_size + 8
            if len(audio_data) < expected_size:
                validation_info["warnings"].append(f"Audio file truncated (actual: {len(audio_data)}, expected: {expected_size})")
                logger.warning(f"[{context}] Audio warning: File may be truncated")

            validation_info["data_size_bytes"] = len(audio_data)
            validation_info["is_valid"] = True

        except Exception as e:
            validation_info["issues"].append(f"Header parsing error: {e}")
            logger.error(f"[{context}] Audio validation failed: Header parsing error - {e}")
            return False, validation_info

        logger.info(f"[{context}] Audio validation passed: {len(audio_data)} bytes")
        return True, validation_info

    def decode_audio_safely(self, input_file: Union[str, BinaryIO, bytes], context: str = "unknown") -> Tuple[bool, np.ndarray, dict]:
        """
        Safe audio decoding with comprehensive error handling

        Args:
            input_file: Audio input source
            context: Context for decoding

        Returns:
            Tuple of (success, audio_array, decode_info)
        """
        decode_info = {
            "context": context,
            "success": False,
            "error": None,
            "shape": None,
            "duration_seconds": None,
            "sample_rate": None
        }

        # Validate input data first
        if isinstance(input_file, bytes):
            is_valid, validation_info = self.validate_audio_data(input_file, f"{context} - input validation")
            if not is_valid:
                decode_info["error"] = f"Input validation failed: {', '.join(validation_info['issues'])}"
                return False, None, decode_info

            input_file = io.BytesIO(input_file)

        try:
            # Attempt audio decoding
            audio = decode_audio(input_file, sampling_rate=self.sampling_rate)

            if audio is None:
                decode_info["error"] = "decode_audio returned None"
                logger.error(f"[{context}] Audio decoding failed: decode_audio returned None")
                return False, None, decode_info

            # Validate decoded audio
            if len(audio) == 0:
                decode_info["error"] = "Decoded audio is empty"
                logger.error(f"[{context}] Audio decoding failed: Decoded audio is empty")
                return False, None, decode_info

            decode_info["success"] = True
            decode_info["shape"] = audio.shape
            decode_info["duration_seconds"] = len(audio) / self.sampling_rate
            decode_info["sample_rate"] = self.sampling_rate

            logger.info(f"[{context}] Audio decoding successful: {len(audio)} samples, {decode_info['duration_seconds']:.2f}s")
            return True, audio, decode_info

        except Exception as e:
            decode_info["error"] = str(e)
            logger.error(f"[{context}] Audio decoding failed: {e}")
            logger.exception(e)
            return False, None, decode_info

    def calculate_smart_chunks(self, audio_duration: float, available_workers: int,
                             file_size_mb: float, min_chunk_duration: float = 60.0) -> int:
        """
        Calculate optimal number of chunks based on multiple factors

        Args:
            audio_duration: Total audio duration in seconds
            available_workers: Number of available workers
            file_size_mb: File size in MB
            min_chunk_duration: Minimum chunk duration in seconds

        Returns:
            Optimal number of chunks
        """
        logger.info(f"Smart chunk calculation:")
        logger.info(f"  Audio duration: {audio_duration:.2f}s")
        logger.info(f"  File size: {file_size_mb:.2f}MB")
        logger.info(f"  Available workers: {available_workers}")
        logger.info(f"  Min chunk duration: {min_chunk_duration}s")

        # Calculate constraints
        max_chunks_by_duration = int(audio_duration / min_chunk_duration)
        max_chunks_by_workers = available_workers
        max_chunks_by_size = int(file_size_mb / 2.0)  # Max 2MB per chunk for stability

        # Target 2-4 chunks per worker for good load balancing
        target_chunks_per_worker = 2
        target_chunks = available_workers * target_chunks_per_worker

        # Choose optimal chunks considering all constraints
        optimal_chunks = min(
            max_chunks_by_duration,
            max_chunks_by_workers,
            max_chunks_by_size,
            target_chunks
        )

        # Ensure at least 1 chunk
        optimal_chunks = max(1, optimal_chunks)

        logger.info(f"  Constraints: duration={max_chunks_by_duration}, workers={max_chunks_by_workers}, size={max_chunks_by_size}, target={target_chunks}")
        logger.info(f"  Optimal chunks: {optimal_chunks}")

        return optimal_chunks

    def split_with_vad_guidance_enhanced(self, input_file: Union[str, BinaryIO, bytes],
                                        num_chunks: int, overlap_seconds: float = 1.0,
                                        context: str = "unknown") -> List[Tuple[bytes, float, float]]:
        """
        Enhanced VAD-guided audio splitting with comprehensive validation

        Args:
            input_file: Audio input source
            num_chunks: Number of chunks to create
            overlap_seconds: Overlap duration
            context: Processing context

        Returns:
            List of tuples: (chunk_data, start_time, end_time)
        """
        logger.info(f"[{context}] Enhanced VAD-guided splitting starting")
        logger.info(f"  Target chunks: {num_chunks}, Overlap: {overlap_seconds}s")

        # Validate and decode audio
        if isinstance(input_file, bytes):
            is_valid, validation_info = self.validate_audio_data(input_file, f"{context} - pre-decoding")
            if not is_valid:
                raise RuntimeError(f"Audio validation failed: {', '.join(validation_info['issues'])}")
            input_file = io.BytesIO(input_file)

        success, audio, decode_info = self.decode_audio_safely(input_file, context)
        if not success:
            raise RuntimeError(f"Audio decoding failed: {decode_info['error']}")

        total_samples = len(audio)
        total_duration = total_samples / self.sampling_rate

        logger.info(f"[{context}] Audio decoded successfully: {total_samples} samples ({total_duration:.2f}s)")

        # Re-calculate optimal chunks based on actual audio duration
        actual_optimal_chunks = self.calculate_smart_chunks(
            total_duration, num_chunks, total_samples * 2 / (1024 * 1024)
        )

        if actual_optimal_chunks != num_chunks:
            logger.warning(f"[{context}] Adjusting chunks from {num_chunks} to {actual_optimal_chunks}")
            num_chunks = actual_optimal_chunks

        # Try VAD-guided splitting
        try:
            chunks = self._split_with_vad_guidance(input_file, num_chunks, overlap_seconds, context)
            if chunks:
                logger.info(f"[{context}] VAD-guided splitting successful: {len(chunks)} chunks")
                self._validate_split_results(chunks, total_duration, context)
                return chunks
            else:
                logger.warning(f"[{context}] VAD-guided splitting returned no chunks")
        except Exception as e:
            logger.error(f"[{context}] VAD-guided splitting failed: {e}")
            logger.exception(e)

        # Fallback to enhanced even splitting
        logger.info(f"[{context}] Falling back to enhanced even splitting")
        return self._split_even_enhanced(input_file, num_chunks, overlap_seconds, context)

    def _split_with_vad_guidance(self, input_file: Union[str, BinaryIO, bytes],
                                  num_chunks: int, overlap_seconds: float,
                                  context: str) -> List[Tuple[bytes, float, float]]:
        """
        Enhanced VAD-guided splitting with better error handling
        """
        try:
            # Get speech timestamps using VAD
            speech_timestamps = get_speech_timestamps(
                audio_file_or_path=input_file,
                vad_options=self.vad_options
            )

            if not speech_timestamps:
                logger.warning(f"[{context}] No speech segments detected by VAD")
                return None

            logger.info(f"[{context}] VAD detected {len(speech_timestamps)} speech segments")

            # Calculate total duration from VAD results
            total_duration = speech_timestamps[-1]['end'] / self.speech_timestamps[0]['sample_rate']

            # Use existing VAD split point logic
            split_points = self._find_vad_split_points_enhanced(
                speech_timestamps, num_chunks, total_duration, context
            )

            if not split_points:
                logger.warning(f"[{context}] Could not find suitable VAD split points")
                return None

            # Create chunks based on VAD split points
            chunks = []

            # Add first chunk from start to first split point
            first_split = split_points[0]
            start_sample = 0
            end_sample = first_split['sample']

            chunk_data = self._encode_audio_chunk(
                audio[start_sample:end_sample], 0, context
            )
            chunks.append((chunk_data, 0.0, first_split['time']))

            # Add middle chunks
            for i in range(len(split_points) - 1):
                start_split = split_points[i]
                end_split = split_points[i + 1]

                chunk_data = self._encode_audio_chunk(
                    audio[start_split['sample']:end_split['sample']], i + 1, context
                )
                chunks.append((chunk_data, start_split['time'], end_split['time']))

            # Add last chunk from last split point to end
            last_split = split_points[-1]
            start_sample = last_split['sample']
            end_sample = total_samples

            chunk_data = self._encode_audio_chunk(
                audio[start_sample:end_sample], len(split_points), context
            )
            chunks.append((chunk_data, last_split['time'], total_duration))

            return chunks

        except Exception as e:
            logger.error(f"[{context}] VAD processing failed: {e}")
            logger.exception(e)
            return None

    def _find_vad_split_points_enhanced(self, speech_segments: List[dict], num_chunks: int,
                                      total_duration: float, context: str) -> List[dict]:
        """
        Enhanced VAD split point finding with better validation
        """
        # Find silences between speech segments
        silences = []
        min_silence_duration = 1.0  # Increased minimum silence duration

        for i in range(len(speech_segments) - 1):
            current_end = speech_segments[i]["end"]
            next_start = speech_segments[i + 1]["start"]
            silence_duration = (next_start - current_end) / speech_segments[0]["sample_rate"]

            if silence_duration >= min_silence_duration:
                silences.append({
                    "sample": (current_end + next_start) // 2,  # Middle of silence
                    "duration": silence_duration,
                    "start_sample": current_end,
                    "end_sample": next_start,
                    "start_time": current_end / speech_segments[0]["sample_rate"],
                    "end_time": next_start / speech_segments[0]["sample_rate"]
                })

        logger.info(f"[{context}] Found {len(silences)} suitable silences (>= {min_silence_duration}s)")
        for i, silence in enumerate(silences[:5]):  # Log first 5
            logger.info(f"[{context}] Silence {i+1}: {silence['start_time']:.2f}s-{silence['end_time']:.2f}s, duration: {silence['duration']:.2f}s")

        if len(silences) > 0:
            logger.info(f"[{context}] Longest silence: {max(s['duration'] for s in silences):.2f}s")

        # Check if we have enough silences
        if len(silences) < num_chunks - 1:
            available_silences = len(silences)
            logger.warning(f"[{context}] Not enough silences for {num_chunks} chunks (need {num_chunks-1}, have {available_silences})")

            # If we have some silences, use them even if not optimal
            if available_silences > 0:
                logger.info(f"[{context}] Using available {available_silences} silences for partial VAD splitting")
            else:
                logger.warning(f"[{context}] No suitable silences found, falling back to even splitting")
                return None

        # Sort silences by duration (prefer longer silences)
        silences.sort(key=lambda x: x['duration'], reverse=True)

        # Calculate target chunk duration
        target_chunk_duration = total_duration / num_chunks
        split_points = []

        logger.info(f"[{context}] Target chunk duration: {target_chunk_duration:.2f}s")

        # Smart split point selection
        current_pos = 0.0
        chunks_created = 0

        for chunk_idx in range(num_chunks - 1):
            target_end_time = current_pos + target_chunk_duration

            # Find the best silence near target position
            best_silence = None
            best_distance = float('inf')

            for silence in silences:
                # Check if silence is available and not already used
                if (silence['start_time'] > current_pos and
                    silence['end_time'] < target_end_time + target_chunk_duration * 0.3):  # Allow some flexibility

                    distance = abs(silence['sample'] / self.sampling_rate - (current_pos + target_chunk_duration / 2))

                    if distance < best_distance:
                        best_distance = distance
                        best_silence = silence

            if best_silence:
                # Use the best silence found
                split_point = best_silence
                split_points.append(split_point)
                current_pos = split_point['end_time']
                chunks_created += 1
                logger.info(f"[{context] Chunk {chunk_idx+1}: Using silence at {split_point['time']:.2f}s (duration: {split_point['duration']:.2f}s)")
            else:
                # No suitable silence, create even split
                split_time = current_pos + target_chunk_duration
                split_sample = int(split_time * self.sampling_rate)

                split_points.append({
                    'sample': split_sample,
                    'time': split_time,
                    'duration': 0,
                    'synthetic': True
                })
                current_pos = split_time
                chunks_created += 1
                logger.warning(f"[{context] Chunk {chunk_idx+1}: Using synthetic split at {split_time:.2f}s (no suitable silence)")

        logger.info(f"[{context}] Created {chunks_created} VAD-guided chunks out of {num_chunks-1} needed")

        return split_points[:num_chunks - 1]  # Return only needed split points

    def _split_even_enhanced(self, input_file: Union[str, BinaryIO, bytes],
                           num_chunks: int, overlap_seconds: float,
                           context: str) -> List[Tuple[bytes, float, float]]:
        """
        Enhanced even splitting with validation
        """
        # Validate and decode audio first
        if isinstance(input_file, bytes):
            is_valid, validation_info = self.validate_audio_data(input_file, f"{context} - even splitting")
            if not is_valid:
                raise RuntimeError(f"Audio validation failed: {', '.join(validation_info['issues'])}")
            input_file = io.BytesIO(input_file)

        success, audio, decode_info = self.decode_audio_safely(input_file, context)
        if not success:
            raise RuntimeError(f"Audio decoding failed: {decode_info['error']}")

        total_samples = len(audio)
        chunk_size = total_samples // num_chunks

        logger.info(f"[{context}] Enhanced even splitting: {total_samples} samples → {num_chunks} chunks ({chunk_size} samples each)")

        chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else total_samples

            chunk_samples = audio[start_idx:end_idx]
            start_time = start_idx / self.sampling_rate
            end_time = end_idx / self.sampling_rate

            chunk_data = self._encode_audio_chunk(chunk_samples, i, context)
            chunks.append((chunk_data, start_time, end_time))

            logger.info(f"[{context}] Chunk {i+1}: {len(chunk_samples)} samples ({start_time:.2f}s - {end_time:.2f}s)")

        return chunks

    def _encode_audio_chunk(self, audio_chunk: np.ndarray, chunk_index: int, context: str) -> bytes:
        """
        Enhanced audio chunk encoding with validation
        """
        # Validate chunk data
        if len(audio_chunk) == 0:
            raise ValueError(f"[{context}] Audio chunk {chunk_index} is empty")

        if len(audio_chunk) < 100:  # Minimum 100 samples
            raise ValueError(f"[{context}] Audio chunk {chunk_index} too small: {len(audio_chunk)} samples")

        # Convert float32 to int16
        try:
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
        except Exception as e:
            raise ValueError(f"[{context}] Failed to convert audio to int16: {e}")

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

        # Validate encoded data
        expected_size = len(audio_chunk) * 2 + 44  # 16-bit samples + WAV header
        actual_size = len(encoded_data)

        if actual_size != expected_size:
            logger.warning(f"[{context}] Chunk {chunk_index} size mismatch: expected {expected_size}, actual {actual_size}")

        logger.debug(f"[{context}] Chunk {chunk_index} encoded: {actual_size} bytes")

        return encoded_data

    def _validate_split_results(self, chunks: List[Tuple[bytes, float, float]],
                              expected_duration: float, context: str) -> None:
        """
        Validate split results for consistency
        """
        if not chunks:
            raise ValueError(f"[{context}] No chunks created")

        # Check chunk count
        logger.info(f"[{context}] Validating {len(chunks)} chunks against expected duration {expected_duration:.2f}s")

        # Check total duration
        total_duration = chunks[-1][2]  # End time of last chunk
        duration_diff = abs(total_duration - expected_duration)

        if duration_diff > 1.0:  # Allow 1 second difference
            logger.warning(f"[{context}] Duration mismatch: expected {expected_duration:.2f}s, got {total_duration:.2f}s (diff: {duration_diff:.2f}s)")

        # Check for gaps or overlaps
        for i in range(len(chunks) - 1):
            current_end = chunks[i][2]
            next_start = chunks[i + 1][1]

            if next_start < current_end:
                overlap = current_end - next_start
                logger.warning(f"[{context] Overlap detected between chunks {i} and {i+1}: {overlap:.2f}s")
            elif next_start > current_end + 0.1:  # Allow 0.1 second gap
                gap = next_start - current_end
                logger.warning(f"[{context] Gap detected between chunks {i} and {i+1}: {gap:.2f}s")

        # Check chunk sizes
        chunk_sizes = [len(chunk[0]) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        size_variance = sum((size - avg_size) ** 2 for size in chunk_sizes) / len(chunk_sizes)

        if size_variance > avg_size ** 2 * 0.25:  # 25% variance threshold
            logger.warning(f"[{context] High chunk size variance: avg {avg_size:.0f}, variance {size_variance:.0f}")

        logger.info(f"[{context}] Split validation completed successfully")

# Convenience function to create enhanced splitter
def create_enhanced_audio_splitter(sampling_rate: int = 16000) -> EnhancedAudioSplitter:
    """Create an enhanced audio splitter instance"""
    return EnhancedAudioSplitter(sampling_rate=sampling_rate)