#!/usr/bin/env python3
"""
Enhanced Audio Splitter with Data Validation and Smart Split Point Selection
增强版音频分割器，包含数据验证和智能分割点选择
"""

import io
import logging
import struct
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

        # Check minimum size for a recognizable audio header
        min_header_size = 44  # WAV header size (其他格式的魔数检查只需前12字节)
        if len(audio_data) < min_header_size:
            validation_info["issues"].append(f"Audio data too small ({len(audio_data)} bytes < {min_header_size} bytes)")
            logger.error(f"[{context}] Audio validation failed: Data too small for audio format")
            return False, validation_info

        # Check audio format by magic bytes (支持 WAV/MP3/FLAC/OGG/M4A/MP4/AIFF/WebM)
        try:
            header = audio_data[:44]

            is_wav = header.startswith(b'RIFF') and b'WAVE' in header[:12]
            is_mp3 = header.startswith(b'ID3') or (header[0] == 0xFF and (header[1] & 0xE0) == 0xE0)
            is_flac = header.startswith(b'fLaC')
            is_ogg = header.startswith(b'OggS')
            is_mp4 = header[4:8] == b'ftyp'  # M4A/MP4/MOV
            is_aiff = header.startswith(b'FORM') and header[8:12] in (b'AIFF', b'AIFC')
            is_webm = header.startswith(b'\x1a\x45\xdf\xa3')  # EBML (WebM/MKV)

            if not (is_wav or is_mp3 or is_flac or is_ogg or is_mp4 or is_aiff or is_webm):
                validation_info["issues"].append("Unrecognized audio format (not WAV/MP3/FLAC/OGG/M4A/AIFF/WebM)")
                logger.error(f"[{context}] Audio validation failed: Unrecognized audio format")
                return False, validation_info

            # WAV 额外做完整性检查（其他格式的深度校验交给后续 ffmpeg 解码）
            if is_wav:
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

    def validate_audio_file(self, file_path: str, context: str = "unknown") -> Tuple[bool, dict]:
        """
        Validate audio file from file path

        Args:
            file_path: Path to audio file
            context: Context for validation

        Returns:
            Tuple of (is_valid, validation_info)
        """
        try:
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            return self.validate_audio_data(audio_data, context)
        except Exception as e:
            validation_info = {
                "context": context,
                "file_path": file_path,
                "is_valid": False,
                "issues": [f"Failed to read file: {str(e)}"],
                "warnings": []
            }
            logger.error(f"[{context}] File validation failed: {str(e)}")
            return False, validation_info

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

        # Progressive VAD-guided splitting with adaptive thresholds
        try:
            chunks = self._progressive_vad_splitting(audio, num_chunks, total_duration, overlap_seconds, context)
            if chunks:
                logger.info(f"[{context}] Progressive VAD-guided splitting successful: {len(chunks)} chunks")
                self._validate_split_results(chunks, total_duration, context)
                return chunks
            else:
                logger.warning(f"[{context}] Progressive VAD-guided splitting returned no chunks")
        except Exception as e:
            logger.error(f"[{context}] Progressive VAD-guided splitting failed: {e}")
            logger.exception(e)

        # Fallback to enhanced even splitting with pre-decoded audio
        logger.info(f"[{context}] Falling back to enhanced even splitting")
        return self._split_even_enhanced_with_audio(audio, total_duration, num_chunks, overlap_seconds, context)

    def _progressive_vad_splitting(self, audio: np.ndarray, num_chunks: int, total_duration: float,
                                overlap_seconds: float, context: str) -> List[Tuple[bytes, float, float]]:
        """
        Enhanced progressive VAD splitting using hybrid VAD detector
        """
        try:
            # Import and use the hybrid VAD detector
            from hybrid_vad_detector import HybridVADDetector

            # Create hybrid VAD detector with energy-primary strategy
            detector = HybridVADDetector(
                sample_rate=self.sampling_rate,
                energy_threshold=0.01,  # Based on your test results
                min_silence_duration_energy=0.5,  # Start with 0.5s threshold
                hybrid_mode="energy_primary",  # Use energy as primary, validated by silero
                confidence_threshold=0.6
            )

            logger.info(f"[{context}] Using hybrid VAD detector for {num_chunks} chunks")

            # Get optimal split points
            split_points = detector.get_optimal_split_points(audio, num_chunks)

            if len(split_points) >= num_chunks - 1:
                logger.info(f"[{context}] Hybrid VAD found {len(split_points)} optimal split points")
                return self._create_chunks_from_split_points(audio, split_points, total_duration,
                                                          num_chunks, overlap_seconds, context)
            else:
                logger.warning(f"[{context}] Hybrid VAD found only {len(split_points)} split points, need {num_chunks - 1}")

        except ImportError:
            logger.warning(f"[{context}] Hybrid VAD detector not available, falling back to original silero VAD")
        except Exception as e:
            logger.error(f"[{context}] Hybrid VAD detector failed: {e}")

        # Fallback to original progressive silero VAD
        return self._progressive_silero_vad_splitting(audio, num_chunks, total_duration, overlap_seconds, context)

    def _split_with_vad_guidance(self, audio: np.ndarray, num_chunks: int, total_duration: float,
                                  overlap_seconds: float, context: str) -> List[Tuple[bytes, float, float]]:
        """
        Enhanced VAD-guided splitting with better error handling
        Accepts pre-decoded audio data to avoid re-decoding
        """
        try:
            # Create VAD options with current threshold
            current_vad_options = VadOptions(
                min_silence_duration_ms=int(min_silence_duration * 1000),
                max_speech_duration_s=60  # Allow longer speech segments
            )

            # Get speech timestamps using VAD with current threshold
            speech_timestamps = get_speech_timestamps(
                audio=audio,
                vad_options=current_vad_options,
                sampling_rate=self.sampling_rate
            )

            if not speech_timestamps:
                logger.warning(f"[{context}] No speech segments detected by VAD")
                return None

            logger.info(f"[{context}] VAD detected {len(speech_timestamps)} speech segments")

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
            chunks.append((chunk_data, 0.0, first_split['start_time']))

            # Add middle chunks
            for i in range(len(split_points) - 1):
                start_split = split_points[i]
                end_split = split_points[i + 1]

                chunk_data = self._encode_audio_chunk(
                    audio[start_split['sample']:end_split['sample']], i + 1, context
                )
                chunks.append((chunk_data, start_split['start_time'], end_split['start_time']))

            # Add last chunk from last split point to end
            last_split = split_points[-1]
            start_sample = last_split['sample']
            end_sample = len(audio)

            chunk_data = self._encode_audio_chunk(
                audio[start_sample:end_sample], len(split_points), context
            )
            chunks.append((chunk_data, last_split['start_time'], total_duration))

            return chunks

        except Exception as e:
            logger.error(f"[{context}] VAD processing failed: {e}")
            logger.exception(e)
            return None

    def _find_vad_split_points_enhanced(self, speech_segments: List[dict], num_chunks: int,
                                      total_duration: float, context: str) -> List[dict]:
        """
        Adaptive VAD split point finding with progressive silence thresholds
        Strategy: 1.0s → 0.5s → 0.3s
        """
        # Progressive threshold strategy
        thresholds = [1.0, 0.5, 0.3]  # From strict to lenient
        needed_silences = num_chunks - 1

        logger.info(f"[{context}] Adaptive VAD: Need {needed_silences} silences for {num_chunks} chunks")
        logger.info(f"[{context}] Starting progressive threshold detection...")

        for attempt, min_silence_duration in enumerate(thresholds):
            logger.info(f"[{context}] Attempt {attempt + 1}/{len(thresholds)}: {min_silence_duration}s silence threshold")

            # Find silences with current threshold
            silences = []
            for i in range(len(speech_segments) - 1):
                current_end = speech_segments[i]["end"]
                next_start = speech_segments[i + 1]["start"]
                silence_duration = (next_start - current_end) / self.sampling_rate

                if silence_duration >= min_silence_duration:
                    silences.append({
                        "sample": (current_end + next_start) // 2,
                        "duration": silence_duration,
                        "start_time": current_end / self.sampling_rate,
                        "end_time": next_start / self.sampling_rate
                    })

            logger.info(f"[{context}] Found {len(silences)} silences ≥ {min_silence_duration}s")

            # Log silence analysis
            if silences:
                longest_silence = max(s['duration'] for s in silences)
                avg_silence = sum(s['duration'] for s in silences) / len(silences)
                logger.info(f"[{context}] Silence stats: longest={longest_silence:.2f}s, avg={avg_silence:.2f}s")
                silence_durations = [f"{s['duration']:.2f}s" for s in silences[:3]]
                logger.info(f"[{context}] First 3 silences: {silence_durations}")

            # Check if we have enough silences
            if len(silences) >= needed_silences:
                logger.info(f"[{context}] ✅ Success! Found {len(silences)} silences with {min_silence_duration}s threshold")
                logger.info(f"[{context}] Using {min_silence_duration}s threshold for optimal splitting")

                # Use balanced split selection with found silences
                return self._balanced_split_selection(silences, num_chunks, total_duration, context)

            # If not enough silences and not the last threshold
            if attempt < len(thresholds) - 1:
                deficit = needed_silences - len(silences)
                logger.warning(f"[{context}] ⚠️  Need {deficit} more silences, trying lower threshold...")
                continue

            # Last threshold - use what we have or fallback
            logger.error(f"[{context}] ❌ Even with {min_silence_duration}s threshold, only found {len(silences)} silences")
            if len(silences) > 0:
                logger.warning(f"[{context}] Using hybrid approach with available silences")
                return self._hybrid_split_approach(silences, num_chunks, total_duration, context)
            else:
                logger.error(f"[{context}] No silences found - this is continuous speech")
                return None

        # This should never be reached
        logger.error(f"[{context}] Unexpected error in progressive threshold detection")
        return None

        # Sort silences by duration (prefer longer silences)
        silences.sort(key=lambda x: x['duration'], reverse=True)

        # Calculate target chunk duration and tolerance
        target_chunk_duration = total_duration / num_chunks
        tolerance = target_chunk_duration * 0.4  # Allow 40% tolerance for better balance
        split_points = []

        logger.info(f"[{context}] Target chunk duration: {target_chunk_duration:.2f}s ±{tolerance:.2f}s")

        # Smart balanced splitting algorithm
        current_pos = 0.0
        used_silences = set()  # Track used silences to avoid reuse

        for chunk_idx in range(num_chunks - 1):
            ideal_split_time = current_pos + target_chunk_duration

            # Find best silence point considering balance
            best_silence = None
            best_score = float('inf')

            for i, silence in enumerate(silences):
                if i in used_silences:
                    continue  # Skip already used silences

                if silence['start_time'] <= current_pos:
                    continue  # Silence must be after current position

                # Calculate score: distance from ideal + silence quality bonus
                distance_from_ideal = abs(silence['start_time'] - ideal_split_time)
                silence_bonus = silence['duration'] * 0.1  # Prefer longer silences
                score = distance_from_ideal - silence_bonus

                # Only consider if within reasonable range
                if distance_from_ideal <= tolerance and score < best_score:
                    best_score = score
                    best_silence = (i, silence)

            if best_silence:
                # Use the selected silence
                silence_idx, selected_silence = best_silence
                split_points.append(selected_silence)
                used_silences.add(silence_idx)
                current_pos = selected_silence['start_time']

                chunk_duration = selected_silence['start_time'] - (split_points[-2]['start_time'] if len(split_points) > 1 else 0)
                logger.info(f"[{context}] Chunk {chunk_idx+1}: {chunk_duration:.2f}s (silence at {selected_silence['start_time']:.2f}s, duration: {selected_silence['duration']:.2f}s)")
            else:
                # No suitable silence, create synthetic split for balance
                split_time = ideal_split_time
                synthetic_point = {
                    'sample': int(split_time * self.sampling_rate),
                    'time': split_time,
                    'duration': 0,
                    'synthetic': True
                }
                split_points.append(synthetic_point)
                current_pos = split_time

                if len(split_points) > 1:
                    chunk_duration = split_time - split_points[-2].get('time', split_points[-2].get('start_time', 0))
                else:
                    chunk_duration = split_time
                logger.warning(f"[{context}] Chunk {chunk_idx+1}: {chunk_duration:.2f}s (synthetic split at {split_time:.2f}s)")

        # Calculate final chunk and check balance
        final_chunk_duration = total_duration - current_pos
        logger.info(f"[{context}] Final chunk: {final_chunk_duration:.2f}s")

        # Check if chunks are reasonably balanced
        chunk_durations = []
        prev_pos = 0.0
        for point in split_points:
            chunk_durations.append(point['start_time'] - prev_pos)
            prev_pos = point['start_time']
        chunk_durations.append(final_chunk_duration)

        avg_duration = sum(chunk_durations) / len(chunk_durations)
        max_variance = max(abs(d - avg_duration) for d in chunk_durations)
        variance_percent = (max_variance / avg_duration) * 100

        logger.info(f"[{context}] Balance analysis: avg={avg_duration:.2f}s, max variance={max_variance:.2f}s ({variance_percent:.1f}%)")

        if variance_percent > 60:  # If variance is too high, suggest adjusting silence threshold
            logger.warning(f"[{context}] High variance detected ({variance_percent:.1f}%). Consider reducing silence threshold to 0.5s for better balance")

        return split_points[:num_chunks - 1]

    def _hybrid_split_approach(self, silences: List[dict], num_chunks: int,
                              total_duration: float, context: str) -> List[dict]:
        """
        Hybrid approach: combine available silences with synthetic splits
        """
        logger.info(f"[{context}] Hybrid approach: combining {len(silences)} silences with synthetic splits")

        # Calculate ideal split positions
        target_duration = total_duration / num_chunks
        split_points = []

        for i in range(num_chunks - 1):
            target_pos = (i + 1) * target_duration

            # Find closest silence to target position
            best_silence = None
            min_distance = float('inf')

            for silence in silences:
                if silence['start_time'] <= target_pos:
                    distance = target_pos - silence['start_time']
                    if distance < min_distance:
                        min_distance = distance
                        best_silence = silence

            if best_silence and min_distance <= target_duration * 0.4:
                split_points.append(best_silence)
                logger.info(f"[{context}] Using silence at {best_silence['start_time']:.2f}s")
            else:
                # Create synthetic split
                synthetic_point = {
                    'sample': int(target_pos * self.sampling_rate),
                    'time': target_pos,
                    'duration': 0.0,
                    'synthetic': True
                }
                split_points.append(synthetic_point)
                logger.warning(f"[{context}] Using synthetic split at {target_pos:.2f}s")

        return split_points

    def _create_chunks_from_silences(self, audio: np.ndarray, silences: List[dict], num_chunks: int,
                                      overlap_seconds: float, context: str) -> List[Tuple[bytes, float, float]]:
        """
        Create chunks from silence points
        """
        # Sort silences by position
        silences.sort(key=lambda x: x['start_time'])

        # Take only needed silences
        needed_silences = silences[:num_chunks - 1]

        chunks = []
        prev_time = 0.0

        for i, silence in enumerate(needed_silences):
            end_time = silence['start_time']
            start_sample = int(prev_time * self.sampling_rate)
            end_sample = int(end_time * self.sampling_rate)

            chunk_audio = audio[start_sample:end_sample]
            chunk_data = self._encode_audio_chunk(chunk_audio, i, context)

            chunks.append((chunk_data, prev_time, end_time))
            logger.info(f"[{context}] Chunk {i+1}: {end_time - prev_time:.2f}s (silence at {silence['start_time']:.2f}s)")
            prev_time = end_time

        # Add final chunk
        final_chunk_data = self._encode_audio_chunk(audio[int(prev_time * self.sampling_rate):], num_chunks, context)
        chunks.append((final_chunk_data, prev_time, len(audio) / self.sampling_rate))

        return chunks

    def _progressive_silero_vad_splitting(self, audio: np.ndarray, num_chunks: int, total_duration: float,
                                         overlap_seconds: float, context: str) -> List[Tuple[bytes, float, float]]:
        """
        Original progressive silero VAD splitting as fallback
        """
        # Progressive threshold strategy: 1.0s → 0.5s → 0.3s
        thresholds = [1.0, 0.5, 0.3]
        needed_silences = num_chunks - 1

        logger.info(f"[{context}] Progressive silero VAD: Need {needed_silences} silences for {num_chunks} chunks")

        for attempt, min_silence_duration in enumerate(thresholds):
            logger.info(f"[{context}] Attempt {attempt + 1}/{len(thresholds)}: {min_silence_duration}s silence threshold")

            # Create VAD options with current threshold
            current_vad_options = VadOptions(
                min_silence_duration_ms=int(min_silence_duration * 1000),
                max_speech_duration_s=60
            )

            # Get speech timestamps using VAD with current threshold
            speech_timestamps = get_speech_timestamps(
                audio=audio,
                vad_options=current_vad_options,
                sampling_rate=self.sampling_rate
            )

            logger.info(f"[{context}] Silero VAD detected {len(speech_timestamps)} speech segments")

            if not speech_timestamps:
                logger.warning(f"[{context}] No speech segments detected by silero VAD")
                continue

            # Find silences between speech segments
            silences = []
            for i in range(len(speech_timestamps) - 1):
                current_end = speech_timestamps[i]["end"]
                next_start = speech_timestamps[i + 1]["start"]
                silence_duration = (next_start - current_end) / self.sampling_rate

                if silence_duration >= min_silence_duration:
                    silences.append({
                        "sample": (current_end + next_start) // 2,
                        "duration": silence_duration,
                        "start_time": current_end / self.sampling_rate,
                        "end_time": next_start / self.sampling_rate
                    })

            logger.info(f"[{context}] Found {len(silences)} silences ≥ {min_silence_duration}s")

            if len(silences) >= needed_silences:
                logger.info(f"[{context}] ✅ Silero VAD success! Using {min_silence_duration}s threshold")
                return self._create_chunks_from_silences(audio, silences, num_chunks, overlap_seconds, context)

            if attempt < len(thresholds) - 1:
                deficit = needed_silences - len(silences)
                logger.warning(f"[{context}] Need {deficit} more silences, trying lower threshold...")
                continue

        # All thresholds failed
        logger.error(f"[{context}] All silero VAD thresholds failed, falling back to even splitting")
        return None

    def _create_chunks_from_split_points(self, audio: np.ndarray, split_points: List[float],
                                       total_duration: float, num_chunks: int,
                                       overlap_seconds: float, context: str) -> List[Tuple[bytes, float, float]]:
        """
        Create chunks from optimal split points
        """
        logger.info(f"[{context}] Creating {num_chunks} chunks from {len(split_points)} split points")

        # Convert split points to boundaries
        boundaries = [0.0] + split_points + [total_duration]

        # Adjust boundaries to ensure we get exactly num_chunks
        if len(boundaries) - 1 != num_chunks:
            boundaries = self._adjust_boundaries_for_chunks(boundaries, num_chunks, total_duration, context)

        chunks = []
        for i in range(num_chunks):
            start_time = boundaries[i]
            end_time = boundaries[i + 1]

            # Apply overlap (except for first and last chunks)
            if i > 0:
                start_time = max(0.0, start_time - overlap_seconds)
            if i < num_chunks - 1:
                end_time = min(total_duration, end_time + overlap_seconds)

            # Convert time to sample indices
            start_sample = int(start_time * self.sampling_rate)
            end_sample = int(end_time * self.sampling_rate)

            # Extract audio chunk
            chunk_samples = audio[start_sample:end_sample]

            # Encode and add to list
            chunk_data = self._encode_audio_chunk(chunk_samples, i, context)
            chunks.append((chunk_data, start_time, end_time))

            logger.info(f"[{context}] Chunk {i+1}/{num_chunks}: {len(chunk_samples)} samples "
                       f"({start_time:.2f}s - {end_time:.2f}s, duration: {end_time-start_time:.2f}s)")

        return chunks

    def _adjust_boundaries_for_chunks(self, boundaries: List[float], num_chunks: int,
                                    total_duration: float, context: str) -> List[float]:
        """
        Adjust boundaries to create exactly num_chunks chunks
        """
        logger.info(f"[{context}] Adjusting boundaries: have {len(boundaries)-1} segments, need {num_chunks}")

        if len(boundaries) - 1 < num_chunks:
            # Need more chunks - add synthetic splits
            current_chunks = len(boundaries) - 1
            needed_splits = num_chunks - current_chunks

            logger.info(f"[{context}] Adding {needed_splits} synthetic splits to reach {num_chunks} chunks")

            # Find longest segments and split them
            for _ in range(needed_splits):
                max_duration = 0
                split_idx = -1

                for i in range(len(boundaries) - 1):
                    duration = boundaries[i + 1] - boundaries[i]
                    if duration > max_duration:
                        max_duration = duration
                        split_idx = i

                if split_idx >= 0:
                    # Split the longest segment at midpoint
                    mid_point = (boundaries[split_idx] + boundaries[split_idx + 1]) / 2
                    boundaries.insert(split_idx + 1, mid_point)

        elif len(boundaries) - 1 > num_chunks:
            # Too many chunks - merge smallest segments
            current_chunks = len(boundaries) - 1
            needed_merges = current_chunks - num_chunks

            logger.info(f"[{context}] Merging {needed_merges} segments to reach {num_chunks} chunks")

            for _ in range(needed_merges):
                min_duration = float('inf')
                merge_idx = -1

                for i in range(len(boundaries) - 2):
                    duration = boundaries[i + 1] - boundaries[i]
                    if duration < min_duration:
                        min_duration = duration
                        merge_idx = i

                if merge_idx >= 0:
                    # Remove the boundary to merge segments
                    boundaries.pop(merge_idx + 1)

        # Ensure boundaries start at 0 and end at total_duration
        boundaries[0] = 0.0
        boundaries[-1] = total_duration

        logger.info(f"[{context}] Adjusted boundaries: {len(boundaries)-1} chunks")
        return boundaries

    def _hybrid_chunks_from_audio(self, audio: np.ndarray, silences: List[dict], num_chunks: int,
                                  total_duration: float, overlap_seconds: float, context: str) -> List[Tuple[bytes, float, float]]:
        """
        Hybrid approach: combine available silences with synthetic splits
        """
        logger.info(f"[{context}] Hybrid approach: combining {len(silences)} silences with synthetic splits")

        target_duration = total_duration / num_chunks
        split_points = []

        for i in range(num_chunks - 1):
            target_pos = (i + 1) * target_duration

            # Find closest silence to target position
            best_silence = None
            min_distance = float('inf')

            for silence in silences:
                if silence['start_time'] <= target_pos:
                    distance = target_pos - silence['start_time']
                    if distance < min_distance:
                        min_distance = distance
                        best_silence = silence

            if best_silence and min_distance <= target_duration * 0.4:
                split_points.append(best_silence['start_time'])
            else:
                split_points.append(target_pos)

        # Create chunks from split points
        chunks = []
        prev_time = 0.0

        for i, split_time in enumerate(split_points):
            start_sample = int(prev_time * self.sampling_rate)
            end_sample = int(split_time * self.sampling_rate)

            chunk_audio = audio[start_sample:end_sample]
            chunk_data = self._encode_audio_chunk(chunk_audio, i, context)

            chunks.append((chunk_data, prev_time, split_time))
            prev_time = split_time

        # Add final chunk
        final_chunk_data = self._encode_audio_chunk(audio[int(prev_time * self.sampling_rate):], num_chunks, context)
        chunks.append((final_chunk_data, prev_time, total_duration))

        return chunks

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
            end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else len(audio)

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
                logger.warning(f"[{context}] Overlap detected between chunks {i} and {i+1}: {overlap:.2f}s")
            elif next_start > current_end + 0.1:  # Allow 0.1 second gap
                gap = next_start - current_end
                logger.warning(f"[{context}] Gap detected between chunks {i} and {i+1}: {gap:.2f}s")

        # Check chunk sizes
        chunk_sizes = [len(chunk[0]) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        size_variance = sum((size - avg_size) ** 2 for size in chunk_sizes) / len(chunk_sizes)

        if size_variance > avg_size ** 2 * 0.25:  # 25% variance threshold
            logger.warning(f"[{context}] High chunk size variance: avg {avg_size:.0f}, variance {size_variance:.0f}")

        logger.info(f"[{context}] Split validation completed successfully")

    def _split_even_enhanced_with_audio(self, audio: np.ndarray, total_duration: float,
                                       num_chunks: int, overlap_seconds: float,
                                       context: str) -> List[Tuple[bytes, float, float]]:
        """
        Enhanced even splitting with pre-decoded audio data
        """
        total_samples = len(audio)
        chunk_size = total_samples // num_chunks
        overlap_samples = int(overlap_seconds * self.sampling_rate)

        logger.info(f"[{context}] Enhanced even splitting: {total_samples} samples → {num_chunks} chunks ({chunk_size} samples each)")

        chunks = []
        for i in range(num_chunks):
            start_sample = i * chunk_size
            end_sample = start_sample + chunk_size + overlap_samples

            # Ensure we don't exceed audio length
            if end_sample > total_samples:
                end_sample = total_samples

            chunk_audio = audio[start_sample:end_sample]
            chunk_data = self._encode_audio_chunk(chunk_audio, i, context)

            start_time = start_sample / self.sampling_rate
            end_time = min(end_sample, total_samples) / self.sampling_rate

            chunks.append((chunk_data, start_time, end_time))
            logger.info(f"[{context}] Chunk {i+1}: {chunk_size} samples ({start_time:.2f}s - {end_time:.2f}s)")

        return chunks

# Convenience function to create enhanced splitter
def create_enhanced_audio_splitter(sampling_rate: int = 16000) -> EnhancedAudioSplitter:
    """Create an enhanced audio splitter instance"""
    return EnhancedAudioSplitter(sampling_rate=sampling_rate)