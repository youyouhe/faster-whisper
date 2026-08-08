#!/usr/bin/env python3
"""
Distributed audio processor for load balancer
Handles splitting large audio files and coordinating multiple workers
"""

import asyncio
import aiohttp
import io
import logging
import os
from typing import List, Tuple, Optional, Dict, Any
from audio_splitter import AudioSplitter
from audio_splitter_enhanced import EnhancedAudioSplitter
from srt_merger import SRTMerger
import re

logger = logging.getLogger(__name__)

async def parse_multipart_data(data: bytes, boundary: str) -> bytes:
    """Parse multipart data and extract audio file content"""
    try:
        # Convert boundary to bytes with proper formatting
        boundary_bytes = f"--{boundary}".encode('utf-8')
        end_boundary_bytes = f"--{boundary}--".encode('utf-8')

        # Find the start of the audio data (after headers)
        # Look for the boundary, then skip headers until we find the empty line
        start_idx = 0
        while start_idx < len(data):
            # Find next boundary
            boundary_idx = data.find(boundary_bytes, start_idx)
            if boundary_idx == -1:
                logger.error("Could not find starting boundary in multipart data")
                raise ValueError("Could not find starting boundary in multipart data")

            # Find the end of headers (double newline)
            header_end_idx = data.find(b'\r\n\r\n', boundary_idx)
            if header_end_idx == -1:
                # Try with just \n\n
                header_end_idx = data.find(b'\n\n', boundary_idx)
                if header_end_idx == -1:
                    start_idx = boundary_idx + len(boundary_bytes)
                    continue

            # Extract content between headers and next boundary
            content_start = header_end_idx + 4 if data[header_end_idx:header_end_idx+4] == b'\r\n\r\n' else header_end_idx + 2
            content_end = data.find(boundary_bytes, content_start)

            # If we found another boundary, this is our content
            if content_end != -1:
                # Check if this section contains the audio file
                # Look for filename in the headers part
                header_part = data[boundary_idx:header_end_idx].decode('utf-8', errors='ignore')
                if 'filename=' in header_part or 'name="audio"' in header_part:
                    # This is the audio content
                    return data[content_start:content_end].strip()

            start_idx = header_end_idx + 4

        logger.error("No audio file found in multipart data")
        raise ValueError("No audio file found in multipart data")

    except Exception as e:
        logger.error(f"Error parsing multipart data: {e}")
        raise


class DistributedProcessor:
    """Coordinates distributed processing of large audio files"""

    def __init__(self):
        self.audio_splitter = AudioSplitter()
        self.enhanced_splitter = EnhancedAudioSplitter()
        self.srt_merger = SRTMerger()
        self.distributed_threshold_mb = int(os.getenv("DISTRIBUTED_THRESHOLD_MB", "10"))  # MB, configurable
        self.overlap_seconds = float(os.getenv("OVERLAP_SECONDS", "0.0"))  # seconds, configurable
        self.min_chunk_size_mb = float(os.getenv("MIN_CHUNK_SIZE_MB", "1.0"))  # MB, minimum chunk size
        self.use_enhanced_splitter = os.getenv("USE_ENHANCED_SPLITTER", "true").lower() == "true"  # Enable enhanced splitter

        # Add concurrency control
        self.max_concurrent_distributed = int(os.getenv("MAX_CONCURRENT_DISTRIBUTED", "3"))  # Allow up to 3 concurrent distributed tasks
        self.current_distributed_count = 0
        self.distributed_lock = asyncio.Lock()

        # Track busy backends during distributed processing
        self.busy_backends = set()
        self.backend_lock = asyncio.Lock()

    async def should_distribute(self, file_size_bytes: int, available_workers: int) -> bool:
        """
        Determine if file should be distributed across workers

        Args:
            file_size_bytes: File size in bytes
            available_workers: Number of available workers

        Returns:
            True if should distribute, False otherwise
        """
        file_size_mb = file_size_bytes / (1024 * 1024)

        # Check basic conditions
        basic_conditions = file_size_mb >= self.distributed_threshold_mb and available_workers > 1

        # Check concurrent distributed processing limit
        async with self.distributed_lock:
            concurrent_ok = self.current_distributed_count < self.max_concurrent_distributed
            if basic_conditions and not concurrent_ok:
                logger.info(f"File size: {file_size_mb:.2f}MB qualifies for distribution, but max concurrent distributed processing ({self.max_concurrent_distributed}) reached")
                logger.info("Falling back to single worker processing")
                return False

        should_distribute = basic_conditions and concurrent_ok

        logger.info(f"File size: {file_size_mb:.2f}MB, Available workers: {available_workers}")
        logger.info(f"Current distributed processing: {self.current_distributed_count}/{self.max_concurrent_distributed}")
        logger.info(f"Should distribute: {should_distribute}")

        return should_distribute

    def calculate_optimal_chunks(self, file_size_mb: float, available_workers: int) -> int:
        """
        Calculate optimal number of chunks based on file size and constraints

        Args:
            file_size_mb: File size in MB
            available_workers: Number of available workers

        Returns:
            Optimal number of chunks
        """
        # Calculate maximum chunks based on minimum chunk size
        max_chunks_by_size = int(file_size_mb / self.min_chunk_size_mb)

        # Calculate optimal chunks (prefer fewer chunks for better efficiency)
        # Target 2-4 chunks per worker for good load balancing
        target_chunks_per_worker = 2
        target_chunks = available_workers * target_chunks_per_worker

        # Choose the minimum of constraints, but at least 1
        optimal_chunks = min(max_chunks_by_size, target_chunks, available_workers)

        # Ensure at least 1 chunk and not more than available workers
        optimal_chunks = max(1, min(optimal_chunks, available_workers))

        logger.info(f"File size: {file_size_mb:.2f}MB, Min chunk size: {self.min_chunk_size_mb}MB")
        logger.info(f"Max chunks by size: {max_chunks_by_size}, Target chunks: {target_chunks}, Available workers: {available_workers}")
        logger.info(f"Optimal chunks: {optimal_chunks}")

        return optimal_chunks

    async def process_distributed(
        self,
        request_body: bytes,
        available_backends: List[str],
        session: aiohttp.ClientSession,
        headers: Dict[str, str]
    ) -> str:
        """
        Process large audio file by distributing across multiple workers

        Args:
            request_body: Original request body (multipart data)
            available_backends: List of available backend URLs
            session: aiohttp ClientSession
            headers: Request headers

        Returns:
            Merged SRT content
        """
        workers_to_use = []  # finally 块兜底：异常发生在赋值前时引用不到
        distributed_count_acquired = False  # 计数器只在真正获取后才允许递减
        try:
            logger.info(f"Starting distributed processing with {len(available_backends)} workers")

            # Extract audio data from multipart request body first
            try:
                # Extract boundary from Content-Type header
                content_type = headers.get('Content-Type', '')
                import re
                boundary_match = re.search(r'boundary=(.+)', content_type)
                if not boundary_match:
                    logger.error("No boundary found in Content-Type header")
                    raise RuntimeError("No boundary found in Content-Type header")

                boundary = boundary_match.group(1)
                logger.info(f"Found boundary: {boundary}")

                # Parse multipart data to extract audio file
                audio_data = await parse_multipart_data(request_body, boundary)
                logger.info(f"Extracted audio file: {len(audio_data)} bytes")

                # Validate extracted audio data
                from audio_splitter_enhanced import EnhancedAudioSplitter
                temp_splitter = EnhancedAudioSplitter()
                is_valid, validation_info = temp_splitter.validate_audio_data(audio_data, "multipart extraction")

                if not is_valid:
                    logger.error(f"Audio data validation failed: {validation_info['issues']}")
                    raise RuntimeError(f"Audio data validation failed: {', '.join(validation_info['issues'])}")

                logger.info(f"Audio data validation passed: {validation_info['data_size_bytes']} bytes")

            except Exception as e:
                logger.error(f"Failed to extract audio from multipart data: {e}")
                raise RuntimeError(f"Audio extraction failed: {e}")

            # Preprocess audio to 16kHz mono format for optimal Whisper performance
            try:
                audio_data = self._preprocess_audio(audio_data)
                logger.info(f"Audio preprocessed to 16kHz mono: {len(audio_data)} bytes")

                # Validate preprocessed audio
                is_valid, validation_info = temp_splitter.validate_audio_data(audio_data, "preprocessing")
                if not is_valid:
                    logger.error(f"Preprocessed audio validation failed: {validation_info['issues']}")
                    logger.warning(f"Continuing with original audio data: {len(audio_data)} bytes")
                    # Revert to original audio if preprocessing failed
                else:
                    logger.info(f"Preprocessed audio validation passed: {validation_info['data_size_bytes']} bytes")

            except Exception as e:
                logger.error(f"Audio preprocessing failed: {e}")
                logger.warning(f"Continuing with original audio data: {len(audio_data)} bytes")
                # Continue with original audio if preprocessing fails

            # Calculate optimal chunk count based on file size and constraints
            file_size_mb = len(audio_data) / (1024 * 1024)
            optimal_chunks = self.calculate_optimal_chunks(file_size_mb, len(available_backends))

            # Determine number of workers to use (use calculated optimal chunks)
            # Use all available workers, but limit to reasonable number to avoid fragmentation
            max_workers = int(os.getenv("MAX_DISTRIBUTED_WORKERS", str(len(available_backends))))
            num_workers = min(optimal_chunks, len(available_backends), max_workers)
            workers_to_use = available_backends[:num_workers]

            logger.info(f"Using {num_workers} workers: {workers_to_use}")

            # Acquire distributed processing lock
            async with self.distributed_lock:
                if self.current_distributed_count >= self.max_concurrent_distributed:
                    raise RuntimeError(f"Maximum concurrent distributed processing ({self.max_concurrent_distributed}) reached")

                # Mark backends as busy before starting processing
                async with self.backend_lock:
                    for backend in workers_to_use:
                        self.busy_backends.add(backend)
                        logger.info(f"Marked backend {backend} as busy for distributed processing")

                # Increment counter
                self.current_distributed_count += 1
                distributed_count_acquired = True
                logger.info(f"Starting distributed processing. Active distributed jobs: {self.current_distributed_count}")

            # Split audio file using enhanced splitter
            try:
                audio_file = io.BytesIO(audio_data)

                # Use enhanced splitter if enabled
                if self.use_enhanced_splitter:
                    logger.info("Using enhanced audio splitter with VAD-guided boundaries")
                    chunk_data_list = self.enhanced_splitter.split_with_vad_guidance_enhanced(
                        audio_file, num_workers, self.overlap_seconds, "distributed processing"
                    )
                else:
                    logger.info("Using standard audio splitter")
                    chunk_data_list = self.audio_splitter.split_with_overlap(
                        audio_file, num_workers, self.overlap_seconds
                    )

                # Extract chunk data and timing information
                chunk_data = [chunk[0] for chunk in chunk_data_list]
                chunk_timings = [(chunk[1], chunk[2]) for chunk in chunk_data_list]

                logger.info(f"Enhanced audio splitting successful: {len(chunk_data)} chunks")

            except Exception as e:
                logger.error(f"Failed to split audio: {e}")
                logger.exception(e)
                raise RuntimeError(f"Audio splitting failed: {e}")

            # Process chunks in parallel
            try:
                chunk_results = await self._process_chunks_parallel(
                    chunk_data, workers_to_use, session, headers
                )

                # Debug: check chunk_results
                logger.info(f"DEBUG: Received {len(chunk_results)} chunk results")
                for i, result in enumerate(chunk_results):
                    logger.info(f"DEBUG: Chunk {i} result length: {len(result) if result else 'None/Empty'}")

                # Merge results
                final_srt = self.srt_merger.merge_chunk_results(chunk_results, chunk_timings)

                # Debug: check final result
                logger.info(f"DEBUG: Final SRT length: {len(final_srt) if final_srt else 'None/Empty'}")
                if final_srt:
                    logger.info("DEBUG: Final SRT preview (first 200 chars):\n" + final_srt[:200])

                logger.info("Distributed processing completed successfully")
                return final_srt

            except Exception as e:
                logger.error(f"Failed during distributed processing: {e}")
                # Don't immediately re-raise, let the finally block clean up first
                raise RuntimeError(f"Distributed processing failed: {e}")

        finally:
            # Always decrement the counter and clear busy backends, even if an exception occurred
            # 注意：只在真正获取过计数器时才递减，否则早期异常会把计数减成负数
            if distributed_count_acquired:
                async with self.distributed_lock:
                    self.current_distributed_count -= 1
                    logger.info(f"Distributed processing completed. Active distributed jobs: {self.current_distributed_count}")

            # Clear busy backends - this is critical for proper cleanup
            # workers_to_use 可能因早期异常仍为空列表，此时无需清理
            if workers_to_use:
                async with self.backend_lock:
                    cleared_backends = []
                    for backend in workers_to_use:
                        if backend in self.busy_backends:
                            self.busy_backends.remove(backend)
                            cleared_backends.append(backend)
                            logger.info(f"Marked backend {backend} as available after distributed processing")
                    if cleared_backends:
                        logger.info(f"Cleared {len(cleared_backends)} backends from busy state: {cleared_backends}")

    async def _process_chunks_parallel(
        self,
        chunk_data: List[bytes],
        workers: List[str],
        session: aiohttp.ClientSession,
        headers: Dict[str, str]
    ) -> List[str]:
        """
        Process audio chunks in parallel on different workers

        Args:
            chunk_data: List of audio chunk data
            workers: List of worker URLs
            session: aiohttp ClientSession
            headers: Request headers

        Returns:
            List of SRT results from each chunk
        """
        logger.info(f"Processing {len(chunk_data)} chunks on {len(workers)} workers")

        # Create tasks for each chunk
        tasks = []
        for i, (chunk, worker) in enumerate(zip(chunk_data, workers)):
            task = self._process_single_chunk(chunk, worker, session, headers, i)
            tasks.append(task)

        # Wait for all tasks to complete
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for exceptions and log detailed errors
            failed_chunks = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Chunk {i} processing failed on worker {workers[i]}: {result}")
                    failed_chunks.append(i)

            # If any chunks failed, we still want to continue with successful ones
            if failed_chunks:
                logger.error(f"Failed chunks: {failed_chunks}, continuing with successful chunks")
                # For now, we'll raise an error to maintain consistency
                raise RuntimeError(f"Some chunks failed: {failed_chunks}")

            logger.info("All chunks processed successfully")
            return results

        except Exception as e:
            logger.error(f"Error processing chunks in parallel: {e}")
            # Don't immediately clear busy backends here, let the finally block in process_distributed handle it
            raise

    async def _process_single_chunk(
        self,
        chunk_data: bytes,
        worker_url: str,
        session: aiohttp.ClientSession,
        headers: Dict[str, str],
        chunk_index: int
    ) -> str:
        """
        Process a single audio chunk on a worker

        Args:
            chunk_data: Audio chunk data
            worker_url: Worker backend URL
            session: aiohttp ClientSession
            headers: Request headers
            chunk_index: Index of the chunk

        Returns:
            SRT result from the chunk
        """
        logger.info(f"Processing chunk {chunk_index} on worker {worker_url}")

        # Manually construct multipart data to match FastAPI expectations
        boundary = f"distributed_chunk_{chunk_index}_{int(asyncio.get_event_loop().time())}"
        multipart_data = self._create_multipart_data(chunk_data, boundary, chunk_index)

        # Create headers for the request
        request_headers = headers.copy()
        request_headers['Content-Type'] = f'multipart/form-data; boundary={boundary}'
        request_headers['Content-Length'] = str(len(multipart_data))
        request_headers.pop('Content-Length', None)  # Remove original to avoid conflicts

        try:
            async with session.post(
                f"{worker_url}/inference",
                data=multipart_data,
                headers=request_headers
            ) as response:
                if response.status == 200:
                    response_text = await response.text()
                    logger.info(f"Chunk {chunk_index} processed successfully on {worker_url}")

                    # Parse JSON response from worker
                    try:
                        import json
                        json_response = json.loads(response_text)
                        if json_response.get('code') == 0:
                            srt_content = json_response.get('data', '')
                            logger.info(f"Chunk {chunk_index} SRT content length: {len(srt_content)}")
                            return srt_content
                        else:
                            error_msg = json_response.get('msg', 'Unknown error')
                            logger.error(f"Worker {worker_url} returned API error: {error_msg}")
                            raise RuntimeError(f"Worker API error: {error_msg}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response from worker {worker_url}: {response_text[:200]}...")
                        raise RuntimeError(f"Invalid JSON response: {e}")
                else:
                    error_text = await response.text()
                    logger.error(f"Worker {worker_url} returned HTTP error {response.status}: {error_text}")
                    raise RuntimeError(f"Worker HTTP error: {response.status} - {error_text}")

        except asyncio.TimeoutError:
            logger.error(f"Chunk {chunk_index} processing timed out on worker {worker_url}")
            raise RuntimeError(f"Chunk {chunk_index} timed out")
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index} on worker {worker_url}: {e}")
            raise

    def _create_multipart_data(self, chunk_data: bytes, boundary: str, chunk_index: int) -> bytes:
        """
        Create properly formatted multipart data for worker

        Args:
            chunk_data: Audio chunk data
            boundary: Multipart boundary
            chunk_index: Chunk index for debugging

        Returns:
            Bytes containing properly formatted multipart data
        """
        parts = [
            f"--{boundary}\r\n".encode(),
            'Content-Disposition: form-data; name="file"; filename="chunk.wav"\r\n'.encode(),
            b'Content-Type: audio/wav\r\n',
            b'\r\n',
            chunk_data,
            b'\r\n',
            f"--{boundary}--\r\n".encode()
        ]

        return b''.join(parts)

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "distributed_threshold_mb": self.distributed_threshold_mb,
            "min_chunk_size_mb": self.min_chunk_size_mb,
            "overlap_seconds": self.overlap_seconds,
            "max_workers": os.getenv("MAX_DISTRIBUTED_WORKERS", "auto"),
            "max_concurrent_distributed": self.max_concurrent_distributed,
            "current_distributed_count": self.current_distributed_count,
            "busy_backends": list(self.busy_backends)
        }

    def _preprocess_audio(self, audio_data: bytes) -> bytes:
        """
        Preprocess audio data to 16kHz mono format for optimal Whisper performance

        Args:
            audio_data: Raw audio data bytes

        Returns:
            Preprocessed audio data bytes
        """
        import subprocess
        import tempfile
        import os

        logger.info("Starting audio preprocessing to 16kHz mono format")

        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input, \
                 tempfile.NamedTemporaryFile(suffix='_16k.wav', delete=False) as temp_output:

                # Write original audio data to temporary input file
                with open(temp_input.name, 'wb') as f:
                    f.write(audio_data)

                # Use FFmpeg to convert to 16kHz mono
                cmd = [
                    'ffmpeg', '-y',
                    '-i', temp_input.name,
                    '-ar', '16000',
                    '-ac', '1',
                    temp_output.name
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )

                logger.info(f"FFmpeg preprocessing output: {result.stdout[:200]}...")
                logger.debug(f"FFmpeg stderr: {result.stderr[:200]}...")

                # Read the converted audio data
                with open(temp_output.name, 'rb') as f:
                    preprocessed_data = f.read()

                # Clean up temporary files
                os.unlink(temp_input.name)
                os.unlink(temp_output.name)

                logger.info(f"Audio preprocessing successful: {len(preprocessed_data)} bytes "
                           f"(original: {len(audio_data)} bytes, "
                           f"size change: {((len(preprocessed_data) - len(audio_data)) / len(audio_data) * 100):+1f}%)")

                return preprocessed_data

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg preprocessing failed: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            raise RuntimeError(f"Audio preprocessing failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during audio preprocessing: {e}")
            raise RuntimeError(f"Audio preprocessing error: {e}")