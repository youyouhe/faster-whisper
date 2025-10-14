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
        self.srt_merger = SRTMerger()
        self.distributed_threshold_mb = int(os.getenv("DISTRIBUTED_THRESHOLD_MB", "10"))  # MB, configurable
        self.overlap_seconds = float(os.getenv("OVERLAP_SECONDS", "2.0"))  # seconds, configurable

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
        should_distribute = file_size_mb >= self.distributed_threshold_mb and available_workers > 1

        logger.info(f"File size: {file_size_mb:.2f}MB, Available workers: {available_workers}")
        logger.info(f"Should distribute: {should_distribute}")

        return should_distribute

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
        logger.info(f"Starting distributed processing with {len(available_backends)} workers")

        # Determine number of workers to use
        num_workers = min(len(available_backends), 4)  # Max 4 workers for now
        workers_to_use = available_backends[:num_workers]

        logger.info(f"Using {num_workers} workers: {workers_to_use}")

        # Extract audio data from multipart request body
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

        except Exception as e:
            logger.error(f"Failed to extract audio from multipart data: {e}")
            raise RuntimeError(f"Audio extraction failed: {e}")

        # Split audio file
        try:
            audio_file = io.BytesIO(audio_data)
            chunk_data_list = self.audio_splitter.split_with_overlap(
                audio_file, num_workers, self.overlap_seconds
            )

            # Extract chunk data and timing information
            chunk_data = [chunk[0] for chunk in chunk_data_list]
            chunk_timings = [(chunk[1], chunk[2]) for chunk in chunk_data_list]

            logger.info(f"Split audio into {len(chunk_data)} chunks")

        except Exception as e:
            logger.error(f"Failed to split audio: {e}")
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
            raise RuntimeError(f"Distributed processing failed: {e}")

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

            # Check for exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Chunk {i} processing failed: {result}")
                    raise RuntimeError(f"Chunk {i} failed: {result}")

            logger.info("All chunks processed successfully")
            return results

        except Exception as e:
            logger.error(f"Error processing chunks in parallel: {e}")
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
            "overlap_seconds": self.overlap_seconds,
            "max_workers": 4
        }