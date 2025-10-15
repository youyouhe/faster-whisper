#!/usr/bin/env python3
"""
FastAPI service for faster-whisper ASR with SRT support
Compatible with existing client calls
"""

import os
import tempfile
import uuid
import time
from typing import Optional, List, Tuple
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
from faster_whisper import WhisperModel
import re
import asyncio
from collections import deque
import signal

# Audio processing libraries
import pydub
from pydub import AudioSegment
import numpy as np

# Import authentication middleware
from auth_middleware import get_auth, require_auth

# Statistics tracking
from dataclasses import dataclass, field
from typing import Dict, Any
import threading
import json

@dataclass
class ProcessingStats:
    """处理统计数据类"""
    # 基本统计
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # 文件处理统计
    total_files_processed: int = 0
    total_file_size_mb: float = 0.0
    total_chunks_processed: int = 0

    # 时间统计
    total_processing_time: float = 0.0
    total_upload_time: float = 0.0

    # 性能统计
    average_processing_speed: float = 0.0  # MB/s
    peak_memory_usage: float = 0.0

    # Chunk详细统计
    chunk_stats: Dict[str, Any] = field(default_factory=dict)

    # 实例信息
    instance_id: str = ""
    gpu_device: str = ""
    port: int = 0
    start_time: float = field(default_factory=time.time)

# 全局统计实例
stats = ProcessingStats()
stats_lock = threading.Lock()

def update_stats(event_type: str, **kwargs):
    """更新统计数据"""
    global stats
    with stats_lock:
        if event_type == "request_start":
            stats.total_requests += 1

        elif event_type == "request_success":
            stats.successful_requests += 1
            if 'processing_time' in kwargs:
                stats.total_processing_time += kwargs['processing_time']
            if 'file_size_mb' in kwargs:
                stats.total_file_size_mb += kwargs['file_size_mb']
                stats.total_files_processed += 1

        elif event_type == "request_failed":
            stats.failed_requests += 1

        elif event_type == "upload_complete":
            if 'upload_time' in kwargs:
                stats.total_upload_time += kwargs['upload_time']

        elif event_type == "chunk_processed":
            stats.total_chunks_processed += 1

        elif event_type == "memory_usage":
            if 'memory_mb' in kwargs:
                stats.peak_memory_usage = max(stats.peak_memory_usage, kwargs['memory_mb'])

        # 更新平均处理速度
        if stats.total_files_processed > 0 and stats.total_processing_time > 0:
            stats.average_processing_speed = stats.total_file_size_mb / (stats.total_processing_time / 3600)  # MB/hour

def get_gpu_display_name() -> str:
    """获取GPU设备的显示名称"""
    if device == "cpu":
        return "cpu"

    # 尝试获取具体的GPU ID
    try:
        import torch
        if torch.cuda.is_available():
            # 如果设置了CUDA_VISIBLE_DEVICES，显示可见的GPU ID
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                visible_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
                return f"cuda:{visible_devices[0]}" if visible_devices else "cuda"
            else:
                # 否则显示实际的GPU设备ID
                gpu_id = os.getenv("GPU_DEVICE_ID", "0")
                return f"cuda:{gpu_id}"
    except:
        pass

    return device

def get_instance_info() -> Dict[str, Any]:
    """获取实例信息"""
    return {
        "instance_id": os.getenv("INSTANCE_ID", "default"),
        "gpu_device": get_gpu_display_name(),
        "port": int(os.getenv("API_PORT", "5001")),
        "model_size": model_size,
        "uptime_seconds": time.time() - stats.start_time
    }

def get_stats() -> Dict[str, Any]:
    """获取统计数据"""
    with stats_lock:
        instance_info = get_instance_info()
        uptime = time.time() - stats.start_time

        # 计算成功率
        success_rate = (stats.successful_requests / stats.total_requests * 100) if stats.total_requests > 0 else 0

        # 计算平均处理时间
        avg_processing_time = (stats.total_processing_time / stats.successful_requests) if stats.successful_requests > 0 else 0

        # 计算吞吐量（请求/小时）
        throughput = stats.total_requests / (uptime / 3600) if uptime > 0 else 0

        return {
            "instance_info": instance_info,
            "uptime_seconds": uptime,
            "request_stats": {
                "total_requests": stats.total_requests,
                "successful_requests": stats.successful_requests,
                "failed_requests": stats.failed_requests,
                "success_rate_percent": round(success_rate, 2),
                "throughput_requests_per_hour": round(throughput, 2)
            },
            "file_stats": {
                "total_files_processed": stats.total_files_processed,
                "total_file_size_mb": round(stats.total_file_size_mb, 2),
                "total_chunks_processed": stats.total_chunks_processed,
                "average_file_size_mb": round(stats.total_file_size_mb / stats.total_files_processed, 2) if stats.total_files_processed > 0 else 0,
                "average_chunks_per_file": round(stats.total_chunks_processed / stats.total_files_processed, 2) if stats.total_files_processed > 0 else 0
            },
            "performance_stats": {
                "total_processing_time": round(stats.total_processing_time, 2),
                "total_upload_time": round(stats.total_upload_time, 2),
                "average_processing_time_seconds": round(avg_processing_time, 2),
                "average_processing_speed_mb_per_hour": round(stats.average_processing_speed, 2),
                "peak_memory_usage_mb": round(stats.peak_memory_usage, 2)
            },
            "current_status": {
                "queue_length": len(task_queue),
                "currently_processing": current_processing_tasks,
                "max_concurrent_tasks": max_concurrent_tasks
            }
        }

# Simplified middleware - only add necessary headers without touching response body
class SimpleResponseMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Only update headers, don't modify response body
        if hasattr(response, 'headers'):
            response.headers["Connection"] = "keep-alive"

        return response

# Initialize FastAPI app with larger client size limit and response handling
app = FastAPI(
    title="faster-whisper ASR Service",
    version="1.0.0",
    client_max_size=500*1024*1024,  # 500MB limit
)

# Add middleware
app.add_middleware(SimpleResponseMiddleware)

# Global model instance
model = None
model_size = "large-v3-turbo"  # Default model size
# Get GPU device ID from environment variable
gpu_device_id = os.getenv('GPU_DEVICE_ID', '0')

# Check if CUDA_VISIBLE_DEVICES is set
import os
if 'CUDA_VISIBLE_DEVICES' in os.environ:
    # When CUDA_VISIBLE_DEVICES is set, each process can only see its assigned GPU(s)
    # In this case, we should use "cuda" without specifying device ID for ctranslate2
    visible_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    print(f"CUDA_VISIBLE_DEVICES set to {visible_devices}, using cuda")
    if gpu_device_id == "cpu":
        device = "cpu"
    else:
        # With CUDA_VISIBLE_DEVICES set, we use "cuda" without device ID for ctranslate2
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
                print("CUDA not available, falling back to CPU")
        except Exception as e:
            device = "cpu"
            print(f"GPU not usable, falling back to CPU: {e}")
else:
    # No CUDA_VISIBLE_DEVICES, use direct device specification
    # Check if CUDA is available before using GPU
    try:
        import torch
        if torch.cuda.is_available() and gpu_device_id != "cpu":
            # Use default CUDA device without specifying ID
            try:
                device = "cuda"
                # Test if the device is actually usable
                torch.zeros(1).to(device)
            except Exception as e:
                device = "cpu"
                print(f"GPU not usable due to compatibility issues, falling back to CPU: {e}")
        else:
            device = "cpu"
            if gpu_device_id == "cpu":
                print("CPU device explicitly requested")
            else:
                print("CUDA not available, falling back to CPU")
    except (ImportError, ValueError) as e:
        device = "cpu"
        print(f"PyTorch not available or invalid GPU device ID, falling back to CPU: {e}")

# Task queue configuration
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "10"))  # Default queue size of 10
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "20"))  # Default max file size in MB
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "3600"))  # Request timeout in seconds (60 minutes)
CHUNK_TIMEOUT = int(os.getenv("CHUNK_TIMEOUT", "3600"))  # Individual chunk timeout in seconds (60 minutes - for large files)
task_queue = deque()
processing_lock = asyncio.Lock()
current_processing_tasks = 0
max_concurrent_tasks = 1  # Serial processing by default

def clean_text(text):
    """Clean text by removing extra spaces and normalizing punctuation"""
    if not text:
        return text
    # Remove extra spaces around punctuation
    text = re.sub(r'\s+([,.!?;:，。！？；：])', r'\1', text)
    # Remove extra spaces between Chinese characters and punctuation
    text = re.sub(r'([^\s])\s+([,.!?;:，。！？；：])', r'\1\2', text)
    # Remove multiple consecutive spaces
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def split_audio_file(file_path: str, max_size_mb: int = 100) -> List[str]:
    """Split large audio file into smaller chunks"""
    try:
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb <= max_size_mb:
            return [file_path]  # No splitting needed
        
        print(f"Splitting audio file {os.path.basename(file_path)} ({file_size_mb:.2f}MB) into chunks of {max_size_mb}MB")
        
        # Load audio file
        audio = AudioSegment.from_file(file_path)
        duration_ms = len(audio)
        
        # Calculate chunk duration based on file size ratio
        chunk_duration_ms = int(duration_ms * (max_size_mb / file_size_mb))
        
        # Ensure minimum chunk size to avoid too many small chunks
        min_chunk_duration_ms = 30000  # 30 seconds minimum
        chunk_duration_ms = max(chunk_duration_ms, min_chunk_duration_ms)
        
        # Split audio into chunks
        chunk_files = []
        start_ms = 0
        chunk_index = 0
        
        while start_ms < duration_ms:
            end_ms = min(start_ms + chunk_duration_ms, duration_ms)
            chunk = audio[start_ms:end_ms]
            
            # Save chunk to temporary file
            chunk_file_path = f"{file_path}_chunk_{chunk_index}.wav"
            chunk.export(chunk_file_path, format="wav")
            chunk_files.append(chunk_file_path)
            
            print(f"Created chunk {chunk_index}: {start_ms}ms - {end_ms}ms")
            start_ms = end_ms
            chunk_index += 1
            
            # Break if we've created too many chunks (safety check)
            if chunk_index > 100:  # Max 100 chunks
                print("Warning: Too many chunks created, stopping split")
                break
        
        print(f"Split audio into {len(chunk_files)} chunks")
        return chunk_files
        
    except Exception as e:
        print(f"Error splitting audio file: {e}")
        # Return original file if splitting fails
        return [file_path]


def merge_srt_results(srt_results: List[str], chunk_durations: List[float]) -> str:
    """Merge multiple SRT results with adjusted timestamps"""
    if not srt_results:
        return ""
    
    if len(srt_results) == 1:
        return srt_results[0]
    
    merged_lines = []
    global_segment_index = 1
    time_offset = 0.0
    
    for i, srt_content in enumerate(srt_results):
        if not srt_content.strip():
            continue
            
        lines = srt_content.strip().split('\n')
        line_index = 0
        
        while line_index < len(lines):
            # Skip empty lines
            if not lines[line_index].strip():
                line_index += 1
                continue
                
            # Get segment number (we'll renumber)
            if line_index < len(lines) and lines[line_index].strip().isdigit():
                line_index += 1  # Skip original segment number
            
            # Get timestamp line
            if line_index < len(lines):
                timestamp_line = lines[line_index]
                line_index += 1
                
                # Parse timestamps and adjust them
                try:
                    # Format: HH:MM:SS,mmm --> HH:MM:SS,mmm
                    parts = timestamp_line.split(' --> ')
                    if len(parts) == 2:
                        start_time = parts[0]
                        end_time = parts[1]
                        
                        # Adjust timestamps with offset
                        adjusted_start = adjust_srt_timestamp(start_time, time_offset)
                        adjusted_end = adjust_srt_timestamp(end_time, time_offset)
                        adjusted_timestamp = f"{adjusted_start} --> {adjusted_end}"
                        
                        # Add segment number
                        merged_lines.append(str(global_segment_index))
                        global_segment_index += 1
                        
                        # Add adjusted timestamp
                        merged_lines.append(adjusted_timestamp)
                        
                        # Add text lines until next segment or end
                        text_lines = []
                        while line_index < len(lines) and lines[line_index].strip() and not lines[line_index].strip().isdigit():
                            text_lines.append(lines[line_index].strip())
                            line_index += 1
                        
                        # Add text content
                        merged_lines.extend(text_lines)
                        merged_lines.append("")  # Empty line after segment
                        
                except Exception as e:
                    print(f"Error processing timestamp line: {e}")
                    line_index += 1
                    continue
        
        # Update time offset for next chunk
        if i < len(chunk_durations):
            time_offset += chunk_durations[i]
    
    # Clean the merged content to ensure no BOM or invalid characters
    merged_content = '\n'.join(merged_lines)

    # Remove BOM if present
    if merged_content.startswith('\ufeff'):
        merged_content = merged_content[1:]

    # Strip any extra whitespace
    merged_content = merged_content.strip()

    return merged_content


def adjust_srt_timestamp(timestamp: str, offset_seconds: float) -> str:
    """Adjust SRT timestamp by offset in seconds"""
    try:
        # Parse timestamp format: HH:MM:SS,mmm
        time_parts = timestamp.replace(',', ':').split(':')
        if len(time_parts) != 4:
            return timestamp
            
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        seconds = int(time_parts[2])
        milliseconds = int(time_parts[3])
        
        # Convert to total seconds
        total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        
        # Add offset
        total_seconds += offset_seconds
        
        # Convert back to timestamp format
        new_hours = int(total_seconds // 3600)
        total_seconds %= 3600
        new_minutes = int(total_seconds // 60)
        total_seconds %= 60
        new_seconds = int(total_seconds)
        new_milliseconds = int((total_seconds - new_seconds) * 1000)
        
        return f"{new_hours:02d}:{new_minutes:02d}:{new_seconds:02d},{new_milliseconds:03d}"
        
    except Exception as e:
        print(f"Error adjusting timestamp {timestamp}: {e}")
        return timestamp

def format_timestamp_srt(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def transcribe_to_srt(file_path: str, language: str = "auto", max_words_per_segment: int = 15, timeout: int = 1800):
    """Transcribe audio file to SRT format with enhanced timeout protection and progress monitoring"""
    global model

    # 根据文件大小动态调整超时时间
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    dynamic_timeout = max(timeout, int(file_size_mb * 30))  # 每MB额外30秒

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Transcription timed out after {dynamic_timeout} seconds (file: {file_size_mb:.1f}MB)")

    # Set up signal handler for timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(dynamic_timeout)

    print(f"🎵 开始转录: {os.path.basename(file_path)} ({file_size_mb:.1f}MB), 超时时间: {dynamic_timeout}秒")
    
    try:
        # Get file information for debugging
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        # Start timing for performance metrics
        total_start_time = time.time()
        
        try:
            # Timing for language detection
            lang_detect_time = 0
            # Detect language if auto
            if language == "auto":
                lang_detect_start = time.time()
                # First do a quick language detection
                temp_model = WhisperModel("tiny", device=device if model else "cpu", compute_type="int8")
                temp_segments, temp_info = temp_model.transcribe(file_path, beam_size=1)
                language = temp_info.language
                lang_detect_time = time.time() - lang_detect_start
                print(f"Debug Info - Language Detection Time: {lang_detect_time:.2f}s")
            
            # Timing for main transcription with VAD
            transcription_start_time = time.time()
            print(f"Debug Info - Starting transcription with VAD filter")
            # Transcribe with word timestamps for better control
            segments, info = model.transcribe(
                file_path,
                beam_size=5,
                word_timestamps=True,
                language=language if language != "auto" else None,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Convert segments to list to ensure all transcription is complete
            segments_list = list(segments)
            print(f"Debug Info - Converted {len(segments_list)} segments to list")
            
            transcription_time = time.time() - transcription_start_time
            print(f"Debug Info - Transcription completed")
            
            # Calculate performance metrics after all transcription work is done
            total_time = time.time() - total_start_time
            audio_duration = info.duration if info.duration else 0
            speed_ratio = audio_duration / total_time if total_time > 0 else 0
            
            # Timing for SRT generation
            srt_generation_start = time.time()
            
            # Generate SRT content
            srt_lines = []
            segment_index = 1
            total_words = 0
            
            # Debug timing for processing
            processing_start = time.time()
            long_segments_count = 0
            short_segments_count = 0
            total_chunks = 0
            
            print(f"Debug Info - Starting SRT generation for {len(segments_list)} segments")
            
            # Process all segments
            for segment_idx, segment in enumerate(segments_list):
                segment_start = time.time()
                
                if segment.words:
                    total_words += len(segment.words)
                
                # Only process non-empty segments
                if segment.text.strip():
                    # Check if we need to split this segment
                    if segment.words and len(segment.words) > max_words_per_segment:
                        long_segments_count += 1
                        # Split into smaller chunks
                        words = segment.words
                        chunk_processing_start = time.time()
                        for i in range(0, len(words), max_words_per_segment):
                            chunk_words = words[i:i + max_words_per_segment]
                            if chunk_words:
                                total_chunks += 1
                                start_time = chunk_words[0].start
                                end_time = chunk_words[-1].end
                                # Extract words and clean up spacing
                                word_extract_start = time.time()
                                word_texts = [word.word for word in chunk_words]
                                word_extract_time = time.time() - word_extract_start
                                
                                text_join_start = time.time()
                                text = "".join(word_texts)  # For Chinese, no spaces needed
                                text_join_time = time.time() - text_join_start
                                
                                clean_start = time.time()
                                # Only clean text if it's not empty
                                if text.strip():
                                    text = clean_text(text)
                                clean_time = time.time() - clean_start
                                
                                if text.strip():  # Only add non-empty segments
                                    srt_lines.append(f"{segment_index}")
                                    srt_lines.append(f"{format_timestamp_srt(start_time)} --> {format_timestamp_srt(end_time)}")
                                    srt_lines.append(f"{text}")
                                    srt_lines.append("")  # Empty line
                                    segment_index += 1
                                
                                # Log chunk processing time periodically
                                if total_chunks % 20 == 0:
                                    chunk_total_time = time.time() - chunk_processing_start
                                    print(f"Debug Info - Processed {total_chunks} chunks, "
                                          f"Word extract: {word_extract_time:.6f}s, "
                                          f"Text join: {text_join_time:.6f}s, "
                                          f"Clean: {clean_time:.6f}s, "
                                          f"Total chunk time: {chunk_total_time:.6f}s")
                                    chunk_processing_start = time.time()
                    else:
                        short_segments_count += 1
                        # No splitting needed, process as is
                        clean_start = time.time()
                        cleaned_text = clean_text(segment.text)
                        clean_time = time.time() - clean_start
                        
                        if cleaned_text.strip():  # Only add non-empty segments
                            srt_lines.append(f"{segment_index}")
                            srt_lines.append(f"{format_timestamp_srt(segment.start)} --> {format_timestamp_srt(segment.end)}")
                            srt_lines.append(f"{cleaned_text}")
                            srt_lines.append("")  # Empty line
                            segment_index += 1
                    
                    # Log segment processing time periodically
                    if (long_segments_count + short_segments_count) % 10 == 0:
                        segment_time = time.time() - segment_start
                        print(f"Debug Info - Processed {long_segments_count + short_segments_count} segments, "
                              f"Last segment time: {segment_time:.6f}s")
                
                # Log overall progress periodically
                if segment_idx > 0 and segment_idx % 20 == 0:
                    elapsed = time.time() - processing_start
                    print(f"Debug Info - Processed {segment_idx}/{len(segments_list)} segments in {elapsed:.2f}s")
            
            processing_time = time.time() - processing_start
            print(f"Debug Info - SRT processing completed: {long_segments_count} long segments, "
                  f"{short_segments_count} short segments, {total_chunks} chunks, "
                  f"Processing time: {processing_time:.2f}s")
            
            # Join all lines at once for better performance
            join_start = time.time()
            srt_content = "\n".join(srt_lines).strip()
            join_time = time.time() - join_start
            print(f"Debug Info - SRT content joined in {join_time:.6f}s")
            
            # Calculate total SRT generation time
            srt_generation_time = time.time() - srt_generation_start
            print(f"Debug Info - SRT generation time: {srt_generation_time:.2f}s (Processing: {processing_time:.2f}s, Join: {join_time:.6f}s)")
            
            # Print debug information
            print(f"Debug Info - File: {file_name}, Size: {file_size} bytes")
            print(f"Debug Info - Language: {info.language}, Duration: {audio_duration:.2f}s")
            print(f"Debug Info - Total Time: {total_time:.2f}s, Speed Ratio: {speed_ratio:.2f}x")
            if lang_detect_time > 0:
                print(f"Debug Info - Language Detection: {lang_detect_time:.2f}s")
            print(f"Debug Info - Transcription (VAD + Whisper): {transcription_time:.2f}s")
            print(f"Debug Info - SRT Generation: {srt_generation_time:.2f}s")
            print(f"Debug Info - Segments: {segment_index-1}, Words: {total_words}")
            
            # Clean SRT content to remove BOM and invalid characters
            final_srt = srt_content.strip()
            if final_srt.startswith('\ufeff'):
                print("Debug Info - Found BOM in SRT content from transcribe_to_srt, removing...")
                final_srt = final_srt[1:]

            # Debug return timing
            return_start = time.time()
            result = final_srt
            return_time = time.time() - return_start
            print(f"Debug Info - Function return preparation time: {return_time:.6f}s")

            print(f"DEBUG: Final SRT content preview: {repr(result[:100])}")

            return result
            
        except TimeoutError as e:
            raise HTTPException(status_code=504, detail=f"Transcription timeout: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
        finally:
            # Clean up signal handler
            signal.alarm(0)  # Cancel the alarm
            signal.signal(signal.SIGALRM, old_handler)  # Restore original handler
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model, stats
    print(f"Initializing {model_size} model on device {device}...")
    import time
    model_init_start = time.time()
    model = WhisperModel(model_size, device=device, compute_type="int8")
    model_init_time = time.time() - model_init_start
    print(f"Model initialized successfully in {model_init_time:.2f}s!")

    # 初始化统计信息
    instance_info = get_instance_info()
    stats.instance_id = instance_info["instance_id"]
    stats.gpu_device = instance_info["gpu_device"]
    stats.port = instance_info["port"]
    print(f"Instance {stats.instance_id} started on GPU {stats.gpu_device}, port {stats.port}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/stats")
async def get_instance_stats():
    """获取实例详细统计数据"""
    return {
        "status": "healthy",
        "stats": get_stats()
    }

@app.get("/test-json")
async def test_json_response():
    """Test endpoint to verify JSON response format"""
    test_data = "1\n00:00:00,000 --> 00:00:02,000\n测试字幕内容\n\n2\n00:00:02,000 --> 00:00:04,000\n第二条字幕"
    return {
        "code": 0,
        "msg": "ok",
        "data": test_data
    }

@app.post("/inference")
async def inference(
    request: Request,
    file: UploadFile = File(...),
    response_format: str = Form("srt"),
    language: str = Form("auto")
):
    """ASR inference endpoint compatible with existing clients - 优化流式上传"""

    # 记录请求开始
    request_start_time = time.time()
    update_stats("request_start")

    # Require API key authentication
    require_auth(request)

    # Validate response format
    if response_format != "srt":
        raise HTTPException(status_code=400, detail="Only SRT format is supported")

    # Check queue size
    if len(task_queue) >= MAX_QUEUE_SIZE:
        raise HTTPException(status_code=503, detail=f"Service busy, queue is full (max {MAX_QUEUE_SIZE} tasks)")

    # Create temporary file
    temp_file_path = None
    try:
        # Save uploaded file to temporary location using streaming
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] if file.filename else ".tmp") as temp_file:
            temp_file_path = temp_file.name

            # 流式写入文件，避免内存溢出
            chunk_size = 1024 * 1024  # 1MB chunks
            total_size = 0
            upload_start = time.time()

            print(f"开始流式接收文件: {file.filename}")

            # 使用正确的FastAPI上传文件读取方法
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                temp_file.write(chunk)
                total_size += len(chunk)

                # 每10MB输出一次进度
                if total_size % (10 * 1024 * 1024) == 0:
                    print(f"已接收: {total_size / (1024*1024):.1f}MB")

            upload_time = time.time() - upload_start
            file_size_mb = total_size / (1024 * 1024)
            print(f"✅ 文件接收完成！大小: {file_size_mb:.2f}MB, 耗时: {upload_time:.2f}秒, 速度: {total_size/(upload_time*1024*1024):.2f}MB/s")

            # 更新上传统计
            update_stats("upload_complete", upload_time=upload_time, file_size_mb=file_size_mb)

        # Add task to queue
        task_id = str(uuid.uuid4())
        future = asyncio.Future()
        task_queue.append((task_id, temp_file_path, language, future, request_start_time, file_size_mb))
        print(f"Task {task_id} added to queue. Queue size: {len(task_queue)}")

        # Process queue if not already processing
        asyncio.create_task(process_queue())

        # Wait for task completion
        result = await future
        return result

    except Exception as e:
        error_msg = f"Processing error: {str(e)}"
        print(f"❌ 文件处理失败: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

    finally:
        # Clean up temporary file
        cleanup_start = time.time()
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        cleanup_time = time.time() - cleanup_start
        print(f"Debug Info - Cleanup time: {cleanup_time:.6f}s")

async def process_queue():
    """Process tasks in the queue serially"""
    global current_processing_tasks

    # Check if we're already processing the maximum number of tasks
    if current_processing_tasks >= max_concurrent_tasks:
        return

    # Acquire lock to ensure thread safety
    async with processing_lock:
        # Check again after acquiring lock
        if current_processing_tasks >= max_concurrent_tasks or len(task_queue) == 0:
            return

        # Get the next task from the queue
        task_id, temp_file_path, language, future, request_start_time, file_size_mb = task_queue.popleft()
        current_processing_tasks += 1
        print(f"Starting processing of task {task_id}. Remaining queue size: {len(task_queue)}")
    
    try:
        # Check if file needs to split (file_size_mb is now passed from queue item)
        if file_size_mb > MAX_FILE_SIZE:
            # Split large file into chunks
            print(f"File {os.path.basename(temp_file_path)} is {file_size_mb:.2f}MB, splitting into chunks...")
            chunk_files = split_audio_file(temp_file_path, MAX_FILE_SIZE)
            
            if len(chunk_files) > 1:
                # Process each chunk separately
                srt_results = []
                chunk_durations = []
                
                for i, chunk_file in enumerate(chunk_files):
                    print(f"Processing chunk {i+1}/{len(chunk_files)}: {os.path.basename(chunk_file)}")
                    chunk_start = time.time()
                    
                    # Get chunk duration for timestamp adjustment
                    try:
                        chunk_audio = AudioSegment.from_file(chunk_file)
                        chunk_duration = len(chunk_audio) / 1000.0  # Convert ms to seconds
                        chunk_durations.append(chunk_duration)
                    except:
                        chunk_durations.append(0.0)
                    
                    # Process chunk with individual chunk timeout
                    chunk_srt = transcribe_to_srt(chunk_file, language, timeout=CHUNK_TIMEOUT)
                    srt_results.append(chunk_srt)

                    # 更新 chunk 统计
                    update_stats("chunk_processed")

                    chunk_time = time.time() - chunk_start
                    print(f"Chunk {i+1} processed in {chunk_time:.2f}s")
                
                # Merge SRT results with adjusted timestamps using proper SRTMerger
                print("Merging SRT results from chunks...")
                from srt_merger import SRTMerger

                # Create timing information for chunks
                chunk_timings = []
                current_time = 0.0
                for duration in chunk_durations:
                    chunk_timings.append((current_time, current_time + duration))
                    current_time += duration

                merger = SRTMerger()
                final_srt = merger.merge_chunk_results(srt_results, chunk_timings)
                srt_content = final_srt
                
                # Clean up chunk files
                for chunk_file in chunk_files:
                    if chunk_file != temp_file_path and os.path.exists(chunk_file):
                        os.unlink(chunk_file)
            else:
                # File doesn't need splitting or splitting failed
                srt_content = transcribe_to_srt(temp_file_path, language, timeout=CHUNK_TIMEOUT)
                # 对于不分块的文件，也算作1个chunk
                update_stats("chunk_processed")
        else:
            # File is small enough, process normally
            srt_content = transcribe_to_srt(temp_file_path, language, timeout=CHUNK_TIMEOUT)
            # 对于小文件，也算作1个chunk
            update_stats("chunk_processed")

        # Process the transcription
        srt_start = time.time()
        print(f"Debug Info - Starting transcription process for {os.path.basename(temp_file_path)}")
        transcribe_start = time.time()
        # transcribe_time is already calculated in the chunk processing above
        srt_total_time = time.time() - srt_start
        transcribe_time = srt_total_time  # Use total time as transcribe time for logging
        print(f"Debug Info - API SRT total time: {srt_total_time:.2f}s")
        print(f"Debug Info - Transcribe function call time: {transcribe_time:.2f}s")
        
        # Check SRT content size
        srt_length = len(srt_content)
        print(f"Debug Info - SRT content length: {srt_length} characters")
        
        # Create response with proper headers for large files
        response_start = time.time()

        # Clean SRT content to ensure valid JSON - debug and handle BOM/Unicode issues
        print(f"DEBUG: Raw SRT content type: {type(srt_content)}")
        print(f"DEBUG: SRT content byte representation: {repr(srt_content[:100].encode('utf-8') if isinstance(srt_content, str) else srt_content[:100])}")

        # Check for BOM (Byte Order Mark) which can cause JSON parsing errors
        if isinstance(srt_content, str):
            cleaned_srt = srt_content
            bom_utf8 = '\ufeff'
            bom_utf8_bytes = b'\xef\xbb\xbf'
        else:
            # Handle bytes case
            cleaned_srt = srt_content.decode('utf-8', errors='ignore') if isinstance(srt_content, bytes) else str(srt_content)

        # Remove UTF-8 BOM if present (after conversion to string)
        if isinstance(cleaned_srt, str) and cleaned_srt.startswith(bom_utf8):
            print("DEBUG: Found UTF-8 BOM in SRT content, removing...")
            cleaned_srt = cleaned_srt[1:]

        # Strip whitespace and newlines
        cleaned_srt = cleaned_srt.strip()
        if cleaned_srt.startswith('\n'):
            cleaned_srt = cleaned_srt[1:]  # Remove leading newline
        if cleaned_srt.startswith('\r\n'):
            cleaned_srt = cleaned_srt[2:]  # Remove leading CRLF

        print(f"DEBUG: Cleaned SRT content preview: {repr(cleaned_srt[:100])}")

        # Final JSON validation check
        import json
        json_response = json.dumps({
            "code": 0,
            "msg": "ok",
            "data": cleaned_srt
        }, ensure_ascii=False)

        # Check for any invisible characters at the start
        json_repr = repr(json_response[:100])
        print(f"DEBUG: Final JSON preview (repr): {json_repr}")

        # Check if JSON starts correctly
        if not json_response.startswith('{'):
            print(f"WARN: JSON doesn't start with '{{', actual start: {repr(json_response[:50])}")

        # Create proper JSON response with explicit content type
        # Create JSON response manually to ensure proper formatting
        import json

        json_data = {
            "code": 0,
            "msg": "ok",
            "data": cleaned_srt
        }

        json_string = json.dumps(json_data, ensure_ascii=False)
        print(f"DEBUG: Final JSON string preview: {json_string[:100]}...")

        response = JSONResponse(
            content=json_data,
            headers={
                "Content-Type": "application/json; charset=utf-8"
            },
            status_code=200
        )

        # CRITICAL: Force Content-Type override using multiple methods
        response.headers.update({"Content-Type": "application/json; charset=utf-8"})
        response.media_type = "application/json"

        # Add debug headers for client inspection
        response.headers.update({
            "X-Debug-Response-Type": "json",
            "X-Debug-Content-Length": str(len(json_string)),
            "X-Debug-Process": "faster_whisper_api"
        })

        # FINAL SAFEGUARD: Force response class recreation to ensure proper JSON format
        response = JSONResponse(
            content=json_data,
            status_code=200,
            media_type="application/json",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "X-Debug-Response-Type": "json",
                "X-Debug-Content-Length": str(len(json_string)),
                "X-Debug-Process": "faster_whisper_api"
            }
        )

        print("=== CRITICAL DEBUG ===")
        print(f"DEBUG: Final JSON: {json_string[:200]}...")
        print(f"DEBUG: Response Content-Type: {response.headers.get('content-type')}")
        print(f"DEBUG: Response media_type: {response.media_type}")
        print(f"DEBUG: All response headers: {dict(response.headers)}")

        # Final verification - check response body format
        response_body = response.body
        print(f"DEBUG: Response body starts with: {response_body[:20]}")
        print(f"DEBUG: Response body type: {type(response_body)}")
        print("=== END CRITICAL DEBUG ===")
        response_time = time.time() - response_start
        print(f"Debug Info - JSON response creation time: {response_time:.6f}s")

        # Set the result and update success statistics
        processing_time = time.time() - request_start_time
        update_stats("request_success", processing_time=processing_time, file_size_mb=file_size_mb)
        future.set_result(response)

    except Exception as e:
        print(f"ERROR: Exception in inference: {str(e)}")
        import traceback
        print(f"ERROR: Traceback: {traceback.format_exc()}")

        error_response = JSONResponse(
            content={
                "code": 500,
                "msg": f"Processing error: {str(e)}",
                "data": ""
            },
            headers={"Content-Type": "application/json; charset=utf-8"}
        )

        # 更新失败统计
        update_stats("request_failed")
        future.set_result(error_response)
    
    finally:
        # Decrement the processing counter
        async with processing_lock:
            current_processing_tasks -= 1
            print(f"Finished processing task {task_id}. Current processing tasks: {current_processing_tasks}")
        
        # Process next task if available
        if len(task_queue) > 0:
            asyncio.create_task(process_queue())

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", "5001"))
    uvicorn.run(app, host="0.0.0.0", port=port)