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
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from faster_whisper import WhisperModel
import re
import asyncio
from collections import deque

# Audio processing libraries
import pydub
from pydub import AudioSegment
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="faster-whisper ASR Service", version="1.0.0")

# Global model instance
model = None
model_size = "large-v3-turbo"  # Default model size

# Task queue configuration
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "10"))  # Default queue size of 10
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "100"))  # Default max file size in MB
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
    
    return '\n'.join(merged_lines)


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

def transcribe_to_srt(file_path: str, language: str = "auto", max_words_per_segment: int = 15):
    """Transcribe audio file to SRT format"""
    global model
    
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
            temp_model = WhisperModel("tiny", device="cuda" if model.model.device == "cuda" else "cpu", compute_type="int8")
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
        
        # Debug return timing
        return_start = time.time()
        result = srt_content.strip()
        return_time = time.time() - return_start
        print(f"Debug Info - Function return preparation time: {return_time:.6f}s")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global model
    print(f"Initializing {model_size} model...")
    import time
    model_init_start = time.time()
    model = WhisperModel(model_size, device="cuda", compute_type="int8")
    model_init_time = time.time() - model_init_start
    print(f"Model initialized successfully in {model_init_time:.2f}s!")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/inference")
async def inference(
    file: UploadFile = File(...),
    response_format: str = Form("srt"),
    language: str = Form("auto")
):
    """ASR inference endpoint compatible with existing clients"""
    
    # Validate response format
    if response_format != "srt":
        raise HTTPException(status_code=400, detail="Only SRT format is supported")
    
    # Check queue size
    if len(task_queue) >= MAX_QUEUE_SIZE:
        raise HTTPException(status_code=503, detail=f"Service busy, queue is full (max {MAX_QUEUE_SIZE} tasks)")
    
    # Create temporary file
    temp_file_path = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] if file.filename else ".tmp") as temp_file:
            temp_file_path = temp_file.name
            contents = await file.read()
            temp_file.write(contents)
        
        # Add task to queue
        task_id = str(uuid.uuid4())
        future = asyncio.Future()
        task_queue.append((task_id, temp_file_path, language, future))
        print(f"Task {task_id} added to queue. Queue size: {len(task_queue)}")
        
        # Process queue if not already processing
        asyncio.create_task(process_queue())
        
        # Wait for task completion
        result = await future
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
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
        task_id, temp_file_path, language, future = task_queue.popleft()
        current_processing_tasks += 1
        print(f"Starting processing of task {task_id}. Remaining queue size: {len(task_queue)}")
    
    try:
        # Check if file needs to be split
        file_size_mb = os.path.getsize(temp_file_path) / (1024 * 1024)
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
                    
                    # Process chunk
                    chunk_srt = transcribe_to_srt(chunk_file, language)
                    srt_results.append(chunk_srt)
                    
                    chunk_time = time.time() - chunk_start
                    print(f"Chunk {i+1} processed in {chunk_time:.2f}s")
                
                # Merge SRT results with adjusted timestamps
                print("Merging SRT results from chunks...")
                final_srt = merge_srt_results(srt_results, chunk_durations)
                srt_content = final_srt
                
                # Clean up chunk files
                for chunk_file in chunk_files:
                    if chunk_file != temp_file_path and os.path.exists(chunk_file):
                        os.unlink(chunk_file)
            else:
                # File doesn't need splitting or splitting failed
                srt_content = transcribe_to_srt(temp_file_path, language)
        else:
            # File is small enough, process normally
            srt_content = transcribe_to_srt(temp_file_path, language)
        
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
        
        # Create response
        response_start = time.time()
        response = JSONResponse(content={
            "code": 0,
            "msg": "ok",
            "data": srt_content
        })
        response_time = time.time() - response_start
        print(f"Debug Info - JSON response creation time: {response_time:.6f}s")
        
        # Set the result
        future.set_result(response)
        
    except Exception as e:
        error_response = JSONResponse(content={
            "code": 500,
            "msg": f"Processing error: {str(e)}",
            "data": ""
        })
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
    uvicorn.run(app, host="0.0.0.0", port=5001)