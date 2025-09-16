#!/usr/bin/env python3
"""
Split WAV file into sub-files for testing distributed processing
"""

import os
import subprocess

def split_wav_file(input_file: str, num_splits: int = 4):
    """
    Split WAV file into multiple sub-files using ffmpeg

    Args:
        input_file: Path to input WAV file
        num_splits: Number of splits to create
    """
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        return

    # Get duration using ffprobe
    cmd_duration = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        input_file
    ]

    result = subprocess.run(cmd_duration, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error getting duration: {result.stderr}")
        return

    import json
    probe_data = json.loads(result.stdout)
    duration = float(probe_data['format']['duration'])

    print(f"Input file: {input_file}")
    print(f"Duration: {duration:.2f}s ({duration/60:.1f} minutes)")

    # Calculate chunk duration
    chunk_duration = duration / num_splits

    print(f"Splitting into {num_splits} chunks of ~{chunk_duration:.1f}s each")

    splits = []

    for i in range(num_splits):
        start_time = i * chunk_duration
        output_file = f"{input_file.rsplit('.', 1)[0]}_chunk_{i+1}.wav"

        cmd_split = [
            'ffmpeg',
            '-i', input_file,
            '-ss', str(start_time),
            '-t', str(chunk_duration),
            '-c', 'copy',  # Copy codec for WAV
            output_file
        ]

        print(f"Splitting chunk {i+1}: {start_time:.1f}s - {start_time + chunk_duration:.1f}s -> {output_file}")
        result = subprocess.run(cmd_split, capture_output=True, text=True)

        if result.returncode == 0:
            splits.append(output_file)
            print(f"✓ Created {output_file}")
        else:
            print(f"✗ Failed to create {output_file}: {result.stderr}")

    return splits

def verify_splits(splits):
    """Verify that the splits were created correctly"""
    print("\nVerifying splits:")
    total_size = 0
    for split_path in splits:
        if os.path.exists(split_path):
            file_size = os.path.getsize(split_path) / (1024 * 1024)  # MB
            print(f"✓ {split_path}: {file_size:.1f} MB")
            total_size += file_size
        else:
            print(f"✗ {split_path} not found")

    print(f"Total split size: {total_size:.1f} MB")

    # Get original file size
    original_file = '117.wav'
    if os.path.exists(original_file):
        original_size = os.path.getsize(original_file) / (1024 * 1024)
        print(f"Original file: {original_size:.1f} MB")
        print(f"Size difference: {((total_size - original_size) / original_size * 100):.1f}%")

if __name__ == "__main__":
    TARGET_FILE = '117.wav'
    splits = split_wav_file(TARGET_FILE, 4)
    if splits:
        verify_splits(splits)