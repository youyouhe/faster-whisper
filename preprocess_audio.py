#!/usr/bin/env python3
"""
Audio preprocessing script for faster-whisper
Converts audio to 16kHz mono format to optimize for Whisper model
"""

import os
import sys
import tempfile
import subprocess
import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)

def preprocess_audio(input_path: str, output_path: str = None) -> str:
    """
    Preprocess audio file to 16kHz mono format

    Args:
        input_path: Input audio file path
        output_path: Output audio file path (optional)

    Returns:
        Path to processed audio file
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.with_suffix('_16k.wav')

    output_path = Path(output_path)

    logger.info(f"Preprocessing audio: {input_path} -> {output_path}")

    try:
        # Use ffmpeg to convert to 16kHz mono
        cmd = [
            'ffmpeg',
            '-i', str(input_path),  # Input file
            '-ar', '16000',          # Sample rate
            '-ac', '1',              # Mono
            '-y',                    # Overwrite output
            str(output_path)       # Output file
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        logger.info(f"Audio preprocessing successful: {output_path}")
        logger.debug(f"FFmpeg output: {result.stdout}")

        return str(output_path)

    except subprocess.CalledProcessError as e:
        logger.error(f"Audio preprocessing failed: {e}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        raise RuntimeError(f"Failed to preprocess audio: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during preprocessing: {e}")
        raise

def main():
    """Main function for standalone usage"""
    parser = argparse.ArgumentParser(description='Preprocess audio for faster-whisper')
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('-o', '--output', help='Output audio file (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        output_path = preprocess_audio(args.input, args.output)
        print(f"✅ Preprocessed audio saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()