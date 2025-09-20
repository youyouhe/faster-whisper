#!/usr/bin/env python3
"""
GPU detection script for Docker environments
"""

import os
import subprocess
import sys

def detect_gpus():
    """Detect the number of available GPUs using multiple methods"""

    # Method 1: Check CUDA_VISIBLE_DEVICES environment variable
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices:
        if cuda_visible_devices.lower() == 'all':
            # Need to actually count GPUs
            pass
        else:
            # Count comma-separated devices
            devices = cuda_visible_devices.split(',')
            return len(devices)

    # Method 2: Check NVIDIA_VISIBLE_DEVICES environment variable
    nvidia_visible_devices = os.environ.get('NVIDIA_VISIBLE_DEVICES')
    if nvidia_visible_devices:
        if nvidia_visible_devices.lower() == 'all':
            # Need to actually count GPUs
            pass
        else:
            # Count comma-separated devices
            devices = nvidia_visible_devices.split(',')
            return len(devices)

    # Method 3: Use nvidia-smi to count GPUs
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            # nvidia-smi returns count per line, so we count lines
            lines = result.stdout.strip().split('\n')
            if lines and lines[0].strip().isdigit():
                return int(lines[0].strip())
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        pass

    # Method 4: Use nvidia-smi -L to list GPUs
    try:
        result = subprocess.run(
            ['nvidia-smi', '-L'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            # Count lines in output (each GPU gets one line)
            lines = result.stdout.strip().split('\n')
            return len([line for line in lines if line.strip()])
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        pass

    # Method 5: Try to import torch and check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass

    # Method 6: Try to import tensorflow and check for GPUs
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            return len(gpus)
    except ImportError:
        pass

    # Default fallback
    return 1

if __name__ == "__main__":
    gpu_count = detect_gpus()
    print(gpu_count)
    sys.exit(0)