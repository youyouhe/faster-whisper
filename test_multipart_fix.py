#!/usr/bin/env python3
"""
Test script for the multipart data creation fix
"""

import time

def create_multipart_data(chunk_data: bytes, boundary: str, chunk_index: int) -> bytes:
    """
    Create properly formatted multipart data for worker
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

def test_multipart_creation():
    """Test the multipart data creation function"""
    print("Testing multipart data creation...")

    # Create fake audio data
    fake_audio = b"audiocontent123456"
    boundary = f"distributed_chunk_0_{int(time.time())}"

    multipart = create_multipart_data(fake_audio, boundary, 0)

    print(f"Boundary: {boundary}")
    print(f"Multipart data length: {len(multipart)} bytes")
    print("Multipart data (first 200 chars):")
    print(repr(multipart[:200]))
    print()

    # Verify structure
    multipart_str = multipart.decode('utf-8', errors='ignore')
    print("Multipart structure validation:")
    lines = multipart_str.split('\r\n')
    for i, line in enumerate(lines[:10]):
        print(f"  {i:2d}: {repr(line)}")

    # Check if it starts and ends correctly
    if multipart.startswith(f"--{boundary}".encode()):
        print("✓ Multipart starts with correct boundary")
    else:
        print("✗ Multipart does not start with correct boundary")

    if multipart.endswith(f"--{boundary}--\r\n".encode()):
        print("✓ Multipart ends with correct terminator")
    else:
        print("✗ Multipart does not end with correct terminator")

    print("Test completed!")

if __name__ == "__main__":
    test_multipart_creation()