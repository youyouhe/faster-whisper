#!/usr/bin/env python3
"""
Simple test for the multipart parsing function
"""

import re

def parse_multipart_data_simple(data: bytes, boundary: str) -> bytes:
    """Simple version of multipart parsing for testing"""
    try:
        # Convert boundary to bytes with proper formatting
        boundary_bytes = f"--{boundary}".encode('utf-8')

        # Find the start of the audio data (after headers)
        # Look for the boundary, then skip headers until we find the empty line
        start_idx = 0
        while start_idx < len(data):
            # Find next boundary
            boundary_idx = data.find(boundary_bytes, start_idx)
            if boundary_idx == -1:
                print("Could not find starting boundary in multipart data")
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

        print("No audio file found in multipart data")
        raise ValueError("No audio file found in multipart data")

    except Exception as e:
        print(f"Error parsing multipart data: {e}")
        raise

def test_multipart_parsing():
    """Test the multipart parsing function"""
    print("Testing multipart parsing function...")

    # Create a simple multipart data example
    boundary = "test_boundary"
    test_data = (
        f"--{boundary}\r\n"
        "Content-Disposition: form-data; name=\"audio\"; filename=\"test.wav\"\r\n"
        "Content-Type: audio/wav\r\n"
        "\r\n"
        "fake_audio_data_content\r\n"
        f"--{boundary}--\r\n"
    ).encode('utf-8')

    print(f"Test data: {test_data}")

    try:
        result = parse_multipart_data_simple(test_data, boundary)
        print(f"Parsed result: {result}")
        print("Test passed!")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_multipart_parsing()