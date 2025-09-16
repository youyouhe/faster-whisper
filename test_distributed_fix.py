#!/usr/bin/env python3
"""
Test script for distributed processing fix
"""

import asyncio
import aiohttp
from distributed_processor import DistributedProcessor

async def test_distributed_processor():
    """Test the distributed processor with sample data"""
    # Create a simple test
    print("Testing distributed processor fix...")

    # Initialize the processor
    processor = DistributedProcessor()

    # Test should_distribute method
    file_size = 50 * 1024 * 1024  # 50MB
    available_workers = 4

    should_distribute = await processor.should_distribute(file_size, available_workers)
    print(f"Should distribute 50MB file with {available_workers} workers: {should_distribute}")

    # Test with smaller file
    small_file_size = 5 * 1024 * 1024  # 5MB
    should_distribute_small = await processor.should_distribute(small_file_size, available_workers)
    print(f"Should distribute 5MB file with {available_workers} workers: {should_distribute_small}")

    print("Test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_distributed_processor())