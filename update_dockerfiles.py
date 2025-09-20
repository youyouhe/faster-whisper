#!/usr/bin/env python3
"""
Update Dockerfiles to use working mirror: dockerproxy.com
"""

import os
import re

def update_dockerfile(dockerfile_path):
    """Update FROM instruction to use dockerproxy.com mirror"""

    # Read the dockerfile
    with open(dockerfile_path, 'r') as f:
        content = f.read()

    # Replace with dockerproxy mirror
    new_content = re.sub(
        r'FROM python:3\.10-alpine',
        'FROM dockerproxy.com/library/python:3.10-alpine',
        content
    )

    # Write back if changed
    if new_content != content:
        with open(dockerfile_path, 'w') as f:
            f.write(new_content)
        print(f"✅ Updated {dockerfile_path}")
    else:
        print(f"ℹ️ No changes needed for {dockerfile_path}")

if __name__ == '__main__':
    # Update all Dockerfiles that use python:3.10-slim (except ASR worker with CUDA)
    dockerfiles = [
        'docker/Dockerfile.api',
        'docker/Dockerfile.tus',
        'docker/Dockerfile.event',
        'docker/Dockerfile.callback'
    ]

    print("🔄 Updating Dockerfiles to use Docker mirror...")

    for df in dockerfiles:
        if os.path.exists(df):
            update_dockerfile(df)
        else:
            print(f"⚠️ Dockerfile not found: {df}")

    print("\n🎯 All Dockerfiles updated!")
    print("\nReady to build with:")
    print("  docker-compose build")
    print("  docker-compose up -d")