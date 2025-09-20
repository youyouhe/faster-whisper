#!/bin/bash
echo "🧪 Testing Docker image optimizations..."
echo ""

# Clean up old test images only
echo "🧹 Cleaning old test images..."
docker rmi -f test-optimized-api test-optimized-worker 2>/dev/null || true
echo ""

# First build with current Dockerfile for comparison
echo "📊 Building original API server..."
docker build -f docker/Dockerfile.api -t original-api-server . 2>/dev/null
echo ""

# Test optimized Dockerfile
echo "📊 Building optimized versions..."
echo "   Building optimized API server..."
time docker build -f docker/Dockerfile.optimized \
  --target=api-server \
  -t test-optimized-api \
  . 2>/dev/null

echo "   Building optimized ASR worker..."
time docker build -f docker/Dockerfile.optimized \
  --target=asr-worker \
  -t test-optimized-worker \
  . 2>/dev/null

echo ""
echo "📏 Image sizes comparison:"
echo ""

# Function to format size
format_size() {
    size="$1"
    if [ "$size" -gt 1073741824 ]; then
        echo "$(( size / 1073741824 ))GB"
    elif [ "$size" -gt 1048576 ]; then
        echo "$(( size / 1048576 ))MB"
    else
        echo "$(( size / 1024 ))KB"
    fi
}

# Get sizes
original_size=$(docker inspect original-api-server --format='{{.Size}}' 2>/dev/null || echo "0")
optimized_api_size=$(docker inspect test-optimized-api --format='{{.Size}}' 2>/dev/null || echo "0")
optimized_worker_size=$(docker inspect test-optimized-worker --format='{{.Size}}' 2>/dev/null || echo "0")

if [ "$original_size" -gt 0 ]; then
    echo "Original API Server:   $(format_size "$original_size")"
fi
if [ "$optimized_api_size" -gt 0 ]; then
    echo "Optimized API Server:  $(format_size "$optimized_api_size")"
fi
if [ "$optimized_worker_size" -gt 0 ]; then
    echo "Optimized ASR Worker:  $(format_size "$optimized_worker_size")"
fi

# Calculate savings
if [ "$original_size" -gt 0 ] && [ "$optimized_api_size" -gt 0 ] && [ "$optimized_api_size" -lt "$original_size" ]; then
    savings=$((original_size - optimized_api_size))
    percent=$(( (savings * 100) / original_size ))

    echo ""
    echo "💾 API Server space savings: $percent%"
    echo "   Absolute reduction: $(format_size "$savings")"
fi

echo ""
echo "✅ Build completed!"
echo ""
echo "🧪 Test optimized API server:"
echo "   docker run --rm test-optimized-api python --version"
echo ""
echo "🧪 Test optimized ASR worker:"
echo "   docker run --rm --gpus all test-optimized-worker python --version"