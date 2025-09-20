#!/bin/bash
# Test script for the dynamic GPU service setup

echo "🧪 Testing Dynamic GPU Service Setup"

# Test 1: Check if required files exist
echo "🔍 Checking for required files..."

REQUIRED_FILES=(
    "docker/Dockerfile.dynamic"
    "docker/start_dynamic_services.sh"
    "docker/detect_gpus.py"
    "docker-compose.dynamic.yml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✅ Found $file"
    else
        echo "❌ Missing $file"
        exit 1
    fi
done

# Test 2: Check if scripts are executable
echo "🔍 Checking script permissions..."

if [[ -x "docker/start_dynamic_services.sh" ]]; then
    echo "✅ start_dynamic_services.sh is executable"
else
    echo "❌ start_dynamic_services.sh is not executable"
    exit 1
fi

# Test 3: Validate Python GPU detection script
echo "🔍 Testing Python GPU detection script..."

if python3 -c "import sys; sys.path.append('docker'); from detect_gpus import detect_gpus; result = detect_gpus(); print(f'Detected GPUs: {result}'); assert isinstance(result, int) and result >= 0"; then
    echo "✅ Python GPU detection script works correctly"
else
    echo "❌ Python GPU detection script failed"
    exit 1
fi

# Test 4: Validate shell script GPU detection function
echo "🔍 Testing shell script GPU detection function..."

# Source the script to test the function
if source docker/start_dynamic_services.sh 2>/dev/null && declare -f detect_gpus >/dev/null; then
    echo "✅ Shell script GPU detection function exists"
else
    echo "❌ Shell script GPU detection function missing"
    exit 1
fi

# Test 5: Check Dockerfile contents
echo "🔍 Checking Dockerfile contents..."

if grep -q "nvidia/cuda" docker/Dockerfile.dynamic && grep -q "start_dynamic_services.sh" docker/Dockerfile.dynamic; then
    echo "✅ Dockerfile.dynamic contains required components"
else
    echo "❌ Dockerfile.dynamic missing required components"
    exit 1
fi

# Test 6: Check docker-compose configuration
echo "🔍 Checking docker-compose configuration..."

if grep -q "faster-whisper-dynamic" docker-compose.dynamic.yml && grep -q "NVIDIA_VISIBLE_DEVICES" docker-compose.dynamic.yml; then
    echo "✅ docker-compose.dynamic.yml contains required configuration"
else
    echo "❌ docker-compose.dynamic.yml missing required configuration"
    exit 1
fi

echo "🎉 All tests passed! The dynamic GPU service setup is ready to use."
echo ""
echo "To use the dynamic setup:"
echo "1. Build and start: docker-compose -f docker-compose.dynamic.yml up --build"
echo "2. The system will automatically detect GPUs and start services"
echo "3. Access the load balancer at http://localhost:5001"