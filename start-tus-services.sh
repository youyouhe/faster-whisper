#!/bin/bash
# Startup script for Tus.io ASR system

set -e

echo "🚀 Starting Tus.io ASR System..."

# Create necessary directories
mkdir -p data logs docker

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build and start services
echo "🏗️  Building and starting services..."
docker-compose up --build -d

# Wait for services to be healthy
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."

# Check Redis
if docker-compose exec -T redis redis-cli ping >/dev/null 2>&1; then
    echo "✅ Redis is healthy"
else
    echo "❌ Redis is not responding"
fi

# Check API server
if curl -f http://localhost:8000/health >/dev/null 2>&1; then
    echo "✅ Tus API Server is healthy"
else
    echo "❌ Tus API Server is not responding"
fi

# Check Tus server
if curl -f http://localhost:1080/health >/dev/null 2>&1; then
    echo "✅ Tus Server is healthy"
else
    echo "❌ Tus Server is not responding"
fi

# Check ASR worker
if curl -f http://localhost:8081/health >/dev/null 2>&1; then
    echo "✅ ASR Worker is healthy"
else
    echo "❌ ASR Worker is not responding"
fi

echo ""
echo "🎉 Tus.io ASR System started successfully!"
echo ""
echo "📋 Service Endpoints:"
echo "  • Tus API Server: http://localhost:8000"
echo "  • Tus Server: http://localhost:1080"
echo "  • ASR Worker Health: http://localhost:8081"
echo "  • Legacy Load Balancer: http://localhost:5001"
echo ""
echo "📖 API Documentation:"
echo "  • API Docs: http://localhost:8000/docs"
echo "  • Health Check: http://localhost:8000/health"
echo ""
echo "🔧 Monitoring:"
echo "  docker-compose logs -f [service_name]"
echo "  docker-compose ps"
echo ""

# Show logs
echo "📜 Service logs (press Ctrl+C to stop):"
docker-compose logs -f --tail=10