FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    ffmpeg \
    git && \
    rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Download default model (tiny for testing, can be changed)
RUN python3 -c "from faster_whisper import WhisperModel; model = WhisperModel('tiny', device='cpu'); print('Model downloaded successfully')"

# Expose ports for all services
EXPOSE 5001 5002 5003 5004

# Start all services
CMD ["./start_all_services.sh"]