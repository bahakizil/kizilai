# SesAI Platform - GPU-enabled Docker Image
# Base: NVIDIA CUDA 12.8 runtime for RTX 5080 Blackwell support

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Create app directory
WORKDIR /app

# Install Chatterbox TTS first (this will install torch 2.6)
RUN pip install --no-cache-dir chatterbox-tts soundfile scipy

# Upgrade to PyTorch NIGHTLY with CUDA 12.8 for RTX 5080 Blackwell (sm_120)
RUN pip install --no-cache-dir --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall

# Fix NumPy version conflict (Numba needs <= 2.3, PyTorch nightly installed 2.4)
RUN pip install --no-cache-dir "numpy<2.4,>=1.24"

# Install Python dependencies
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# Copy application code
COPY src/ ./src/

# Create non-root user for security
RUN useradd -m -u 1000 sesai \
    && chown -R sesai:sesai /app

USER sesai

# Expose ports
# 8080: HTTP API (future)
# 8765: AudioSocket
EXPOSE 8080 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import asyncio; from src.services.llm.service import LLMService; asyncio.run(LLMService().health_check())" || exit 1

# Run the application
CMD ["python", "-m", "src.main"]
