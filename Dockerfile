# Use a more compatible base image for ARM64
FROM python:3.9-slim

# Set environment variables for deterministic training
ENV PYTHONHASHSEED=0
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""
ENV PYTORCH_ENABLE_MPS_FALLBACK=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch and other dependencies
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Create output directories
RUN mkdir -p outputs/models

# Set default command
CMD ["python", "-u", "train_binary_classifier.py"]
