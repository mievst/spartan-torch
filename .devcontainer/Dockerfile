# SPDX-License-Identifier: MIT
# spartan-torch Development Container
# CUDA 13.3

FROM nvidia/cuda:13.3.0-tensorrt-devel-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"



# Configure git safe directory
RUN git config --global --add safe.directory /workspace

# Set up Triton cache
ENV TRITON_CACHE_DIR=/workspace/.triton_cache
RUN mkdir -p /workspace/.triton_cache

# Set up uv cache
ENV UV_CACHE_DIR=/workspace/.uv_cache
RUN mkdir -p /workspace/.uv_cache

# Create workspace directory
WORKDIR /workspace

# Default command
CMD ["bash"]