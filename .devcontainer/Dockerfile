# SPDX-License-Identifier: MIT
# spartan-torch Development Container

ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.13
ARG TARGETPLATFORM
FROM --platform=${TARGETPLATFORM} ghcr.io/astral-sh/uv:latest AS uv

FROM --platform=${TARGETPLATFORM} nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
COPY --from=uv /uv /uvx /bin/

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