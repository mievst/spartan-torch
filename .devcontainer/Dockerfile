# SPDX-License-Identifier: MIT
# spartan-torch Development Container

ARG CUDA_VERSION=13.2.1
ARG UBUNTU_VERSION=24.04
ARG PYTHON_VERSION=3.13
ARG TARGETPLATFORM
FROM --platform=${TARGETPLATFORM} ghcr.io/astral-sh/uv:latest AS uv

FROM --platform=${TARGETPLATFORM} nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG JUST_VERSION=1.57.0
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


# Configure git safe directory (repo path depends on workspace mount point)
RUN git config --global --add safe.directory '*'

# Install just (task runner) — pinned, from GitHub releases
RUN curl -sSfL "https://github.com/casey/just/releases/download/${JUST_VERSION}/just-${JUST_VERSION}-x86_64-unknown-linux-musl.tar.gz" \
    | tar xzf - -C /usr/local/bin just

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