# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure with spartan-torch package
- Core blocks: attention, norm, conv, ssm, moe, generative, rl, robotics
- Efficiency utilities: gradient checkpointing, quantization, FlashAttention
- VRAM tracking utilities
- Strict dev tooling (ruff, ty, pytest, bandit)
- Custom opencode skill with review, benchmark, profile, scaffold, release commands
- DevContainer with CUDA 12.4

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0] - 2024-07-01

### Added
- Project initialization
- Core package structure
- Configuration files (pyproject.toml, Makefile)
- Agent instructions and custom skill
- DevContainer configuration
- Initial documentation
