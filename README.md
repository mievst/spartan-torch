# spartan-torch

**Efficient Deep Learning blocks and paper replications for low-VRAM (4GB) training.**

## Overview

`spartan-torch` provides reusable, optimized building blocks for deep learning with a strict focus on **memory efficiency**. Every block is designed to fit within **4GB VRAM** budgets while maintaining performance through:

- **Gradient Checkpointing** - Sublinear memory scaling
- **FlashAttention-2** - Fused attention kernels
- **Quantization** - 8-bit optimizers, 4-bit weights
- **Triton Kernels** - Custom fused operations

## Installation

```bash
# Core dependencies (CPU + CUDA 12.4)
pip install spartan-torch

# With benchmarking extras (Triton, FlashAttention)
pip install spartan-torch[bench]

# Development
pip install spartan-torch[dev]
```

### Install with uv (recommended)

```bash
git clone https://github.com/yourusername/spartan-torch.git
cd spartan-torch
uv sync --extra dev --extra bench
```

## Quick Start

```python
import torch
from spartan_torch import GradientCheckpointing
from spartan_torch.efficiency import FlashAttention, quantize_model_int8

# FlashAttention (SDPA + flash_attn fallback)
attn = FlashAttention(d_model=256, n_heads=8)
attn_ckpt = GradientCheckpointing(attn)

x = torch.randn(4, 512, 256, device="cuda", dtype=torch.half)
out = attn_ckpt(x)

# Quantize model to 8-bit
model = MyModel()
quantized = quantize_model_int8(model)
```

## Efficiency Utilities

```python
from spartan_torch.efficiency import (
    # Gradient checkpointing
    GradientCheckpointing,
    checkpoint_sequential,
    
    # Quantization
    quantize_model_int8,
    quantize_model_int4,
    QuantizationConfig,
    
    # FlashAttention
    FlashAttention,
    enable_flash_attention,
    is_flash_attention_available,
)
```

## VRAM Budget (4GB Target)

| Component | Budget | Notes |
|-----------|--------|-------|
| Model weights (fp16) | ~1.5 GB | 200M params max |
| Activations (grad checkpoint) | ~1.0 GB | Checkpoint every 2-3 layers |
| Optimizer states (8-bit Adam) | ~0.5 GB | Use `bitsandbytes` |
| Gradients | ~0.3 GB | Accumulate if needed |
| **Headroom** | **~0.7 GB** | CUDA context, fragmentation |

## Development

### Prerequisites

- Python 3.13+
- CUDA 12.4 (recommended)
- `uv` (recommended) or pip

### Setup

```bash
# Clone
git clone https://github.com/yourusername/spartan-torch.git
cd spartan-torch

# Install with uv
uv sync --extra dev --extra bench
```

### Commands

```bash
# Quick check (ruff + ty + unit tests)
make check

# Lint
make lint

# Format
make format

# Type check
make typecheck

# Run tests
make test

# Security scan
make security

# Build package
make build
```

### Opencode Skill

```bash
# Full strict review
opencode run spartan-review

# Quick check
opencode run spartan-check

# Scaffold new block
opencode run spartan-new-block --category attention --name MyAttention

# Release
opencode run spartan-release --patch
```

## DevContainer (Linux + CUDA)

```bash
docker build -t spartan-torch-dev .devcontainer
docker run --rm -it --gpus all -v ${PWD}:/workspace -w /workspace spartan-torch-dev bash
```

## Project Structure

```
spartan-torch/
├── src/spartan_torch/
│   ├── efficiency/
│   │   ├── checkpointing.py
│   │   ├── quantization.py
│   │   └── flash_attn.py
│   └── utils/
│       └── vram.py
├── tests/                 # Unit tests
├── .opencode/             # Opencode skill + agents
├── AGENTS.md              # Agent instructions
└── Makefile               # Common commands
```

## Contributing

1. Fork and create a feature branch
2. Run `make check` locally
3. Ensure VRAM annotations on new blocks
4. Add unit tests
5. Submit PR

### Code Standards

- **Type hints mandatory** - ty strict mode
- **NumPy docstrings** - Public API must have Args/Returns/Examples
- **VRAM annotations** - `# VRAM: ~X MB @ batch=Y, seq=Z`
- **Benchmarks required** - For any optimization claim

## License

MIT License - see [LICENSE](LICENSE)

## Citation

```bibtex
@software{spartan-torch,
  title = {spartan-torch: Efficient Deep Learning for 4GB VRAM},
  author = {spartan-torch contributors},
  year = {2024},
  url = {https://github.com/yourusername/spartan-torch}
}
```
