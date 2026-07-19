# spartan-torch Agent Instructions

This document defines mandatory rules and workflows for all agents working on the spartan-torch project.

## Core Principles

1. **VRAM-First Design**: Every block must target ≤4GB VRAM. Annotate VRAM usage in docstrings.
2. **Strict Quality Gates**: No merge without passing ALL checks (lint, type, test).
3. **Performance Proof Required**: Optimizations (torch.compile, FlashAttention, Triton kernels) must include benchmark evidence.
4. **Minimal Dependencies**: Core uses only torch + bitsandbytes. Optional deps in extras.
5. **Reproducibility**: All experiments must be reproducible with seeds and configs.

---

## Mandatory Review Checklist

### For Every PR/Commit

- [ ] **Ruff**: `ruff check .` passes with zero warnings
- [ ] **Ruff Format**: `ruff format --check .` passes
- [ ] **Ty**: `ty check src/spartan_torch` passes
- [ ] **Unit Tests**: `pytest tests/unit -x --tb=short` passes
- [ ] **Bandit**: `bandit -r src/spartan_torch` passes (no high-severity)
- [ ] **VRAM Annotations**: All new/modified blocks have VRAM docstring comments

### For Optimizations

- [ ] **Before/After Benchmarks**: Compare throughput (it/s) and VRAM (MB)
- [ ] **Correctness Test**: `torch.allclose(optimized_out, baseline_out, rtol=1e-3, atol=1e-4)`
- [ ] **Gradcheck**: `torch.autograd.gradcheck` passes for both
- [ ] **No Silent Failures**: FlashAttention/Triton kernels have CPU fallback

---

## Code Style Requirements

### Type Hints (Mandatory)
```python
# GOOD
def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    ...

# BAD - no types
def forward(self, x, mask=None):
    ...
```

### VRAM Annotations (Mandatory for Blocks)
```python
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional FlashAttention-2.

    # VRAM: ~200 MB @ batch=4, seq=512, d_model=256 (fp16 + grad checkpoint)
    # VRAM: ~350 MB @ batch=4, seq=512, d_model=256 (fp16, no checkpoint)
    """
```

### NumPy Docstrings (Mandatory for Public API)
```python
def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    Apply multi-head attention.

    Parameters
    ----------
    x : Tensor
        Input tensor of shape (batch, seq_len, d_model).
    mask : Optional[Tensor], default=None
        Attention mask of shape (batch, seq_len) or (batch, 1, seq_len, seq_len).

    Returns
    -------
    Tensor
        Output tensor of shape (batch, seq_len, d_model).

    Raises
    ------
    ValueError
        If `x.ndim != 3` or `mask` shape incompatible.

    Examples
    --------
    >>> attn = MultiHeadAttention(256, 8)
    >>> x = torch.randn(2, 128, 256)
    >>> out = attn(x)
    >>> out.shape
    torch.Size([2, 128, 256])
    """
```

---

## Block Implementation Template

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 spartan-torch contributors

"""
<Module docstring with VRAM annotation>
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from spartan_torch.efficiency import GradientCheckpointing, enable_flash_attention


class BlockName(nn.Module):
    """
    <One-line summary>.

    <Extended description>.

    # VRAM: ~X MB @ batch=Y, seq=Z, d_model=W (fp16 + grad_ckpt)
    """

    def __init__(
        self,
        d_model: int,
        *,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn and enable_flash_attention()

        # ... layers ...

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # ... implementation ...

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, dropout={self.dropout}"
```

---

## Local Development

### Quick Commands
```bash
make check          # ruff + ty + pytest unit
make lint           # ruff check + format check
make format         # ruff format + fix
make typecheck      # ty check
make test           # all unit tests
make security       # bandit scan
make build          # wheel + sdist
```

### Opencode Skill
```bash
opencode run spartan-review      # full strict review
opencode run spartan-check       # quick check
opencode run spartan-new-block   # scaffold new block
opencode run spartan-release     # release automation
```

---

## VRAM Budget Guidelines

| Component | Target (4GB) | Notes |
|-----------|--------------|-------|
| Model weights (fp16) | ~1.5 GB | 200M params max |
| Activations (grad checkpoint) | ~1.0 GB | Checkpoint every 2-3 layers |
| Optimizer states (8-bit Adam) | ~0.5 GB | Use `bitsandbytes` |
| Gradients | ~0.3 GB | Accumulate if needed |
| **Headroom** | **~0.7 GB** | CUDA context, fragmentation |

---

## Testing Standards

### Unit Tests
- Test forward + backward pass
- Test `torch.autograd.gradcheck` (double precision)
- Test shape invariants for various input sizes
- Test with/without optional args (mask, dropout=0)

---

## Dependency Policy

### Core (Always Installed)
- `torch`, `torchvision`, `bitsandbytes`

### Optional Extras
| Extra | Purpose | When to Use |
|-------|---------|-------------|
| `dev` | Lint, type, test, security | Development |
| `bench` | Triton, FlashAttn | Benchmarking |

### Adding Dependencies
1. Add to `pyproject.toml` under appropriate `[project.optional-dependencies]`
2. Run `uv lock`

---

## Release Process

1. Run `make check` to verify everything passes
2. Bump version: `opencode run spartan-release --patch` (or --minor/--major)
3. Tag and push: `git tag v0.1.1 && git push origin main --tags`

---

## Project Vision & Roles

### What is spartan-torch
A collection of PyTorch building blocks for training LLMs on a single consumer GPU (≤4GB VRAM). Focus on efficiency: quantization, LoRA, gradient checkpointing, FlashAttention, Triton kernels. Each block is a clean, well-documented, tested nn.Module.

### Who does what
- **Human (maintainer)**: Writes all domain blocks. Makes architectural decisions. Manages GitHub repo, commits, issues, CI/CD setup. Chooses priorities and starting blocks.
- **Agent (opencode)**: Code review (ruff, ty, types, VRAM annotations, NumPy docstrings). Writes unit tests (gradcheck, shape, backward, edge cases). Formats and lints. Runs benchmarks for optimizations. Writes documentation. Handles releases. Triton kernels if needed. Monitors VRAM budget ≤4GB per block.

### Workflow per block
```
Human: writes block in blocks/<category>/<name>.py
  → Agent: ruff check + ruff format + ty check
  → Agent: writes tests in tests/unit/blocks/<category>/
  → Agent: gradcheck + VRAM profile
  → Agent: style/typing fixes if needed
  → Done → next block
```

### Block Roadmap (27 weeks)

| Stage | Weeks | Blocks |
|-------|-------|--------|
| **1A** CV Foundation | 1-3 | ResNet, MobileNetV2, ViT, MAE, DINO |
| **1B** NLP + Efficient Seq | 4-6 | Transformer, RoPE, Linformer, Synthesizer, HybridNorm, TinyLlama, Mamba, MoE (Shazeer/Switch/Mixtral) |
| **2A** Generative: VAE | 7-9 | VAE, VQ-VAE, VQ-VAE-2, iGPT |
| **2B** Generative: Diffusion | 10-12 | DDPM, Improved DDPM, DDIM, CFG, Latent Diffusion |
| **3** Multimodal + Alignment | 13-16 | CLIP, LLaVA, InstructGPT/RLHF |
| **4** Reinforcement Learning | 17-20 | DQN, PPO, SAC |
| **5** Agents + Tools | 21-23 | ReAct, Toolformer |
| **6** Robotics / Embodied AI | 24-27 | ACT, Diffusion Policy, RT-2 |

### Infrastructure (shared)
- GitHub repo + first commit → Human
- CI/CD (GitHub Actions) → Agent
- LoRA/QLoRA → Human or Agent (TBD)
- DevContainer fix → Human
- README URL fix → Human
- docs/ → Agent
- Triton kernels → Agent
- Experiments (training scripts) → Agent
- VRAM tests for utils/vram.py → Agent

---

## Contacts

- **Maintainer**: @yourusername
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Security**: GH Security Advisories
