#!/usr/bin/env python3
"""
spartan-new-block: Scaffold a new block with tests and benchmarks.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent.parent


BLOCK_TEMPLATE = '''# SPDX-License-Identifier: MIT
# Copyright (c) 2024 spartan-torch contributors

"""
{description}

# VRAM: ~{vram} MB @ batch={batch}, seq={seq}, d_model={d_model} (fp16 + grad_ckpt)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from spartan_torch.efficiency import GradientCheckpointing, enable_flash_attention


class {class_name}(nn.Module):
    """
    {description}.

    # VRAM: ~{vram} MB @ batch={batch}, seq={seq}, d_model={d_model} (fp16 + grad_ckpt)
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

        # TODO: Add layers
        # self.layer = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # TODO: Implement forward pass
        return x

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, dropout={self.dropout}"
'''

TEST_TEMPLATE = '''# SPDX-License-Identifier: MIT
# Copyright (c) 2024 spartan-torch contributors

"""
Tests for {class_name}.
"""

import pytest
import torch
from torch import Tensor

from spartan_torch.blocks.{category}.{module_name} import {class_name}


class Test{class_name}:
    """Tests for {class_name}."""

    @pytest.fixture
    def module(self) -> {class_name}:
        return {class_name}(d_model=256)

    @pytest.fixture
    def input_tensor(self) -> Tensor:
        return torch.randn(2, 128, 256)

    def test_forward_shape(self, module: {class_name}, input_tensor: Tensor) -> None:
        """Test forward pass output shape."""
        out = module(input_tensor)
        assert out.shape == input_tensor.shape

    def test_forward_dtype(self, module: {class_name}) -> None:
        """Test dtype preservation."""
        x = torch.randn(2, 128, 256, dtype=torch.half)
        out = module(x)
        assert out.dtype == torch.half

    def test_gradient_flow(self, module: {class_name}, input_tensor: Tensor) -> None:
        """Test gradients flow through module."""
        x = input_tensor.clone().requires_grad_(True)
        out = module(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradcheck(self, module: {class_name}) -> None:
        """Test autograd gradcheck (double precision)."""
        module = module.double()
        x = torch.randn(1, 16, 64, dtype=torch.double, requires_grad=True)
        torch.autograd.gradcheck(module, x, raise_exception=True)

    def test_no_mask(self, module: {class_name}, input_tensor: Tensor) -> None:
        """Test forward without mask."""
        out = module(input_tensor)
        assert out.shape == input_tensor.shape

    def test_with_mask(self, module: {class_name}, input_tensor: Tensor) -> None:
        """Test forward with mask."""
        mask = torch.ones(2, 128, dtype=torch.bool)
        mask[:, 64:] = False
        out = module(input_tensor, mask=mask)
        assert out.shape == input_tensor.shape

    def test_dropout_zero(self, input_tensor: Tensor) -> None:
        """Test with dropout=0."""
        module = {class_name}(d_model=256, dropout=0.0)
        out = module(input_tensor)
        assert out.shape == input_tensor.shape

    def test_eval_mode(self, module: {class_name}, input_tensor: Tensor) -> None:
        """Test eval mode."""
        module.eval()
        with torch.no_grad():
            out = module(input_tensor)
        assert out.shape == input_tensor.shape
'''

INIT_TEMPLATE = """from spartan_torch.blocks.{category}.{module_name} import {class_name}

__all__ = ["{class_name}"]
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaffold new block")
    parser.add_argument(
        "--category",
        required=True,
        choices=[
            "attention",
            "norm",
            "conv",
            "ssm",
            "moe",
            "generative",
            "rl",
            "robotics",
        ],
    )
    parser.add_argument("--name", required=True, help="Block name (PascalCase)")
    parser.add_argument("--description", default="Efficient block implementation")
    parser.add_argument("--vram", type=int, default=100, help="Estimated VRAM in MB")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    args = parser.parse_args()

    class_name = args.name
    module_name = "".join(["_" + c.lower() if c.isupper() else c for c in class_name]).lstrip("_")

    # Paths
    block_dir = ROOT / "src" / "spartan_torch" / "blocks" / args.category
    test_dir = ROOT / "tests" / "unit" / "blocks" / args.category

    block_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create files
    block_file = block_dir / f"{module_name}.py"
    test_file = test_dir / f"test_{module_name}.py"
    init_file = block_dir / "__init__.py"

    # Render templates
    context = {
        "class_name": class_name,
        "module_name": module_name,
        "category": args.category,
        "description": args.description,
        "vram": args.vram,
        "batch": args.batch,
        "seq": args.seq,
        "d_model": args.d_model,
    }

    block_file.write_text(BLOCK_TEMPLATE.format(**context))
    test_file.write_text(TEST_TEMPLATE.format(**context))

    # Update __init__.py
    init_content = INIT_TEMPLATE.format(**context)
    if init_file.exists():
        existing = init_file.read_text()
        if class_name not in existing:
            init_file.write_text(existing + "\n" + init_content)
    else:
        init_file.write_text(init_content)

    print("Created:")
    print(f"  {block_file.relative_to(ROOT)}")
    print(f"  {test_file.relative_to(ROOT)}")
    print(f"  Updated: {init_file.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
