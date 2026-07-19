# SPDX-License-Identifier: MIT
# Copyright (c) 2024 spartan-torch contributors

"""
Gradient checkpointing utilities for memory-efficient training.

Wraps ``torch.utils.checkpoint`` with a clean API:

- :class:`GradientCheckpointing` -- ``nn.Module`` wrapper for any module.
- :func:`checkpoint_sequential` -- segment-based checkpointing for ``Sequential``.
- :func:`selective_checkpoint` -- selective AC via ``CheckpointPolicy``.

# VRAM: ~0 MB (wrapper, no parameters)

References
----------
.. [1] Chen et al., "Training Deep Nets with Sublinear Memory Cost"
       https://arxiv.org/abs/1604.06174

Examples
--------
>>> import torch.nn as nn
>>> from spartan_torch.efficiency.checkpointing import GradientCheckpointing
>>> block = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
>>> ckpt = GradientCheckpointing(block)
>>> x = torch.randn(4, 256, requires_grad=True)
>>> y = ckpt(x)
>>> y.shape
torch.Size([4, 256])
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import (
    CheckpointPolicy as _CheckpointPolicy,
)
from torch.utils.checkpoint import (
    checkpoint as _torch_checkpoint,
)
from torch.utils.checkpoint import (
    checkpoint_sequential as _torch_checkpoint_sequential,
)
from torch.utils.checkpoint import (
    create_selective_checkpoint_contexts,
)

__all__ = [
    "CheckpointPolicy",
    "GradientCheckpointing",
    "checkpoint_sequential",
    "selective_checkpoint",
]

CheckpointPolicy = _CheckpointPolicy


class GradientCheckpointing(nn.Module):
    """
    Wrap a module with activation checkpointing.

    During forward, activations are discarded after use and recomputed
    during backward. Trades ~30% extra compute for O(sqrt(n)) memory.

    Parameters
    ----------
    module : nn.Module
        Module to checkpoint.
    preserve_rng_state : bool, default=True
        Preserve CUDA/CPU RNG state for reproducibility with dropout.
    use_reentrant : bool, default=False
        Use reentrant checkpointing. ``False`` is recommended (PyTorch 2.9+).
        Reentrant variant does not support leaf tensors that require grad
        in the checkpointed segment.

    # VRAM: ~0 MB (wrapper, no parameters)

    Examples
    --------
    >>> block = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256))
    >>> ckpt = GradientCheckpointing(block)
    >>> x = torch.randn(2, 128, 256, requires_grad=True)
    >>> y = ckpt(x)
    >>> y.sum().backward()  # activations recomputed, not stored
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        preserve_rng_state: bool = True,
        use_reentrant: bool = False,
    ) -> None:
        super().__init__()
        self.module = module
        self.preserve_rng_state = preserve_rng_state
        self.use_reentrant = use_reentrant

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward with activation checkpointing.

        All positional arguments that are ``Tensor`` with ``requires_grad``
        are passed through ``torch.utils.checkpoint.checkpoint``.

        Returns
        -------
        Any
            Output of the wrapped module.
        """

        def _run() -> Any:
            return self.module(*args, **kwargs)

        return _torch_checkpoint(
            _run,
            use_reentrant=self.use_reentrant,
            preserve_rng_state=self.preserve_rng_state,
        )

    def extra_repr(self) -> str:
        mod_str = repr(self.module) if len(repr(self.module)) < 60 else "..."
        return (
            f"module={mod_str}, "
            f"preserve_rng_state={self.preserve_rng_state}, "
            f"use_reentrant={self.use_reentrant}"
        )


def checkpoint_sequential(
    functions: nn.Sequential | list[nn.Module],
    segments: int,
    input: Tensor,
    *,
    preserve_rng_state: bool = True,
    use_reentrant: bool = False,
) -> Tensor:
    """
    Checkpoint a sequential model by splitting into segments.

    Each segment's activations are discarded after forward and recomputed
    during backward. Use ``segments`` to control granularity:
    more segments = less memory but more recomputation.

    Parameters
    ----------
    functions : nn.Sequential or list of nn.Module
        Sequential container or list of modules.
    segments : int
        Number of segments to split the sequential into.
    input : Tensor
        Input tensor.
    preserve_rng_state : bool, default=True
        Preserve RNG state for reproducibility.
    use_reentrant : bool, default=False
        Use reentrant checkpointing (not recommended).

    Returns
    -------
    Tensor
        Output of the sequential model.

    Examples
    --------
    >>> model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
    >>> x = torch.randn(4, 256)
    >>> y = checkpoint_sequential(model, segments=2, input=x)
    """
    result: Tensor = _torch_checkpoint_sequential(  # type: ignore[no-untyped-call]
        functions,
        segments,
        input,
        use_reentrant=use_reentrant,
        preserve_rng_state=preserve_rng_state,
    )
    return result


@contextmanager
def selective_checkpoint(
    policy: Callable[..., _CheckpointPolicy] | list[Any],
) -> Iterator[None]:
    """
    Context manager for selective activation checkpointing.

    Only recomputes operations that match the policy, passing others through.
    Reduces compute overhead of full checkpointing while keeping most memory
    savings.

    Parameters
    ----------
    policy : callable or list of OpOverload
        Either a function ``(fn, *args, **kwargs) -> CheckpointPolicy``
        or a list of ``OpOverload`` objects to always recompute.

    Yields
    ------
    None

    Examples
    --------
    >>> from torch.utils.checkpoint import CheckpointPolicy
    >>> def my_policy(fn, *args, **kwargs):
    ...     return CheckpointPolicy.MUST_RECOMPUTE
    >>> with selective_checkpoint(my_policy):
    ...     y = model(x)
    """
    ctx1, ctx2 = create_selective_checkpoint_contexts(policy)  # type: ignore[no-untyped-call]
    with ctx1, ctx2:
        yield
