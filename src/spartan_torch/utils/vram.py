# SPDX-License-Identifier: MIT
# Copyright (c) 2024 spartan-torch contributors

"""
VRAM tracking utilities for profiling PyTorch model memory usage.

Provides ``VRAMTracker`` with two modes:

1. ``track()`` context manager — measure VRAM between two points in code.
2. ``track_model(model)`` — wrap ``model.forward`` to auto-measure every call.

All measurements are accumulated and exported via ``report()``, ``to_dict()``,
``to_json()``, or ``save_snapshot()``.

# VRAM: ~0 MB (wrapper only, no parameters)

Examples
--------
>>> tracker = VRAMTracker()
>>> for _ in range(3):
...     with tracker.track():
...         x = torch.randn(4, 512, 256, device="cuda")
...         y = x @ x.T
...         y.sum().backward()
>>> tracker.report()
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _to_mb(bytes_: int) -> float:
    return bytes_ / (1024 * 1024)


def _ensure_device(device: int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    return device


def get_vram_usage(device: int | None = None) -> float:
    """Current allocated VRAM in MB."""
    return _to_mb(torch.cuda.memory_allocated(_ensure_device(device)))


def get_max_vram_allocated(device: int | None = None) -> float:
    """Peak allocated VRAM (since last reset) in MB."""
    return _to_mb(torch.cuda.max_memory_allocated(_ensure_device(device)))


def get_vram_reserved(device: int | None = None) -> float:
    """Current reserved VRAM (caching allocator pool) in MB."""
    return _to_mb(torch.cuda.memory_reserved(_ensure_device(device)))


def reset_vram_stats(device: int | None = None) -> None:
    """Reset CUDA memory stats for *device*."""
    torch.cuda.reset_peak_memory_stats(_ensure_device(device))
    torch.cuda.reset_accumulated_memory_stats(_ensure_device(device))


# ---------------------------------------------------------------------------
# Per-run measurement container
# ---------------------------------------------------------------------------


@dataclass
class TrackResult:
    """Single ``track()`` measurement."""

    peak_allocated_mb: float = 0.0
    current_allocated_mb: float = 0.0
    peak_reserved_mb: float = 0.0
    duration_ms: float = 0.0
    is_warmup: bool = False
    triton_kernels: list[str] = field(default_factory=list)
    flash_attention_detected: bool = False

    @property
    def fragment_mb(self) -> float:
        """Difference between peak reserved and peak allocated."""
        return max(0.0, self.peak_reserved_mb - self.peak_allocated_mb)

    @property
    def fragment_pct(self) -> float:
        """Fragment as percentage of reserved.  0 if no reserved."""
        if self.peak_reserved_mb == 0:
            return 0.0
        return (self.fragment_mb / self.peak_reserved_mb) * 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_allocated_mb": round(self.peak_allocated_mb, 2),
            "current_allocated_mb": round(self.current_allocated_mb, 2),
            "peak_reserved_mb": round(self.peak_reserved_mb, 2),
            "fragment_mb": round(self.fragment_mb, 2),
            "fragment_pct": round(self.fragment_pct, 2),
            "duration_ms": round(self.duration_ms, 2),
            "is_warmup": self.is_warmup,
            "triton_kernels": self.triton_kernels,
            "flash_attention_detected": self.flash_attention_detected,
        }


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

_ALL_TRITON_PREFIXES = ("triton_", "flash_attn")


class _TrackContext:
    """Returned by ``VRAMTracker.track()``."""

    def __init__(self, tracker: VRAMTracker, warmup: bool) -> None:
        self._tracker = tracker
        self._warmup = warmup
        self._start_event: torch.cuda.Event | None = None
        self._end_event: torch.cuda.Event | None = None
        self.result = TrackResult(is_warmup=warmup)

    def __enter__(self) -> TrackResult:
        if not self._tracker.enabled:
            return self.result
        device = self._tracker._device
        self._tracker._check_cuda()

        # Events for accurate timing
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

        # Reset CUDA peak stats so peak reflects only this run
        reset_vram_stats(device)

        # Record pre-run alloc for growth calculation
        self._pre_alloc = get_vram_usage(device)

        self._start_event.record()
        return self.result

    def __exit__(self, *args: Any) -> None:
        if not self._tracker.enabled:
            return
        torch.cuda.synchronize()
        device = self._tracker._device

        if self._end_event is not None and self._start_event is not None:
            self._end_event.record()
            self._end_event.synchronize()
            self.result.duration_ms = self._start_event.elapsed_time(self._end_event)

        self.result.peak_allocated_mb = get_max_vram_allocated(device)
        self.result.current_allocated_mb = get_vram_usage(device)
        self.result.peak_reserved_mb = get_vram_reserved(device)

        # Triton / FlashAttention detection via memory history
        if self._tracker._track_triton:
            self._detect_triton_kernels()

        # Store in tracker's run list
        self._tracker._runs.append(self.result)

    def _detect_triton_kernels(self) -> None:
        try:
            snapshot = torch.cuda.memory._snapshot()
        except (RuntimeError, AttributeError):
            return

        kernels: set[str] = set()
        flash_attn = False

        for seg in snapshot.get("segments", []):
            for block in seg.get("blocks", []):
                stack = block.get("history", [])
                for entry in stack:
                    frames = entry.get("frames", [])
                    for frame in frames:
                        filename = frame.get("filename", "")
                        for prefix in _ALL_TRITON_PREFIXES:
                            if prefix in filename:
                                kernels.add(filename)
                                if "flash_attn" in filename:
                                    flash_attn = True

        self.result.triton_kernels = sorted(kernels)
        self.result.flash_attention_detected = flash_attn


class VRAMTracker:
    """
    Accumulate VRAM measurements across multiple ``track()`` or tracked-model calls.

    Parameters
    ----------
    enabled : bool, default=True
        When ``False`` all ``track()`` calls are no-ops (zero overhead).
    device : Optional[int], default=None
        CUDA device index. ``None`` uses ``torch.cuda.current_device()``.
    track_triton : bool, default=False
        Attempt Triton / FlashAttention kernel detection via memory snapshot.
        Adds overhead — only enable when debugging kernel-level allocations.
    auto_warmup : int, default=1
        Number of initial ``track()`` calls treated as warmup (excluded from
        aggregate stats).  Set to 0 to disable.

    # VRAM: ~0 MB

    Examples
    --------
    >>> tracker = VRAMTracker(track_triton=True)
    >>> for _ in range(3):
    ...     with tracker.track():
    ...         x = torch.randn(2, 128, 256, device="cuda")
    ...         _ = x @ x.T
    >>> tracker.report()
    """

    def __init__(
        self,
        enabled: bool = True,
        device: int | None = None,
        track_triton: bool = False,
        auto_warmup: int = 1,
    ) -> None:
        self.enabled = enabled
        self._device = None if device is None else _ensure_device(device)
        self._track_triton = track_triton
        self._auto_warmup = auto_warmup
        self._runs: list[TrackResult] = []
        self._track_count = 0
        self._model_backup: Callable[..., Tensor] | None = None

    def _check_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available — VRAMTracker requires a GPU.")

    # ---- Public API -------------------------------------------------------

    def track(self, warmup: bool = False) -> _TrackContext:
        """
        Return a context manager that measures VRAM inside its block.

        Parameters
        ----------
        warmup : bool, default=False
            Mark this run as warmup (excluded from aggregate report).
        """
        if not self.enabled:
            return _TrackContext(self, warmup=True)  # no-op
        return _TrackContext(self, warmup=warmup or self._track_count < self._auto_warmup)

    def track_model(self, model: nn.Module) -> None:
        """
        Wrap ``model.forward`` so every call is automatically measured.

        Restore the original forward with ``restore_model(model)`` or
        by creating a new ``VRAMTracker``.

        Parameters
        ----------
        model : nn.Module
            Module whose ``forward`` method will be patched.
        """
        original_forward = model.forward

        def wrapped_forward(*args: Any, **kwargs: Any) -> Tensor:
            with self.track() as t:
                result = original_forward(*args, **kwargs)
            # Attach last track result to the model for in-code inspection
            model._last_vram_track = t  # ty: ignore[unresolved-attribute]
            return result

        model.forward = wrapped_forward  # type: ignore[method-assign]
        self._model_backup = original_forward

    def restore_model(self, model: nn.Module) -> None:
        """Restore original ``model.forward`` after ``track_model()``."""
        if self._model_backup is not None:
            model.forward = self._model_backup  # type: ignore[method-assign]
            self._model_backup = None

    def reset(self) -> None:
        """Clear all accumulated runs.  Does NOT reset CUDA stats."""
        self._runs.clear()
        self._track_count = 0

    @property
    def runs(self) -> list[TrackResult]:
        """All accumulated track results (including warmup)."""
        return self._runs

    @property
    def warmup_runs(self) -> list[TrackResult]:
        """Warmup-only runs."""
        return [r for r in self._runs if r.is_warmup]

    @property
    def measured_runs(self) -> list[TrackResult]:
        """Non-warmup measured runs."""
        return [r for r in self._runs if not r.is_warmup]

    # ---- Reporting --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Aggregate metrics as a dictionary.

        Fields
        ------
        num_runs : int
            Number of measured (non-warmup) runs.
        num_warmup : int
            Number of warmup runs.
        peak_allocated_mb : float
            Max peak across all measured runs.
        avg_allocated_mb : float
            Average peak across measured runs.
        current_allocated_mb : float
            Current allocated VRAM after last run.
        peak_reserved_mb : float
            Max reserved across measured runs.
        avg_reserved_mb : float
            Average reserved across measured runs.
        fragment_mb : float
            Max fragment = peak_reserved - peak_allocated.
        avg_fragment_pct : float
            Average fragmentation % of reserved.
        avg_duration_ms : float
            Average wall time per measured run (ms).
        std_duration_ms : float
            Std dev of duration.
        triton_kernels : list[str]
            Unique Triton kernel filenames seen.
        flash_attention_detected : bool
            Whether FlashAttention was observed.
        """
        measured = self.measured_runs
        if not measured:
            return {"num_runs": 0, "num_warmup": len(self.warmup_runs)}

        peaks = [r.peak_allocated_mb for r in measured]
        reserveds = [r.peak_reserved_mb for r in measured]
        fragments = [r.fragment_mb for r in measured]
        durations = [r.duration_ms for r in measured]
        all_kernels: set[str] = set()
        flash_attn = False
        for r in measured:
            all_kernels.update(r.triton_kernels)
            if r.flash_attention_detected:
                flash_attn = True

        return {
            "num_runs": len(measured),
            "num_warmup": len(self.warmup_runs),
            "peak_allocated_mb": round(max(peaks), 2),
            "avg_allocated_mb": round(sum(peaks) / len(peaks), 2),
            "std_allocated_mb": round(_std(peaks), 2),
            "current_allocated_mb": round(measured[-1].current_allocated_mb, 2),
            "peak_reserved_mb": round(max(reserveds), 2),
            "avg_reserved_mb": round(sum(reserveds) / len(reserveds), 2),
            "max_fragment_mb": round(max(fragments), 2),
            "avg_fragment_pct": round(sum(r.fragment_pct for r in measured) / len(measured), 2),
            "avg_duration_ms": round(sum(durations) / len(durations), 2),
            "std_duration_ms": round(_std(durations), 2),
            "triton_kernels": sorted(all_kernels),
            "flash_attention_detected": flash_attn,
        }

    def report(self) -> None:
        """Print formatted VRAM report to stdout."""
        measured = self.measured_runs
        num_warmup = len(self.warmup_runs)
        num_total = len(self._runs)
        num_measured = len(measured)

        print("VRAM Report")
        print("-" * 55)
        print(f"  Total runs:      {num_total}  (warmup: {num_warmup}, measured: {num_measured})")

        if not measured:
            print("  (no measured data)")
            print()
            return

        d = self.to_dict()
        print(f"  Peak allocated:  {d['peak_allocated_mb']:>8.1f} MB")
        print(
            f"  Avg allocated:   {d['avg_allocated_mb']:>8.1f} MB  +/- {d['std_allocated_mb']:.1f}"
        )
        print(f"  Current alloc:   {d['current_allocated_mb']:>8.1f} MB  (after last run)")
        print(f"  Peak reserved:   {d['peak_reserved_mb']:>8.1f} MB")
        print(
            f"  Max fragment:    {d['max_fragment_mb']:>8.1f} MB  ({d['avg_fragment_pct']:.1f}% avg)"
        )
        print(f"  Avg duration:    {d['avg_duration_ms']:>8.2f} ms  +/- {d['std_duration_ms']:.2f}")

        if d["triton_kernels"]:
            print(f"  Triton kernels:  {', '.join(d['triton_kernels'])}")
        if d["flash_attention_detected"]:
            print("  FlashAttention:  detected")

        print()

        # Per-run table
        if num_measured > 1:
            self._print_per_run(measured)

    def _print_per_run(self, measured: list[TrackResult]) -> None:
        print(f"  {'Run':<5} {'Peak(MB)':<10} {'Resv(MB)':<10} {'Frag(%)':<9} {'Dur(ms)':<9}")
        print(f"  {'-' * 5} {'-' * 10} {'-' * 10} {'-' * 9} {'-' * 9}")
        for i, r in enumerate(measured, 1):
            print(
                f"""{i:<5} {r.peak_allocated_mb:<10.1f}
                {r.peak_reserved_mb:<10.1f} {r.fragment_pct:<9.1f}
                {r.duration_ms:<9.2f}"""
            )
        print()

    def to_json(self, path: str, indent: int = 2) -> None:
        """Export aggregate report to a JSON file."""
        d = self.to_dict()
        d["runs"] = [r.to_dict() for r in self._runs]
        with open(path, "w") as f:
            json.dump(d, f, indent=indent)

    def save_snapshot(self, path: str) -> None:
        """
        Dump ``torch.cuda.memory._dump_snapshot()`` to *path* for
        chrome://tracing analysis.

        Requires ``torch.cuda.memory._record_memory_history()`` to have been
        called beforehand.
        """
        try:
            snapshot = torch.cuda.memory._dump_snapshot()
        except (RuntimeError, AttributeError) as e:
            warnings.warn(f"Cannot dump snapshot: {e}")
            return
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=1)
        print(f"Snapshot saved to {path}")


# ---------------------------------------------------------------------------
# vram_profile — decorator
# ---------------------------------------------------------------------------


class vram_profile:
    """
    Decorator that wraps a function with VRAM tracking.

    The decorated function's return value becomes a tuple ``(original_result, tracker)``
    where *tracker* is a ``VRAMTracker`` instance.

    Examples
    --------
    >>> @vram_profile
    ... def train_step(x):
    ...     return x @ x.T
    >>>
    >>> result, tracker = train_step(torch.randn(4, 256, device="cuda"))
    >>> tracker.report()
    """

    def __init__(self, enabled: bool = True, track_triton: bool = False) -> None:
        self._enabled = enabled
        self._track_triton = track_triton

    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, VRAMTracker]:
            tracker = VRAMTracker(
                enabled=self._enabled,
                track_triton=self._track_triton,
            )
            with tracker.track():
                result = func(*args, **kwargs)
            return result, tracker

        return wrapper


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return variance**0.5
