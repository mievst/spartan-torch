"""Sweep attention variants over sequence length: latency + peak memory.

Methodology (frozen, see AGENTS.md "benchmark methodology"):

- CUDA (refuses to run on CPU — numbers would be meaningless), eval, no_grad
- warmup 10 iters, then median of 30 timed iters with
  ``torch.cuda.synchronize()`` around each iter
- ``torch.cuda.reset_peak_memory_stats()`` before the timed section,
  ``torch.cuda.max_memory_allocated()`` after
- fixed seed, fixed batch/dtype, one variant × one length per measurement
- OOM is recorded as empty cells, the sweep continues

Usage::

    uv run python scripts/bench_attention.py [--seq-lens 256 512 1024 2048 4096 8192]
                                             [--batch-size 8] [--write-results]

Outputs: ``bench/results_attention.csv``,
``bench/pareto_seq_latency_mem.png``. ``--write-results`` refreshes the bench
table in ``RESULTS.md``.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spartan_torch import (  # noqa: E402
    LinearTransformerAttention,
    LinformerAttention,
    MultiHeadAttention,
    PerformerAttention,
    ReformerAttention,
)

D_MODEL, N_HEADS, HS = 512, 8, 64
WARMUP, REPS = 10, 30


def build_variant(name: str, max_len: int) -> torch.nn.Module:
    if name == "mha_manual":
        return MultiHeadAttention(D_MODEL, HS, N_HEADS, D_MODEL)
    if name == "mha_sdpa":
        return MultiHeadAttention(D_MODEL, HS, N_HEADS, D_MODEL, use_sdpa=True)
    if name == "linformer":
        return LinformerAttention(D_MODEL, HS, N_HEADS, D_MODEL, proj_k=256, max_seq_len=max_len)
    if name == "performer":
        return PerformerAttention(D_MODEL, HS, N_HEADS, D_MODEL)
    if name == "linear":
        return LinearTransformerAttention(D_MODEL, HS, N_HEADS, D_MODEL)
    if name == "reformer":
        return ReformerAttention(D_MODEL, HS, N_HEADS, D_MODEL, n_hashes=4)
    raise ValueError(name)


VARIANTS = ["mha_manual", "mha_sdpa", "linformer", "performer", "linear", "reformer"]


def measure(model: torch.nn.Module, x: torch.Tensor) -> tuple[float, float, str]:
    """Return (median latency ms, peak memory MB, status).

    Status is ``ok``; ``SPILL`` when the allocator oversubscribed device
    memory (WDDM shared-memory paging — the latency number then measures
    paging, not the kernel, and is invalid as a GPU benchmark); OOM raises
    ``torch.OutOfMemoryError``.
    """
    model.eval()
    total_mb = torch.cuda.get_device_properties(x.device).total_memory / 2**20
    with torch.no_grad():
        for _ in range(WARMUP):
            out = model(x, x, x)
            out = out[0] if isinstance(out, tuple) else out
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            out = model(x, x, x)
            out = out[0] if isinstance(out, tuple) else out
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        peak_mb = torch.cuda.max_memory_allocated() / 2**20
    status = "SPILL" if peak_mb > total_mb else "ok"
    return statistics.median(times), peak_mb, status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--write-results", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("bench_attention.py requires CUDA")
    device = torch.device("cuda")
    env = f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | cuda {torch.version.cuda}"

    torch.manual_seed(0)
    rows: list[dict] = []
    max_len = max(args.seq_lens)
    for name in args.variants:
        model = build_variant(name, max_len).to(device)
        for seq in args.seq_lens:
            x = torch.randn(args.batch_size, seq, D_MODEL, device=device)
            try:
                lat_ms, peak_mb, status = measure(model, x)
                rows.append({"variant": name, "seq_len": seq, "batch": args.batch_size,
                             "latency_ms": f"{lat_ms:.2f}" if status == "ok" else "",
                             "peak_mem_mb": f"{peak_mb:.1f}", "status": status})
                shown = f"lat={lat_ms:8.2f}ms peak={peak_mb:8.1f}MB" if status == "ok" else f"{status} peak={peak_mb:.1f}MB"
                print(f"{name:12s} seq={seq:5d} {shown}")
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                rows.append({"variant": name, "seq_len": seq, "batch": args.batch_size,
                             "latency_ms": "", "peak_mem_mb": "", "status": "OOM"})
                print(f"{name:12s} seq={seq:5d} OOM")
        del model
        torch.cuda.empty_cache()

    bench_dir = ROOT / "bench"
    bench_dir.mkdir(exist_ok=True)
    csv_path = bench_dir / "results_attention.csv"
    with open(csv_path, "w", newline="") as f:
        f.write(f"# {env}\n")
        w = csv.DictWriter(f, fieldnames=["variant", "seq_len", "batch", "latency_ms", "peak_mem_mb", "status"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}")

    png_path = bench_dir / "pareto_seq_latency_mem.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for name in args.variants:
            ok = [r for r in rows if r["variant"] == name and r.get("status") == "ok"]
            xs = [int(r["seq_len"]) for r in ok]
            ys = [float(r["latency_ms"]) for r in ok]
            ms = [float(r["peak_mem_mb"]) for r in ok]
            if xs:
                ax1.plot(xs, ys, marker="o", label=name)
                ax2.plot(xs, ms, marker="o", label=name)
        for ax, title, ylabel in ((ax1, "latency vs seq_len", "median ms"),
                                 (ax2, "peak memory vs seq_len", "MB")):
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("seq_len")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, which="both", alpha=0.3)
        fig.suptitle(f"attention pareto (batch={args.batch_size})\n{env}", fontsize=8)
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        print(f"wrote {png_path}")
    except ImportError:
        print("matplotlib missing — plot skipped")

    if args.write_results:
        lines = ["| variant | " + " | ".join(f"seq {s}" for s in args.seq_lens) + " |",
                 "| --- | " + " | ".join("---" for _ in args.seq_lens) + " |"]
        for name in args.variants:
            cells = []
            for s in args.seq_lens:
                hit = next((r for r in rows if r["variant"] == name and int(r["seq_len"]) == s), None)
                cells.append(f"{hit['latency_ms']}ms / {hit['peak_mem_mb']}MB"
                             if hit and hit.get("status") == "ok" else (hit["status"] if hit else "?"))
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        table = f"env: {env} | batch={args.batch_size}\n\n" + "\n".join(lines)
        path = ROOT / "RESULTS.md"
        text = path.read_text()
        start, end = "<!-- bench:begin -->", "<!-- bench:end -->"
        block = f"{start}\n{table}\n{end}"
        pre, rest = text.split(start, 1)
        _, post = rest.split(end, 1)
        path.write_text(pre + block + post)
        print(f"updated {path}")


if __name__ == "__main__":
    main()
