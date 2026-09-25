"""Sweep sequence mixers over sequence length: latency + peak memory.

Methodology (frozen, see AGENTS.md "benchmark methodology"):

- CUDA (refuses to run on CPU — numbers would be meaningless), eval, no_grad
- warmup 10 iters, then median of 30 timed iters with
  ``torch.cuda.synchronize()`` around each iter
- ``torch.cuda.reset_peak_memory_stats()`` before the timed section,
  ``torch.cuda.max_memory_allocated()`` after
- fixed seed, fixed batch/dtype, one variant × one length per measurement
- OOM is recorded as empty cells, the sweep continues

Modes:

- ``prefill`` (default table): full-sequence forward. Attention variants
  run ``model(x, x, x)``; SSM mixers run ``model(x)``.
- ``decode``: KV/SSM-state decode steps. The state is built by one
  full-sequence prefill of length ``seq_len``, then single-token steps are
  timed (latency per step). MHA runs with ``is_causal=True`` and its
  ``past_key_value`` cache; subquadratic attentions expose no step API and
  are excluded from this mode.

SSM variants bench the pure-PyTorch paths (``mamba-ssm`` kernels are
Linux-only and not installed here) — see the Mamba docstrings.

Usage::

    uv run python scripts/bench_attention.py [--mode prefill|decode]
                                             [--seq-lens 256 512 ... 32768]
                                             [--batch-size 8] [--write-results]

Outputs: ``bench/results_attention.csv`` (``mode`` column),
``bench/pareto_seq_latency_mem.png`` (prefill),
``bench/pareto_decode_latency_mem.png`` (decode). ``--write-results``
refreshes both bench tables in ``RESULTS.md`` (rows merge across modes —
run each mode once before writing).
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
    Mamba2Mixer,
    Mamba3Mixer,
    MambaMixer,
    MultiHeadAttention,
    PerformerAttention,
    ReformerAttention,
)

D_MODEL, N_HEADS, HS = 512, 8, 64
WARMUP, REPS = 10, 30

# torch>=2.13 raises AcceleratorError (not OutOfMemoryError) on some CUDA
# allocation failures — catch both so the sweep survives either spelling.
_OOM_ERRORS = (torch.OutOfMemoryError, getattr(torch, "AcceleratorError", torch.OutOfMemoryError))

ATTENTION_VARIANTS = ["mha_manual", "mha_sdpa", "linformer", "performer", "linear", "reformer"]
SSM_VARIANTS = ["mamba1", "mamba2", "mamba3_siso", "mamba3_mimo"]
VARIANTS_PREFILL = ATTENTION_VARIANTS + SSM_VARIANTS
# Subquadratic attentions expose no single-step API — decode compares the
# KV-cached causal MHA against the SSM states.
VARIANTS_DECODE = ["mha_manual", "mha_sdpa"] + SSM_VARIANTS


def build_variant(name: str, max_len: int, mode: str = "prefill") -> torch.nn.Module:
    causal = mode == "decode"
    if name == "mha_manual":
        return MultiHeadAttention(D_MODEL, HS, N_HEADS, D_MODEL, is_causal=causal)
    if name == "mha_sdpa":
        return MultiHeadAttention(D_MODEL, HS, N_HEADS, D_MODEL, is_causal=causal, use_sdpa=True)
    if name == "linformer":
        return LinformerAttention(D_MODEL, HS, N_HEADS, D_MODEL, proj_k=256, max_seq_len=max_len)
    if name == "performer":
        return PerformerAttention(D_MODEL, HS, N_HEADS, D_MODEL)
    if name == "linear":
        return LinearTransformerAttention(D_MODEL, HS, N_HEADS, D_MODEL)
    if name == "reformer":
        return ReformerAttention(D_MODEL, HS, N_HEADS, D_MODEL, n_hashes=4)
    if name == "mamba1":
        # Sequential loop, not the associative scan: identical math
        # (unit-tested bitwise), but the assoc path materializes (B,E,L,N)
        # and OOMs small cards — it would benchmark our memory overhead,
        # not the architecture's linear profile.
        return MambaMixer(D_MODEL, use_associative_scan=False)
    if name == "mamba2":
        return Mamba2Mixer(D_MODEL, d_state=64, num_heads=16, head_dim=64, n_groups=2)
    if name == "mamba3_siso":
        return Mamba3Mixer(D_MODEL, d_state=64, head_dim=64, n_groups=1)
    if name == "mamba3_mimo":
        return Mamba3Mixer(D_MODEL, d_state=64, head_dim=64, n_groups=1, is_mimo=True, mimo_rank=4)
    raise ValueError(name)


def _is_ssm(name: str) -> bool:
    return name in SSM_VARIANTS


def _unwrap(out: torch.Tensor | tuple) -> torch.Tensor:
    return out[0] if isinstance(out, tuple) else out


def measure_prefill(name: str, model: torch.nn.Module, x: torch.Tensor) -> tuple[float, float, str]:
    """Return (median latency ms, peak memory MB, status) for a full forward.

    Status is ``ok``; ``SPILL`` when the allocator oversubscribed device
    memory (WDDM shared-memory paging — the latency number then measures
    paging, not the kernel, and is invalid as a GPU benchmark); OOM raises
    ``torch.OutOfMemoryError``.
    """
    model.eval()
    total_mb = torch.cuda.get_device_properties(x.device).total_memory / 2**20
    ssm = _is_ssm(name)
    with torch.no_grad():
        for _ in range(WARMUP):
            _unwrap(model(x) if ssm else model(x, x, x))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            _unwrap(model(x) if ssm else model(x, x, x))
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        peak_mb = torch.cuda.max_memory_allocated() / 2**20
    status = "SPILL" if peak_mb > total_mb else "ok"
    return statistics.median(times), peak_mb, status


def measure_decode(name: str, model: torch.nn.Module, x: torch.Tensor) -> tuple[float, float, str]:
    """Return (median latency ms, peak memory MB, status) for one decode step.

    The recurrent/KV state is built by a single full-sequence prefill of
    ``x`` (length ``seq_len``); then single-token steps are timed. Memory
    is the step peak (prefill allocations are freed from the peak counter
    by ``reset_peak_memory_stats`` after the state is built).
    """
    model.eval()
    total_mb = torch.cuda.get_device_properties(x.device).total_memory / 2**20
    tok = x[:, -1:]
    with torch.no_grad():
        if _is_ssm(name):
            _, cache = model(x)
            def step():
                nonlocal cache
                _, cache = model(tok, cache)
        else:
            _, past = model(x, x, x)
            def step():
                nonlocal past
                _, new_kv = model(tok, tok, tok, past_key_value=past)
                past = (
                    torch.cat([past[0], new_kv[0]], dim=-2),
                    torch.cat([past[1], new_kv[1]], dim=-2),
                )
        for _ in range(WARMUP):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        peak_mb = torch.cuda.max_memory_allocated() / 2**20
    status = "SPILL" if peak_mb > total_mb else "ok"
    return statistics.median(times), peak_mb, status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="prefill", choices=["prefill", "decode"])
    ap.add_argument("--seq-lens", type=int, nargs="+",
                    default=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--write-results", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("bench_attention.py requires CUDA")
    device = torch.device("cuda")
    env = f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | cuda {torch.version.cuda}"

    variants = args.variants or (VARIANTS_PREFILL if args.mode == "prefill" else VARIANTS_DECODE)
    if args.mode == "decode":
        bad = [v for v in variants if v not in VARIANTS_DECODE]
        if bad:
            sys.exit(f"decode mode has no step API for: {', '.join(bad)}")
    measure = measure_decode if args.mode == "decode" else measure_prefill

    torch.manual_seed(0)
    bench_dir = ROOT / "bench"
    bench_dir.mkdir(exist_ok=True)
    csv_path = bench_dir / "results_attention.csv"
    rows: list[dict] = []
    max_len = max(args.seq_lens)
    for name in variants:
        model = build_variant(name, max_len, args.mode).to(device)
        for seq in args.seq_lens:
            x = torch.randn(args.batch_size, seq, D_MODEL, device=device)
            try:
                lat_ms, peak_mb, status = measure(name, model, x)
                rows.append({"variant": name, "seq_len": seq, "batch": args.batch_size,
                             "mode": args.mode,
                             "latency_ms": f"{lat_ms:.2f}" if status == "ok" else "",
                             "peak_mem_mb": f"{peak_mb:.1f}", "status": status})
                shown = f"lat={lat_ms:8.2f}ms peak={peak_mb:8.1f}MB" if status == "ok" else f"{status} peak={peak_mb:.1f}MB"
                print(f"{name:12s} seq={seq:5d} {shown}", flush=True)
            except _OOM_ERRORS:
                torch.cuda.empty_cache()
                rows.append({"variant": name, "seq_len": seq, "batch": args.batch_size,
                             "mode": args.mode,
                             "latency_ms": "", "peak_mem_mb": "", "status": "OOM"})
                print(f"{name:12s} seq={seq:5d} OOM", flush=True)
            _persist_csv(csv_path, env, rows, args)
        del model
        torch.cuda.empty_cache()

    png_name = "pareto_decode_latency_mem.png" if args.mode == "decode" else "pareto_seq_latency_mem.png"
    merged = _load_existing_rows(csv_path)
    _write_plot(bench_dir / png_name, env, [r for r in merged if r.get("mode", "prefill") == args.mode],
                title=f"attention {args.mode} (batch={args.batch_size})", names=variants)

    if args.write_results:
        table = _render_table(merged, args.batch_size, "prefill", args.seq_lens) + "\n\n" + \
            _render_table(merged, args.batch_size, "decode", args.seq_lens)
        block = f"<!-- bench:begin -->\nenv: {env} | batch={args.batch_size}\n\n{table}\n<!-- bench:end -->"
        path = ROOT / "RESULTS.md"
        text = path.read_text()
        start, end = "<!-- bench:begin -->", "<!-- bench:end -->"
        pre, rest = text.split(start, 1)
        _, post = rest.split(end, 1)
        path.write_text(pre + block + post)
        print(f"updated {path}")


def _persist_csv(csv_path: Path, env: str, rows: list[dict], args) -> None:
    """Merge current-mode rows into the CSV after every cell (crash-safe)."""
    merged = _load_existing_rows(csv_path)
    done = {(r["variant"], str(r["seq_len"])) for r in rows}
    merged = [r for r in merged
              if (r.get("mode", "prefill"), r.get("batch"), r["variant"], str(r["seq_len"]))
              not in {(args.mode, str(args.batch_size), v, s) for v, s in done}]
    merged.extend(rows)
    with open(csv_path, "w", newline="") as f:
        f.write(f"# {env}\n")
        w = csv.DictWriter(f, fieldnames=["variant", "seq_len", "batch", "mode", "latency_ms", "peak_mem_mb", "status"])
        w.writeheader()
        w.writerows([{k: str(v) for k, v in r.items()} for r in merged])
    print(f"wrote {csv_path}", flush=True)


def _load_existing_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    if len(lines) < 2:
        return []
    reader = csv.DictReader(lines)
    out = []
    for r in reader:
        r.setdefault("mode", "prefill")
        out.append({k: str(v) for k, v in r.items()})
    return out


def _render_table(rows: list[dict], batch: int, mode: str, seq_lens: list[int]) -> str:
    seqs = sorted({int(r["seq_len"]) for r in rows if r.get("mode", "prefill") == mode})
    seqs = seqs or seq_lens
    unit = "ms/step" if mode == "decode" else "ms"
    lines = [f"**{mode}** (latency {unit} / peak mem)",
             "| variant | " + " | ".join(f"seq {s}" for s in seqs) + " |",
             "| --- | " + " | ".join("---" for _ in seqs) + " |"]
    names = [r["variant"] for r in rows if r.get("mode", "prefill") == mode]
    names = list(dict.fromkeys(names)) or (VARIANTS_DECODE if mode == "decode" else VARIANTS_PREFILL)
    for name in names:
        cells = []
        for s in seqs:
            hit = next((r for r in rows if r["variant"] == name and int(r["seq_len"]) == s
                        and r.get("mode", "prefill") == mode and int(r["batch"]) == batch), None)
            cells.append(f"{hit['latency_ms']}ms / {hit['peak_mem_mb']}MB"
                         if hit and hit.get("status") == "ok" else (hit["status"] if hit else "?"))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_plot(png_path: Path, env: str, rows: list[dict], title: str, names: list[str]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        seen = list(dict.fromkeys(r["variant"] for r in rows)) or names
        for name in seen:
            ok = [r for r in rows if r["variant"] == name and r.get("status") == "ok"]
            xs = [int(r["seq_len"]) for r in ok]
            ys = [float(r["latency_ms"]) for r in ok]
            ms = [float(r["peak_mem_mb"]) for r in ok]
            if xs:
                ax1.plot(xs, ys, marker="o", label=name)
                ax2.plot(xs, ms, marker="o", label=name)
        for ax, plot_title, ylabel in ((ax1, "latency vs seq_len", "median ms"),
                                      (ax2, "peak memory vs seq_len", "MB")):
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("seq_len")
            ax.set_ylabel(ylabel)
            ax.set_title(plot_title)
            ax.legend(fontsize=8)
            ax.grid(True, which="both", alpha=0.3)
        fig.suptitle(f"{title}\n{env}", fontsize=8)
        fig.tight_layout()
        fig.savefig(png_path, dpi=120)
        print(f"wrote {png_path}")
    except ImportError:
        print("matplotlib missing — plot skipped")


if __name__ == "__main__":
    main()
