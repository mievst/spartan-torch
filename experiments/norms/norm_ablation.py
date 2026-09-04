"""Ablate norm placement × norm type on one task.

Grid: ``pre | post`` × ``layernorm | rmsnorm`` (``hybrid`` is covered by
``tests/test_hybridnorm.py``). Same tiny-LM task/seed/steps for all cells;
report final train loss + max grad norm (stability proxy: post-norm without
warmup is expected to spike).

Usage::

    uv run python experiments/norms/norm_ablation.py [--steps 2000]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from _tiny_lm import TinyLM, eval_loss, train  # noqa: E402

SEQ, MOTIF = 64, 7
GRID = [("pre", "layernorm"), ("pre", "rmsnorm"), ("post", "layernorm"), ("post", "rmsnorm")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    rows = []
    for norm_mode, norm in GRID:
        torch.manual_seed(0)
        # RoPE everywhere so position handling is held constant.
        model = TinyLM(pos="rope", seq_len=SEQ, n_layers=args.n_layers,
                       norm_mode=norm_mode, norm=norm).to(device)
        hist, max_gnorm = train(model, args.steps, SEQ, MOTIF, args.lr, device)
        e64 = eval_loss(model, SEQ, MOTIF, 10, device)
        rows.append({"n_layers": args.n_layers, "norm_mode": norm_mode, "norm": norm, "train_loss": f"{hist[-1]:.4f}",
                     "max_grad_norm": f"{max_gnorm:.2f}", "eval_64": f"{e64:.4f}"})
        print(f"L={args.n_layers} {norm_mode:5s} x {norm:9s} train={hist[-1]:.4f} gnorm={max_gnorm:.2f} eval={e64:.4f}")

    out = Path(__file__).resolve().parent / "results_norm.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_layers", "norm_mode", "norm", "train_loss", "max_grad_norm", "eval_64"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
