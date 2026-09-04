"""Compare positional schemes on one task + length extrapolation.

Task: repeating-motif next-token prediction (see ``experiments/_tiny_lm.py``).
Train at ``seq_len=64``, eval loss at 64 / 128 / 256 (motif continues, so a
position scheme that extrapolates keeps loss flat).

Usage::

    uv run python experiments/positional/pos_compare.py [--steps 2000]
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

VARIANTS = ["none", "learned", "sinusoidal", "rope", "alibi"]
SEQ, MOTIF = 64, 7


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    rows = []
    for pos in VARIANTS:
        torch.manual_seed(0)
        model = TinyLM(pos=pos, seq_len=SEQ).to(device)
        hist, max_gnorm = train(model, args.steps, SEQ, MOTIF, args.lr, device)
        train_loss = hist[-1]

        def safe_eval(n: int) -> str:
            # Learned/sinusoidal tables end at the training length — past it
            # they raise instead of extrapolating; "n/a" IS the result.
            try:
                return f"{eval_loss(model, n, MOTIF, 10, device):.4f}"
            except (IndexError, ValueError):
                return "n/a"

        e64, e128, e256 = safe_eval(64), safe_eval(128), safe_eval(256)
        rows.append({"pos": pos, "train_loss": f"{train_loss:.4f}", "max_grad_norm": f"{max_gnorm:.2f}",
                     "eval_64": e64, "eval_128": e128, "eval_256": e256})
        print(f"{pos:10s} train={train_loss:.4f} gnorm={max_gnorm:.2f} "
              f"eval[64]={e64} eval[128]={e128} eval[256]={e256}")

    out = Path(__file__).resolve().parent / "results_pos.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pos", "train_loss", "max_grad_norm", "eval_64", "eval_128", "eval_256"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
