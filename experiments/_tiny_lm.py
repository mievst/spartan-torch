"""Shared tiny causal-LM harness for positional/norm ablations.

Synthetic task: repeating-motif next-token prediction. Each sequence is a
random motif of length ``motif`` tiled to ``seq_len``; the model must track
phase to predict the next token — a task that genuinely needs position
information and extends naturally past the training length (motif just
continues), which is what makes it a length-extrapolation probe.

Model: token embedding → (pos add) → N × causal TransformerBlock → norm →
linear head. All configs share seeds/steps/data order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spartan_torch import (  # noqa: E402
    ALiBiBias,
    LearnablePositionEmbedding,
    PositionalEncoding,
    RMSNorm,
    RotaryPositionalEmbedding,
    TransformerBlock,
)

VOCAB = 32


class TinyLM(nn.Module):
    def __init__(
        self,
        pos: str,
        seq_len: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        norm_mode: str = "pre",
        norm: str = "layernorm",
    ):
        super().__init__()
        assert pos in ("none", "learned", "sinusoidal", "rope", "alibi")
        self.pos = pos
        self.seq_len = seq_len
        hs = d_model // n_heads
        norm_layer = RMSNorm if norm == "rmsnorm" else nn.LayerNorm
        self.tok = nn.Embedding(VOCAB, d_model)
        self.add_pos = None
        self.qk_mod = None
        self.alibi = None
        if pos == "learned":
            self.add_pos = LearnablePositionEmbedding(seq_len, d_model)
        elif pos == "sinusoidal":
            self.add_pos = PositionalEncoding(d_model, max_seq_len=seq_len)
        elif pos == "rope":
            self.qk_mod = RotaryPositionalEmbedding(hs, max_seq_len=seq_len)
        elif pos == "alibi":
            self.alibi = ALiBiBias(n_heads)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, hs, n_heads, d_model, 4 * d_model,
                is_causal=(pos != "alibi"), qk_mod=self.qk_mod,
                norm_layer=norm_layer, norm_mode=norm_mode,
            )
            for _ in range(n_layers)
        ])
        self.norm = norm_layer(d_model)
        self.head = nn.Linear(d_model, VOCAB, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        L = ids.size(1)
        x = self.tok(ids)
        if self.add_pos is not None:
            x = self.add_pos(x)
        mask = self.alibi(L, L).to(x.device) if self.alibi is not None else None
        for block in self.blocks:
            x, _ = block(x, mask=mask)
        return self.head(self.norm(x))


def motif_batch(batch: int, seq_len: int, motif: int, gen: torch.Generator, device: torch.device):
    """(inputs, targets): tiled random motifs, next-token targets."""
    m = torch.randint(0, VOCAB, (batch, motif), generator=gen, device="cpu").to(device)
    full = m.repeat(1, seq_len // motif + 2)[:, : seq_len + 1]
    return full[:, :-1], full[:, 1:]


@torch.no_grad()
def eval_loss(model: TinyLM, seq_len: int, motif: int, batches: int, device: torch.device) -> float:
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(1234)
    tot, n = 0.0, 0
    for _ in range(batches):
        ids, tgt = motif_batch(32, seq_len, motif, gen, device)
        tot += nn.functional.cross_entropy(model(ids).reshape(-1, VOCAB), tgt.reshape(-1)).item()
        n += 1
    model.train()
    return tot / n


def train(model: TinyLM, steps: int, seq_len: int, motif: int, lr: float, device: torch.device,
          log_every: int = 100) -> tuple[list[float], float]:
    """Train; return (loss history at log_every, max grad norm seen)."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(0)
    hist, max_gnorm = [], 0.0
    for step in range(1, steps + 1):
        ids, tgt = motif_batch(32, seq_len, motif, gen, device)
        loss = nn.functional.cross_entropy(model(ids).reshape(-1, VOCAB), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        max_gnorm = max(max_gnorm, float(gnorm))
        opt.step()
        if step % log_every == 0:
            hist.append(loss.item())
    return hist, max_gnorm
