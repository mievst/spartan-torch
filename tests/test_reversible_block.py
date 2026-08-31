import pytest
import torch
from torch import nn

from spartan_torch import (
    ChunkedFeedForward,
    FeedForward,
    ReformerAttention,
    ReversibleBlock,
    SelfAttention,
)

B, N, D, HID = 2, 7, 16, 32
HALF = D // 2


@pytest.fixture()
def x():
    torch.manual_seed(0)
    return torch.randn(B, N, D)


def close(a, b, tol=1e-6):
    return torch.allclose(a, b, atol=tol, rtol=tol)


def make_block(use_checkpoint=False, **attn_kw):
    kw = dict(n_hashes=2, bucket_size=4, n_buckets=8)
    kw.update(attn_kw)
    return ReversibleBlock(
        SelfAttention(ReformerAttention(HALF, 8, 2, HALF, **kw)),
        ChunkedFeedForward(FeedForward(HALF, 2 * HALF), chunk_size=4),
        hidden_size=HALF,
        use_checkpoint=use_checkpoint,
    )


class TestChunkedFeedForward:
    def test_equals_plain_ff(self):
        ff = FeedForward(HALF, HID)
        chunked = ChunkedFeedForward(ff, chunk_size=2)
        x = torch.randn(B, 7, HALF)
        assert torch.equal(chunked(x), ff(x))

    def test_single_call_when_within_chunk(self):
        ff = FeedForward(HALF, HID)
        chunked = ChunkedFeedForward(ff, chunk_size=1024)
        x = torch.randn(B, 7, HALF)
        assert torch.equal(chunked(x), ff(x))

    def test_longer_than_chunk_size(self):
        ff = FeedForward(HALF, HID)
        chunked = ChunkedFeedForward(ff, chunk_size=3)
        x = torch.randn(B, 10, HALF)
        assert torch.equal(chunked(x), ff(x))


class TestReversibleBlock:
    def test_forward_shape(self, x):
        block = make_block().eval()
        assert block(x).shape == (B, N, D)

    def test_reverse_recovers_input(self, x):
        block = make_block().eval()
        y = block(x)
        assert close(block.reverse(y), x, tol=1e-5)

    def test_stacked_blocks_roundtrip(self, x):
        torch.manual_seed(1)
        blocks = nn.Sequential(*[make_block() for _ in range(3)]).eval()
        y = blocks(x)
        for block in reversed(blocks):
            y = block.reverse(y)
        assert close(y, x, tol=1e-4)

    def test_reverse_raises_in_train(self, x):
        block = make_block()
        block.train()
        y = block(x)
        with pytest.raises(RuntimeError, match="eval"):
            block.reverse(y)

    def test_checkpoint_equals_plain(self, x):
        block = make_block().eval()
        y_plain = block(x)
        block.use_checkpoint = True
        y_ckpt = block(x.clone().requires_grad_(True))
        assert close(y_plain, y_ckpt.detach())

    def test_checkpoint_gradients_equal_plain(self, x):
        block = make_block()
        xp = x.clone().requires_grad_(True)
        block.use_checkpoint = True
        loss = block(xp).sum()
        loss.backward()
        grads_ckpt = {k: p.grad.clone() for k, p in block.named_parameters() if p.grad is not None}
        block.zero_grad()
        block.use_checkpoint = False
        block(xp).sum().backward()
        grads_plain = {k: p.grad.clone() for k, p in block.named_parameters() if p.grad is not None}
        assert grads_ckpt.keys() == grads_plain.keys()
        assert all(torch.allclose(grads_ckpt[k], grads_plain[k], atol=1e-5) for k in grads_ckpt)

    def test_state_dict_no_duplicates(self, x):
        block = make_block()
        names = list(block.state_dict())
        assert len(names) == len(set(names))
        assert any("f.attention.R" in k for k in names)
        assert any("norm_f.weight" in k for k in names)

    def test_state_dict_roundtrip(self, x):
        block = make_block()
        block2 = make_block()
        block2.load_state_dict(block.state_dict())
        block.eval()
        block2.eval()
        assert torch.equal(block(x), block2(x))
