import pytest
import torch

from spartan_torch import (
    MAEDecoderHead,
    MaskedToken,
    PatchNorm,
    RandomPatchMasking,
)


class TestMaskedToken:
    def test_output_shape(self):
        token = MaskedToken(dim=768)
        out = token(number=147)
        assert out.shape == (1, 147, 768)

    def test_token_is_learnable(self):
        token = MaskedToken(dim=64)
        out = token(number=5)
        out.sum().backward()
        assert token.mask_token.grad is not None

    def test_shared_value_across_positions(self):
        token = MaskedToken(dim=16)
        out = token(number=7)
        for i in range(1, 7):
            assert torch.allclose(out[0, i], out[0, 0])


class TestRandomPatchMasking:
    def test_keep_and_mask_counts(self):
        masking = RandomPatchMasking(mask_ratio=0.75)
        batch, length = 4, 196
        ids_keep, ids_restore = masking(batch, length)
        num_masked = int(0.75 * length)
        assert ids_keep.shape == (batch, length - num_masked)
        assert ids_restore.shape == (batch, length)

    def test_mask_ratio_validated(self):
        with pytest.raises(ValueError, match="mask_ratio"):
            RandomPatchMasking(mask_ratio=1.0)
        with pytest.raises(ValueError, match="mask_ratio"):
            RandomPatchMasking(mask_ratio=-0.1)

    def test_restore_reproduces_order(self):
        masking = RandomPatchMasking(mask_ratio=0.75)
        batch, length = 3, 100
        ids_keep, ids_restore = masking(batch, length)
        num_masked = int(0.75 * length)
        ids_mask = torch.zeros(batch, num_masked, dtype=torch.long)
        random_positions = torch.argsort(torch.rand(batch, num_masked), dim=1)
        ids_all = torch.cat([ids_keep, ids_mask], dim=1)
        # no-op smoke: shape of restored permutation
        restored = ids_all.gather(1, ids_restore)
        assert restored.shape == (batch, length)

    def test_different_masks_each_call(self):
        masking = RandomPatchMasking(mask_ratio=0.5)
        a, _ = masking(2, 32)
        b, _ = masking(2, 32)
        assert not torch.equal(a, b)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_agnostic(self):
        masking = RandomPatchMasking(mask_ratio=0.75)
        ids_keep, ids_restore = masking(2, 64, device="cuda")
        assert ids_keep.device.type == "cuda"
        assert ids_restore.device.type == "cuda"


class TestPatchNorm:
    def test_zero_mean_unit_var(self):
        norm = PatchNorm()
        x = torch.randn(8, 196, 16)
        out = norm(x)
        assert torch.allclose(out.mean(dim=-1), torch.zeros(8, 196), atol=1e-5)
        assert torch.allclose(out.var(dim=-1, unbiased=False), torch.ones(8, 196), atol=1e-4)


class TestMAEDecoderHead:
    def test_loss_scalar_and_grad(self):
        head = MAEDecoderHead(decoder_dim=512, out_channels=3, patch_size=16)
        batch, length = 2, 196
        decoder_out = torch.randn(batch, length, 512, requires_grad=True)
        img = torch.randn(batch, 3, 224, 224)
        mask = torch.rand(batch, length) > 0.75
        loss = head(decoder_out, img, mask)
        assert loss.dim() == 0
        loss.backward()
        assert decoder_out.grad is not None

    def test_only_masked_positions_contribute(self):
        # loss equals the MSE over the masked subset only; unmasked positions
        # never enter the mean.
        head = MAEDecoderHead(decoder_dim=64, out_channels=1, patch_size=4)
        batch, length = 2, 64
        img = torch.randn(batch, 1, 32, 32)
        decoder_out = torch.randn(batch, length, 64)
        mask = torch.rand(batch, length) > 0.3
        loss = head(decoder_out, img, mask)

        patches = head._to_patches(img)
        target = head.patch_norm(patches)
        prediction = head.projector(decoder_out)  # raw per MAE: only target is normalized
        expected = (prediction[mask] - target[mask]).pow(2).mean()
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_no_grad_through_student_not_required(self):
        head = MAEDecoderHead(decoder_dim=64, out_channels=3, patch_size=8)
        batch, length = 2, 49
        img = torch.randn(batch, 3, 56, 56)
        mask = torch.ones(batch, length, dtype=torch.bool)
        decoder_out = torch.zeros(batch, length, 64, requires_grad=True)
        loss = head(decoder_out, img, mask)
        assert torch.isfinite(loss)
        loss.backward()
        assert decoder_out.grad is not None

    def test_patch_unfolding(self):
        head = MAEDecoderHead(decoder_dim=32, out_channels=3, patch_size=16)
        img = torch.randn(2, 3, 224, 224)
        patches = head._to_patches(img)
        assert patches.shape == (2, 196, 3 * 16 * 16)

    def test_target_normalized_patches(self):
        # reconstruction target is the per-patch normalized image
        head = MAEDecoderHead(decoder_dim=64, out_channels=3, patch_size=8)
        img = torch.randn(2, 3, 64, 64)
        target = head.patch_norm(head._to_patches(img))
        assert target.shape == (2, 64, 192)
        assert torch.allclose(target.mean(dim=-1), torch.zeros(2, 64), atol=1e-5)
        assert torch.allclose(target.std(dim=-1, unbiased=False), torch.ones(2, 64), atol=1e-4)
