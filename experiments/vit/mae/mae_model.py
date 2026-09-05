"""MAE (Masked Autoencoder) built from spartan-torch primitives.

Implements the asymmetric encoder-decoder of "Masked Autoencoders Are Scalable
Vision Learners" (He et al., 2021, arXiv:2111.06377):
* a full ViT **encoder** that runs only on the visible (unmasked) patches —
  no `[CLS]` token, no mask tokens, which is the source of the 3-4x speedup;
* a lightweight **decoder** that reassembles the full sequence from the
  encoded visible patches and learned mask tokens, then reconstructs the
  (per-patch normalized) pixels.

This is a *model-assembly*, not a library primitive — it stays in
`experiments/vit/mae/` per the project convention that complete models live
in experiments while the library holds only reusable primitives
(`PatchEmbedding`, `TransformerBlock`, `MaskedToken`, `RandomPatchMasking`,
`MAEDecoderHead`).

All modules are framework-agnostic `torch.nn`; the `MAELightning` wrapper
plugs the pretrain (and optionally a linear fine-tune head) into pytorch-lightning.

Typical MAE-Base config from the paper (for reference):
img_size=224, patch_size=16, in_channels=3, encoder={embed 768, depth 12,
heads 12}, mask_ratio=0.75, decoder={embed 512, depth 8, heads 8}.
"""

from __future__ import annotations

import torch
from torch import nn

from spartan_torch import (
    MAEDecoderHead,
    MaskedToken,
    PatchEmbedding,
    RandomPatchMasking,
    TransformerBlock,
)


def _get_pos_embed(num_tokens: int, embed_dim: int) -> nn.Parameter:
    """Sinusoidal position embedding (MAE does not learn it in the paper)."""
    pos = torch.arange(num_tokens, dtype=torch.float32).unsqueeze(1)
    half = embed_dim // 2
    inv_freq = 10000 ** (-torch.arange(half, dtype=torch.float32) / half)
    pe = torch.zeros(num_tokens, embed_dim)
    pe[:, 0::2] = torch.sin(pos * inv_freq)
    pe[:, 1::2] = torch.cos(pos * inv_freq)
    return nn.Parameter(pe.unsqueeze(0), requires_grad=False)


class _ViTEncoder(nn.Module):
    """Full ViT without `[CLS]` token and without mask tokens.

    Key difference from the classification ViT: this encoder processes only
    the *kept* (unmasked) patch tokens selected by the masking schedule. The
    batch stays constant — kept tokens are packed along the sequence dim — so
    batch size must be large enough that no batch element is left with an
    empty sequence.

    state_dict layout follows the shared `experiments/vit/vision_transformer.py`
    naming (`patch_embed`, `blocks.*`, `norm`) so MAE weights transfer to the
    classification ViT for fine-tuning.
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        ff_hidden_size: int | None = None,
        attn_p: float = 0.0,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        ff_hidden = ff_hidden_size or embed_dim * 4
        self.pos_embed = _get_pos_embed(1_000, embed_dim)  # cap; trimmed per batch
        self.blocks = nn.ModuleList([
            TransformerBlock(
                in_size=embed_dim,
                head_size=embed_dim // num_heads,
                num_heads=num_heads,
                out_size=embed_dim,
                ff_hidden_size=ff_hidden,
                attn_p=attn_p,
                dropout_p=dropout_p,
                use_sdpa=True,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode visible patch tokens.

        Parameters
        ----------
        x : torch.Tensor
            Image patches ``(B, L, embed_dim)`` after `ids_keep` selection,
            i.e. unmasked tokens only.

        Returns
        -------
        torch.Tensor
            Encoded ``(B, L_keep, embed_dim)``.
        """
        L = x.size(1)
        x = x + self.pos_embed[:, :L]
        for block in self.blocks:
            x, _ = block(x)
        return self.norm(x)


class _ViTDecoder(nn.Module):
    """Lightweight MAE decoder that reconstructs the full token sequence."""

    def __init__(
        self,
        patch_embed_dim: int,
        decoder_embed_dim: int,
        decoder_depth: int,
        num_heads: int,
        ff_hidden_size: int | None = None,
    ):
        super().__init__()
        self.decoder_embed = nn.Linear(patch_embed_dim, decoder_embed_dim)
        self.mask_token = MaskedToken(decoder_embed_dim)
        self.pos_embed = _get_pos_embed(1_000, decoder_embed_dim)
        ff_hidden = ff_hidden_size or decoder_embed_dim * 4
        self.blocks = nn.ModuleList([
            TransformerBlock(
                in_size=decoder_embed_dim,
                head_size=decoder_embed_dim // num_heads,
                num_heads=num_heads,
                out_size=decoder_embed_dim,
                ff_hidden_size=ff_hidden,
                use_sdpa=True,
            )
            for _ in range(decoder_depth)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)

    def forward(
        self,
        encoded: torch.Tensor,
        ids_restore: torch.Tensor,
        num_masked: int,
    ) -> torch.Tensor:
        """Reassemble and decode.

        Parameters
        ----------
        encoded : torch.Tensor
            Encoder output for visible tokens ``(B, L_keep, patch_embed_dim)``.
        ids_restore : torch.Tensor
            ``(B, L_full)`` permutation that restores the original token order.
        num_masked : int
            Number of masked (reconstructed) positions per batch element.

        Returns
        -------
        torch.Tensor
            Decoder hidden states ``(B, L_full, decoder_embed_dim)``.
        """
        x = self.decoder_embed(encoded)
        mask_tokens = self.mask_token(num_masked).expand(x.size(0), -1, -1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        x = x + self.pos_embed[:, : x.size(1)]
        for block in self.blocks:
            x, _ = block(x)
        return self.norm(x)


class MAEModel(nn.Module):
    """Asymmetric masked autoencoder composed of spartan-torch primitives.

    Parameters
    ----------
    img_size : int
        Square input image size (token count derived from it).
    patch_size : int
        Square patch size.
    in_channels : int
        Image channels.
    mask_ratio : float, default=0.75
        Fraction of patches masked for pretraining.
    encoder_embed_dim : int, default=768
    encoder_depth : int, default=12
    encoder_heads : int, default=12
        Encoder ViT geometry.
    decoder_embed_dim : int, default=512
    decoder_depth : int, default=8
    decoder_heads : int, default=8
        Decoder ViT geometry (lightweight vs encoder).
    ff_hidden_size : int | None, default=None
        Shared FFN hidden size; ``None`` -> 4x the block embed dim.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        mask_ratio: float = 0.75,
        encoder_embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_heads: int = 8,
        ff_hidden_size: int | None = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches

        self.masking = RandomPatchMasking(mask_ratio)
        self.encoder = _ViTEncoder(
            in_channels, patch_size, encoder_embed_dim,
            encoder_depth, encoder_heads, ff_hidden_size,
        )
        self.decoder = _ViTDecoder(
            encoder_embed_dim, decoder_embed_dim,
            decoder_depth, decoder_heads, ff_hidden_size,
        )
        self.head = MAEDecoderHead(
            decoder_embed_dim, in_channels, patch_size,
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Pretraining forward: encode visible, decode, reconstruct.

        Parameters
        ----------
        image : torch.Tensor
            ``(B, C, H, W)`` with ``H = W``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``loss`` (scalar), ``pred`` (B, L, pixels), ``target`` (normalized
            pixels), ``mask`` (B, L, bool).
        """
        tokens = self.encoder.patch_embed(image)
        ids_keep, ids_restore = self.masking(image.size(0), self.num_patches, image.device)
        visible = torch.gather(tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))
        encoded = self.encoder(visible)
        num_masked = self.num_patches - ids_keep.size(1)

        mask = torch.ones(image.size(0), self.num_patches, dtype=torch.bool, device=image.device)
        mask.scatter_(1, ids_keep, False)

        decoder_out = self.decoder(encoded, ids_restore, num_masked)
        loss = self.head(decoder_out, image, mask)
        # decoder_out is already in the original token order (the decoder
        # reassembled it via ids_restore) — no second gather here.
        pred = self.head.patch_norm(self.head.projector(decoder_out))
        target = self.head.patch_norm(self.head._to_patches(image))
        return {"loss": loss, "pred": pred, "target": target, "mask": mask}
