import torch
from torch import nn
import torch.nn.functional as F


class PatchNorm(nn.Module):
    """Per-patch mean/variance normalization.

    Normalizes each patch token *independently* by its own mean and variance
    over the feature dim, before the reconstruction target is computed. This
    is the target normalization used by MAE: ``x_norm = (x - mean) / sqrt(var +
    eps)`` where ``mean``/``var`` are taken over the (flattened) pixels of each
    patch.

    Works on both the reconstruction target ``(B, L, patch_pixels)`` shape and
    on the decoder output (before the final projection is applied). Only the
    target is normalized in the MAE objective; the prediction is compared raw
    (see :class:`MAEDecoderHead`). Only the masked patches contribute to
    the loss.

    Parameters
    ----------
    eps : float, default=1e-6
        Small constant added to the variance to avoid division by zero.

    References
    ----------
    "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021,
    arXiv:2111.06377).
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize per patch.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, L, patch_pixels)``.

        Returns
        -------
        torch.Tensor
            Same shape, each patch normalized to unit-ish scale.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps)


class MAEDecoderHead(nn.Module):
    """Reconstruction head and loss for a masked autoencoder decoder.

    Takes the final decoder hidden states ``(B, L, decoder_dim)`` and up-projects
    each token to the flattened per-patch pixel count ``patch_size ** 2 * C``.
    The prediction is compared against patch-normalized image pixels, and the
    mean-squared error is averaged over the *masked* tokens only (the kept
    tokens are not supervised), matching MAE.

    Compose this with a
    :class:`~spartan_torch.MaskedToken` + decoder
    :class:`~spartan_torch.TransformerBlock` stack; the caller is responsible
    for assembling the full sequence (kept + mask tokens, restored order) and
    reshaping the input image into flattened patches ``(B, L, patch_pixels)``.

    Parameters
    ----------
    decoder_dim : int
        Hidden size of the decoder's last layer.
    out_channels : int
        Input image channels (3 for RGB).
    patch_size : int
        Side length of a square patch.
    eps : float, default=1e-6
        Variance epsilon for :class:`PatchNorm`.

    References
    ----------
    "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021,
    arXiv:2111.06377).
    """

    def __init__(
        self,
        decoder_dim: int,
        out_channels: int,
        patch_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.projector = nn.Linear(decoder_dim, out_channels * patch_size * patch_size)
        self.patch_norm = PatchNorm(eps=eps)

    def forward(
        self,
        decoder_out: torch.Tensor,
        image: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the MAE reconstruction loss over masked patches.

        Parameters
        ----------
        decoder_out : torch.Tensor
            Decoder hidden states ``(B, L, decoder_dim)``.
        image : torch.Tensor
            Original input image ``(B, C, H, W)`` where
            ``H = W = patch_size * sqrt(L)``.
        mask : torch.Tensor
            Boolean mask ``(B, L)``, ``True`` = masked out (reconstructed).
            Only these positions contribute to the loss.

        Returns
        -------
        torch.Tensor
            Scalar mean-squared error over the masked patches, backpropagated
            through both the decoder projection and the encoder (via the kept
            tokens).
        """
        prediction = self.projector(decoder_out)
        patches = self._to_patches(image)
        target = self.patch_norm(patches)
        loss = (prediction - target) ** 2
        return loss[mask].mean()

    def _to_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Flatten ``(B, C, H, W)`` into per-patch pixel vectors ``(B, L, patch_pixels)``."""
        b, c, h, w = image.shape
        p = self.patch_size
        patches = image.view(b, c, h // p, p, w // p, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        return patches.view(b, -1, c * p * p)
