import torch
from torch import nn


class RandomPatchMasking(nn.Module):
    """Random masking strategy for masked image modelling (MAE).

    Samples ``mask_ratio`` of the ``(L,)`` token positions uniformly at random
    (independently per batch element) and returns the permutation indices
    needed to (a) keep the unmasked tokens for the encoder and (b) restore the
    original order after the decoder reassembles the full sequence from the
    kept tokens and the learned mask tokens.

    Mirrors the masking scheme of He et al. (2021): a high ``mask_ratio``
    (e.g. 0.75) makes the reconstruction task nontrivial and forces the
    encoder to reason about the whole image rather than interpolating local
    pixels.

    The module is stateless: the mask is drawn on every forward call, so the
    same input yields a *different* mask each time — random masking itself
    serves as data augmentation.

    Parameters
    ----------
    mask_ratio : float, default=0.75
        Fraction of positions to mask out, in ``[0, 1)``.

    References
    ----------
    "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021,
    arXiv:2111.06377).
    """

    def __init__(self, mask_ratio: float = 0.75):
        super().__init__()
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")
        self.mask_ratio = mask_ratio

    def forward(
        self,
        batch: int,
        length: int,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a random mask.

        Parameters
        ----------
        batch : int
            Batch size.
        length : int
            Sequence length (number of tokens), ``L``.
        device : torch.device | str | None
            Device to create the index tensors on. Defaults to the device of
            the module's parameters (CPU for this stateless module) — pass
            the input device explicitly when following tensors live on GPU.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(ids_keep, ids_restore)``, both ``(batch, L)``:

            * ``ids_keep`` — the indices of the kept (unmasked) positions,
              ``(batch, len_keep)``, sorted by the *shuffled* order produced
              by :func:`torch.randperm`. Kept tokens are extracted as
              ``x.gather(1, ids_keep)``.
            * ``ids_restore`` — the full-length permutation that puts kept
              tokens (in their shuffled order) followed by mask tokens back
              into the original order: ``torch.cat([ids_keep, ids_mask],
              dim=1).gather(1, ids_restore)`` reproduces the original
              sequence once mask tokens are substituted at the masked
              positions.
        """
        device = self._params_device() if device is None else device
        mask = torch.rand(batch, length, device=device)
        num_masked = int(self.mask_ratio * length)
        ids_shuffle = torch.argsort(mask, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, : length - num_masked]
        return ids_keep, ids_restore

    def _params_device(self) -> torch.device:
        param = next(self.parameters(), None)
        return param.device if param is not None else torch.device("cpu")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mask_ratio={self.mask_ratio})"
