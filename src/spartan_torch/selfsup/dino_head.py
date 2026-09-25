import torch
from torch import nn
import torch.nn.functional as F


class DINOProjectionHead(nn.Module):
    """DINO projection head: bottleneck MLP + L2 norm + weight-normalized FC.

    Faithful to ``DINOHead`` in ``facebookresearch/dino``
    (``vision_transformer.py``):

    * an MLP of ``nb_layers`` linear stages
      (``in_dim -> hidden_dim -> ... -> bottleneck_dim``) with GELU
      activations;
    * L2 normalization of the bottleneck embedding;
    * a final weight-normalized fully-connected layer
      (``bottleneck_dim -> out_dim``, no bias).

    ``norm_last_layer`` does **not** toggle weight normalization itself (it is
    always applied) — it freezes the ``weight_g`` scale at 1, exactly as in
    the original.

    A ``use_bn`` (batch norm) variant optionally applies batch norm before
    each GELU (used when the backbone is a convnet); for ViT backbones it is
    left off, matching the paper's batch-norm-free setup.

    Parameters
    ----------
    in_dim : int
        Backbone feature dimension.
    hidden_dim : int
        Hidden dimension of the intermediate MLP layers.
    out_dim : int
        Number of output logits (prototype dimension).
    bottleneck_dim : int, default=256
        Bottleneck embedding dimension (L2-normalized, feeds the last layer).
    nb_layers : int, default=3
        Number of MLP linear stages before the final projection.
        ``1`` means a single ``in_dim -> bottleneck_dim`` linear.
    use_bn : bool, default=False
        Insert a batch norm before each non-final activation (convnet variant).
    norm_last_layer : bool, default=True
        Freeze the last layer's ``weight_g`` scale at 1 (original default).

    References
    ----------
    "Emerging Properties in Self-Supervised Vision Transformers" (Caron et al.,
    2021, arXiv:2104.14294).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        bottleneck_dim: int = 256,
        nb_layers: int = 3,
        use_bn: bool = False,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        if nb_layers < 1:
            raise ValueError(f"nb_layers must be >= 1, got {nb_layers}")

        if nb_layers == 1:
            self.mlp: nn.Module = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nb_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        # Alias kept for backward compatibility (`last_linear` was the old name).
        self.last_linear = self.last_layer
        # `original0` is the `weight_g` scale of the weight-norm
        # parametrization (`original1` is `weight_v`).
        weight_g = self.last_layer.parametrizations.weight.original0
        weight_g.data.fill_(1.0)
        if norm_last_layer:
            weight_g.requires_grad_(False)
        self.norm_last_layer = norm_last_layer

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project a batch of features to prototype logits.

        Parameters
        ----------
        x : torch.Tensor
            Backbone features ``(B, in_dim)``.

        Returns
        -------
        torch.Tensor
            ``(B, out_dim)`` unnormalized logits (softmaxed with temperature
            in :class:`DINOLoss`). The bottleneck embedding is L2-normalized
            *before* the last layer, not after.
        """
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)
