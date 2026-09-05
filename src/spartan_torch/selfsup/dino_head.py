import torch
from torch import nn
import torch.nn.functional as F


class DINOProjectionHead(nn.Module):
    """DINO projection head: bottleneck MLP + L2 norm + weight-normalized FC.

    Maps a backbone feature (e.g. the ViT ``[CLS]`` token) to a ``dims[-1]``-
    dimensional pseudo-probability logit that is later softmaxed against the
    teacher's distribution. Follows the DINO design:

    * a 3-layer MLP with a bottleneck (``in -> hidden -> dims[-1]``) and GELU
      activations;
    * L2 normalization of the last hidden layer (~ the penultimate embedding);
    * a final fully-connected layer trained with weight normalization
      (``torch.nn.utils.weight_norm``), which is part of why the head resists
      collapse without batch normalization.

    A ``bn`` (batch norm) head variant optionally applies batch norm before
    each GELU (used when the backbone is a convnet); for ViT backbones it is
    left off, matching the paper's batch-norm-free setup.

    Parameters
    ----------
    in_dim : int
        Backbone feature dimension.
    hidden_dim : int
        Hidden dimension of the intermediate MLP layer(s).
    out_dim : int
        Number of output logits (cluster/prototype dimension).
    nb_layers : int, default=3
        Number of MLP layers before the final projection.
    use_bn : bool, default=False
        Insert a batch norm before each non-final activation (convnet variant).
    norm_last_layer : bool, default=False
        Apply weight normalization to the final layer. Turned off (False) for
        the teacher head.

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
        nb_layers: int = 3,
        use_bn: bool = False,
        norm_last_layer: bool = False,
    ):
        super().__init__()
        if nb_layers < 2:
            raise ValueError(f"nb_layers must be >= 2, got {nb_layers}")

        dims = [in_dim] + [hidden_dim] * (nb_layers - 1) + [out_dim]
        layers = []
        for i in range(nb_layers):
            linear = nn.Linear(dims[i], dims[i + 1], bias=False)
            layers.append(linear)
            if i < nb_layers - 1:
                if use_bn:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.GELU())
            else:
                self.last_linear = linear
        layers.append(nn.LayerNorm(dims[-1]))

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        if norm_last_layer:
            nn.utils.parametrizations.weight_norm(self.last_linear)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            scale = min(1.0, 0.02 / (m.weight.shape[0] ** 0.5))
            nn.init.trunc_normal_(m.weight, std=scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project+normalize a batch of features.

        Parameters
        ----------
        x : torch.Tensor
            Backbone features ``(B, in_dim)``.

        Returns
        -------
        torch.Tensor
            ``(B, out_dim)`` L2-normalized logits.
        """
        out = self.mlp(x)
        return F.normalize(out, dim=-1)
