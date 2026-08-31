from collections.abc import Callable

import torch
from torch import nn


class Bottleneck(nn.Module):
    """Three-layer bottleneck block from ResNet-50/101/152.

    Conv1(1x1, reduce) -> BN -> Act -> Conv2(3x3, spatial) -> BN -> Act
        -> Conv3(1x1, expand) -> BN -> [+skip] -> Act.

    Output channel count is ``out_c * expansion`` (default 4x).

    Parameters
    ----------
    in_c : int
        Input channels.
    out_c : int
        Bottleneck width. Actual output channels = ``out_c * expansion``.
    kernel_size : int | tuple[int, int], default=3
        Middle convolution kernel size.
    stride : int, default=1
        Stride for middle convolution. Output spatial size is divided by stride.
    expansion : int, default=4
        Channel expansion factor for final ``1x1`` conv.
    activation : type[nn.Module], default=nn.ReLU
        Activation class, instantiated with ``inplace=True`` when supported
        (falls back to a plain instantiation otherwise).
    norm_layer : Callable[[int], nn.Module], default=nn.BatchNorm2d
        Normalization factory called with the channel count.
    use_skip : bool, default=True
        When True, adds residual connection. When False, block acts as plain conv stack.
    dropout_p : float | None, default=None
        Dropout probability after third BN. ``None`` disables dropout.
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int = 1,
        expansion: int = 4,
        activation: type[nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
        use_skip: bool = True,
        dropout_p: float | None = None,
    ):
        super().__init__()
        self.use_skip = use_skip
        self.expansion = expansion
        try:
            self.activation = activation(inplace=True)
        except TypeError:
            self.activation = activation()

        pad = kernel_size // 2 if isinstance(kernel_size, int) else (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn1 = norm_layer(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn2 = norm_layer(out_c)

        self.conv3 = nn.Conv2d(out_c, out_c * expansion, 1, bias=False)
        self.bn3 = norm_layer(out_c * expansion)

        self.dropout = nn.Dropout2d(dropout_p) if dropout_p is not None else nn.Identity()

        if use_skip and (stride != 1 or in_c != out_c * expansion):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c * expansion, 1, stride=stride, bias=False),
                norm_layer(out_c * expansion),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)

        if self.use_skip:
            identity = self.downsample(identity)
            out = out + identity

        out = self.activation(out)
        return out
