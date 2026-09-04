from collections.abc import Callable

import torch
from torch import nn


class InvertedResidual(nn.Module):
    """Inverted residual with linear bottleneck (MobileNetV2 building block).

    Expansion(1x1) -> BN -> Act -> Depthwise(k x k, stride S) -> BN -> Act
        -> Project(1x1) -> BN -> [+skip].

    Unlike a classic ResNet bottleneck (narrow -> wide -> wide), this block goes
    narrow -> wide (expand) -> narrow (project): the residual shortcut connects
    the thin bottleneck tensors, and the wide middle is where the depthwise
    filtering happens. The final ``1x1`` projection has NO activation ("linear
    bottleneck", section 3.2 of the MobileNetV2 paper) because ReLU destroys
    information in low-dimensional representations.

    Parameters
    ----------
    in_c : int
        Input channels.
    out_c : int
        Output channels.
    stride : int, default=1
        Stride for the depthwise convolution (``1`` or ``2``).
        Output spatial size is divided by stride.
    expansion : int, default=6
        Expansion factor. Intermediate (hidden) channels = ``in_c * expansion``.
        ``expansion <= 1`` disables the expand layer (used by the first
        block of MobileNetV2).
    kernel_size : int | tuple[int, int], default=3
        Kernel size for the depthwise convolution.
    activation : type[nn.Module], default=nn.ReLU6
        Activation class, instantiated with ``inplace=True`` when supported.
        The final projection stays linear regardless of this parameter.
    norm_layer : Callable[[int], nn.Module], default=nn.BatchNorm2d
        Normalization factory called with the channel count.
    use_skip : bool, default=True
        When True, adds a residual connection between input and output.
        Effective only when ``stride == 1`` and ``in_c == out_c``.
    dropout_p : float | None, default=None
        Dropout probability after the projection BN. ``None`` disables dropout.

    References
    ----------
    "MobileNetV2: Inverted Residuals and Linear Bottlenecks" (Sandler et al.,
    2018, arXiv:1801.04381).
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        stride: int = 1,
        expansion: int = 6,
        kernel_size: int | tuple[int, int] = 3,
        activation: type[nn.Module] = nn.ReLU6,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
        use_skip: bool = True,
        dropout_p: float | None = None,
    ):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"InvertedResidual: stride must be 1 or 2, got {stride}")
        self.use_skip = use_skip and stride == 1 and in_c == out_c
        self.use_expand = expansion > 1
        self.stride = stride
        self.expansion = expansion
        try:
            self.activation = activation(inplace=True)
        except TypeError:
            self.activation = activation()

        pad = kernel_size // 2 if isinstance(kernel_size, int) else (kernel_size[0] // 2, kernel_size[1] // 2)

        hidden = int(in_c * expansion)
        if self.use_expand:
            self.conv1 = nn.Conv2d(in_c, hidden, 1, bias=False)
            self.bn1 = norm_layer(hidden)
        else:
            hidden = in_c
            self.conv1 = nn.Identity()
            self.bn1 = nn.Identity()

        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size, stride=stride, padding=pad, groups=hidden, bias=False)
        self.bn2 = norm_layer(hidden)

        self.conv3 = nn.Conv2d(hidden, out_c, 1, bias=False)
        self.bn3 = norm_layer(out_c)

        self.dropout = nn.Dropout2d(dropout_p) if dropout_p is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if self.use_expand:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.activation(out)
        else:
            out = x

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropout(out)

        if self.use_skip:
            out = out + identity

        return out
