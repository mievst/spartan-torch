from collections.abc import Callable

import torch
from torch import nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution (MobileNetV1 building block).

    Depthwise (k x k, stride S) -> BN -> Act -> Pointwise (1x1) -> BN -> [+skip] -> Act.

    Replaces a regular ``k x k`` convolution with two cheaper ones: a depthwise
    pass that filters each input channel independently (``groups=in_c``) and a
    pointwise ``1x1`` projection that mixes the channels. Cost drops from
    ``k^2 * in_c * out_c`` to ``k^2 * in_c + in_c * out_c`` per spatial position.

    Parameters
    ----------
    in_c : int
        Input channels.
    out_c : int
        Output channels (after pointwise convolution).
    stride : int, default=1
        Stride for the depthwise convolution. Output spatial size is divided by stride.
    kernel_size : int | tuple[int, int], default=3
        Kernel size for the depthwise convolution.
    activation : type[nn.Module], default=nn.ReLU
        Activation class, instantiated with ``inplace=True`` when supported
        (falls back to a plain instantiation otherwise).
    norm_layer : Callable[[int], nn.Module], default=nn.BatchNorm2d
        Normalization factory called with the channel count.
    use_skip : bool, default=False
        When True, adds a residual connection between input and output.
        Canonical MobileNetV1 has no residuals, so the default is off; enable
        explicitly when you want the depthwise-separable-plus-skip pattern
        (e.g. Xception-style middle flow). Effective only when
        ``stride == 1`` and ``in_c == out_c`` (no downsampling projection is
        created, unlike ResNet blocks).
    dropout_p : float | None, default=None
        Dropout probability after the second BN. ``None`` disables dropout.
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        stride: int = 1,
        kernel_size: int | tuple[int, int] = 3,
        activation: type[nn.Module] = nn.ReLU,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
        use_skip: bool = False,
        dropout_p: float | None = None,
    ):
        super().__init__()
        self.use_skip = use_skip and stride == 1 and in_c == out_c
        self.stride = stride
        try:
            self.activation = activation(inplace=True)
        except TypeError:
            self.activation = activation()

        pad = kernel_size // 2 if isinstance(kernel_size, int) else (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size, stride=stride, padding=pad, groups=in_c, bias=False)
        self.bn1 = norm_layer(in_c)

        self.conv2 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn2 = norm_layer(out_c)

        self.dropout = nn.Dropout2d(dropout_p) if dropout_p is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        if self.use_skip:
            out = out + identity

        out = self.activation(out)
        return out
