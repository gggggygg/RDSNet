# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock
import time
from ultralytics.utils.sample_grid import get_pixel_angle, coords_grid, get_omni_offset
import numpy as np
from torch.nn.modules.utils import _pair
from collections import OrderedDict
import math
import torchvision
import config

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
    "rConv",
    "RDSConv",
    "r_customConv",
    "rC2f",
    "RDSC2f",
    "r_customC2f",
    "FW_Conv",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class rC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(rBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class RDSC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(RDSBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class light_RDSC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(light_RDSBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class r_customC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(r_customBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        if config.rapid_dark53_bottleneck_kernel:
            self.cv1 = Conv(c1, c_, 1, 1)
        else:
            self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class rBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = rConv(c1, c_, k[0], 1)
        self.cv2 = rConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class RDSBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RDSConv(c1, c_, k[0], 1)
        self.cv2 = RDSConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class r_customBottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = r_customConv(c1, c_, k[0], 1)
        self.cv2 = r_customConv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Contrastive Head for YOLO-World compute the region-text scores according to the similarity between image and text
    features.
    """

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        self.bias = nn.Parameter(torch.zeros([]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(nn.Module):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcut option, groups and expansion
        ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass through RepBottleneck layer."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class RepCSP(nn.Module):
    """Rep CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through RepCSP layer."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class Silence(nn.Module):
    """Silence."""

    def __init__(self):
        """Initializes the Silence module."""
        super(Silence, self).__init__()

    def forward(self, x):
        """Forward pass through Silence layer."""
        return x


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out

import RDSConv_ext
from RDSConv.RDSConv import RDSConv2d


class RDSConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, in_h=1024, in_w=1024, batch_size = 1):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        if type(k) is tuple:
            k = k[0]

        self.weight = nn.Parameter(torch.Tensor(
            c2, c1, k, k))

        self.kernel_size = _pair(k)
        self.stride = _pair(1)   ###modified now###

        self.padding = _pair(autopad(k, p, d))
        self.in_channels = c1
        self.out_channels = c2
        self.stride_in = s

        self.bias = None

        self.rotaed_offset = None
        self.offset_cache = OrderedDict()
        self.max_cache_size = 10
        self.deform_mask = None
        self.valid_index = None

        if s==2:    ###modified now###
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)\


    def tuple_mul(self, t, s):
        return tuple(i * s for i in t)

    def get_valid_kernel_idx(self, in_ch, kernel_size, batch_sz, out_h, out_w, device='cuda'):
        cx, cy = out_w // 2, out_h // 2

        y_grid = torch.arange(out_h, device=device)
        x_grid = torch.arange(out_w, device=device)
        y, x = torch.meshgrid(y_grid, x_grid, indexing='ij')
        dx = x - cx
        dy = y - cy
        dis_center = torch.sqrt(dx.float() ** 2 + dy.float() ** 2).to(torch.int32)
        cnt_down = config.imgsz//out_w

        if cnt_down>8:
            r = out_w // 2 - int(16*out_w/config.imgsz) #这个-16不对啊: 基本不影响
            r1 = r + 3  # # 外圈半径，这个+5没有按比例缩减: 问题不大
            r2 = r*2 // 3
            r3 = 0
        elif cnt_down>4:
            r = out_w // 2 - int(16*out_w/config.imgsz) #这个-16不对啊: 基本不影响
            r1 = r + 3  # # 外圈半径，这个+5没有按比例缩减: 问题不大
            r2 = r*2 // 3
            r3 = 0
        elif cnt_down>2:
            r = out_w // 2 - int(16*out_w/config.imgsz) #这个-16不对啊: 基本不影响
            r1 = r + 3  # # 外圈半径，这个+5没有按比例缩减: 问题不大
            r2 = r // 2
            r3 = 0
        elif cnt_down>1:
            r = out_w // 2 - int(16*out_w/config.imgsz) #这个-16不对啊: 基本不影响
            r1 = r + 3  # # 外圈半径，这个+5没有按比例缩减: 问题不大
            r2 = r // 2
            r3 = 0
        else:
            r = out_w // 2 - int(16*out_w/config.imgsz) #这个-16不对啊: 基本不影响
            r1 = r + 3  # # 外圈半径，这个+5没有按比例缩减: 问题不大
            r2 = 0
            r3 = 0



        if config.Conv_sampling_only_inner:
            valid_mask = torch.zeros((out_h, out_w), dtype=torch.bool, device=device)
            dilation_map = torch.ones((out_h, out_w), dtype=torch.int32, device=device)
            r1 = out_w // 2 + 3
            outer_mask = (dis_center <= r1)
            valid_mask |= outer_mask
        else:

            valid_mask = torch.zeros((out_h, out_w), dtype=torch.bool, device=device)
            dilation_map = torch.ones((out_h, out_w), dtype=torch.int32, device=device)

            # 外圈
            if config.Conv_Sparse_sampling_traing:
                outer_mask = (r2<=dis_center) # & (dis_center <= r1)  #训练的时候，数据增强后，可能会满出外圈
            else:
                outer_mask = (r2 <= dis_center) & (dis_center <= r1)

            valid_mask |= outer_mask

            if self.stride_in==2:
                # 中圈
                middle_mask = ( r3<= dis_center)&(dis_center < r2)
                sampled_middle = middle_mask & ((x+y)%2==0)
                valid_mask |= sampled_middle

                # 内圈
                inner_mask = (dis_center < r3)
                sampled_inner = inner_mask & (x%2==0) & (y%2==0)
                valid_mask |= sampled_inner
            else: #c2f 不稀疏采样
                # 中圈
                middle_mask = (r3 <= dis_center) & (dis_center < r2)
                valid_mask |= middle_mask

                # 内圈
                inner_mask = (dis_center < r3)
                valid_mask |= inner_mask

        # 扩展到 [1, 1, H, W] → 再广播到 [B, k², H, W]
        valid_mask_expand = valid_mask[None, None, :, :]  # shape [1, 1, H, W]
        valid_mask_expand = valid_mask_expand.expand(batch_sz, kernel_size * kernel_size, out_h, out_w).contiguous()  # shape [B, k², H, W]

        # 获取合法索引 [N, 2]
        valid_hw = valid_mask.nonzero(as_tuple=False)  # shape [N, 2], (y, x)
        num_valid = valid_hw.size(0)

        # broadcast in_ch x batch_sz x valid_hw
        in_c_grid = torch.arange(in_ch, device=device).view(-1, 1, 1)
        out_b_grid = torch.arange(batch_sz, device=device).view(1, -1, 1)
        out_y_grid = valid_hw[:, 0].view(1, 1, -1)
        out_x_grid = valid_hw[:, 1].view(1, 1, -1)

        # expand
        in_c_grid = in_c_grid.expand(in_ch, batch_sz, num_valid)
        out_b_grid = out_b_grid.expand(in_ch, batch_sz, num_valid)
        out_y_grid = out_y_grid.expand(in_ch, batch_sz, num_valid)
        out_x_grid = out_x_grid.expand(in_ch, batch_sz, num_valid)

        # flat index
        index = (
                in_c_grid * (batch_sz * out_h * out_w) +
                out_b_grid * (out_h * out_w) +
                out_y_grid * out_w +
                out_x_grid
        ).reshape(-1).to(torch.int32) #(N,)

        return index, dilation_map, valid_mask_expand

    def get_offset_dilation_kernel(self, in_ch, kernel_size, batch_sz, out_h, out_w, in_h, in_w, dilation_map):
        H, W = in_h, in_w
        grid_ori = coords_grid(H, W)
        angles, distances = get_pixel_angle(grid_ori)
        angles = angles.to('cuda')

        y, x = torch.meshgrid(
            torch.arange(kernel_size).float() - kernel_size // 2,
            torch.arange(kernel_size).float() - kernel_size // 2,
            indexing='ij')
        kernel_offset = torch.stack([x, y], dim=-1).to('cuda')  # [k, k, 2]

        # Apply dilation scaling
        if isinstance(dilation_map, torch.Tensor):
            dilation_map = dilation_map[None, None, None, :, :]  # → [1,1,1,H,W] for broadcasting
        kernel_offset = kernel_offset[..., None, None]
        kernel_offset_dilation = kernel_offset * dilation_map  # → [k,k,2,H,W] or scalar

        theta = torch.deg2rad(angles)[None, None]  # [1, 1, H, W]

        offset_x = kernel_offset_dilation[..., 0, :, :] * torch.cos(theta) - kernel_offset_dilation[..., 1, :, :] * torch.sin(theta)
        offset_y = kernel_offset_dilation[..., 0, :, :] * torch.sin(theta) + kernel_offset_dilation[..., 1, :, :] * torch.cos(theta)

        rotated_offset = torch.stack([offset_x, offset_y], dim=2)  # [k,k,2,H,W]
        offset = rotated_offset - kernel_offset  # get final offset

        offset =  torch.flip(offset, dims=[2])  # [k, k, 2, H, W]

        offset = offset.view(-1, H, W)  # 偶数下标存 offset_h、奇数存 offset_w
        # offset = nn.Parameter(torch.Tensor(offset), requires_grad=False)    #它随着模型保存/加载一起保留，而不是每次重建 相当于推理的时候无须再求一次，那两个预处理之后也可以加一下，大点就大点吧
        offset = nn.Parameter(offset.clone().detach(), requires_grad=False)

        return offset

    def _deform_conv2d(self, x):

        if self.rotaed_offset is None or self.rotaed_offset.shape[2:] != x.shape[-2:] or self.rotaed_offset.shape[0] != x.shape[0]:  #//原版
            key = (x.shape[0], *x.shape[-2:])

            if not hasattr(self, 'offset_cache') or self.offset_cache==None:
                self.offset_cache = OrderedDict()
                self.max_cache_size = 10

            if key in self.offset_cache:
                if self.offset_cache[key].device != x.device:
                    self.offset_cache[key] = self.offset_cache[key].to(x.device)
                self.rotaed_offset = self.offset_cache[key]
            else:
                self.rotaed_offset = get_omni_offset(x.shape[-2:], kernel_size=self.kernel_size[0]).to(x.device)[None]
                self.rotaed_offset = self.rotaed_offset.repeat_interleave(x.shape[0], 0)
                self.offset_cache[key] = self.rotaed_offset.clone().cuda()
                # 保证 cache 不超过最大容量
                if len(self.offset_cache) > self.max_cache_size:
                    self.offset_cache.popitem(last=False)  # 移除最早添加的项（最久未使用）

            self.valid_index, self.deform_mask = None, None
            if config.Conv_Sparse_sampling or config.Conv_Sparse_sampling_traing:
                if 1: #x.shape[-1]>=config.imgsz//2 : #
                    if config.Conv_Sparse_sampling_traing:  #训练过程中
                        _, dilation_map, self.deform_mask = self.get_valid_kernel_idx(self.in_channels, self.kernel_size[0], x.shape[0], x.shape[-2], x.shape[-1], device=x.device)
                    else: #非训练过程中
                        self.valid_index, dilation_map, _ = self.get_valid_kernel_idx(self.in_channels, self.kernel_size[0], x.shape[0], x.shape[-2], x.shape[-1], device=x.device)
                    self.rotaed_offset = self.get_offset_dilation_kernel(self.in_channels, self.kernel_size[0], x.shape[0], x.shape[-2], x.shape[-1], x.shape[-2], x.shape[-1],dilation_map).to(x.device)[None]
                    self.rotaed_offset = self.rotaed_offset.repeat_interleave(x.shape[0], 0)
                else:
                    self.valid_index, self.deform_mask = None, None

        if self.rotaed_offset.device != x.device:
            self.rotaed_offset = self.rotaed_offset.to(x.device)

        mask_pred = None

        input_dtype = [x.dtype, self.weight.dtype]
        if not any(dtype==self.rotaed_offset.dtype for dtype in input_dtype):
            self.rotaed_offset = self.rotaed_offset.to(x.dtype)

        if self.deform_mask is not None and self.deform_mask.dtype != x.dtype:
            self.deform_mask = self.deform_mask.to(x.dtype)

        try:
            if self.valid_index is None or config.Conv_Sparse_sampling_traing:
                if config.Conv_Sparse_sampling_traing:
                    y = torchvision.ops.deform_conv2d(x,
                                                      offset=self.rotaed_offset,  # 每像素卷积核偏置
                                                      weight=self.weight,
                                                      bias=None,
                                                      stride=self.stride,
                                                      padding=self.padding,
                                                      mask=self.deform_mask, )
                else:
                    y = torchvision.ops.deform_conv2d(x,
                                                      offset=self.rotaed_offset,  # 每像素卷积核偏置
                                                      weight=self.weight,
                                                      bias=None,
                                                      stride=self.stride,
                                                      padding=self.padding,
                                                      mask=None, )
            else:
                y = RDSConv2d(x,  #自定义可变形卷积
                              offset=self.rotaed_offset,  # 每像素卷积核偏置
                              weight=self.weight,
                              mapping_idx = self.valid_index,
                              bias=None,
                              stride=self.stride,
                              padding=self.padding,
                              mask=None, )

        except Exception as e:
            print("执行以下代码时报错：")
            print("y = torchvision.ops.deform_conv2d(input, offset=offset, weight=self.weight, bias=None, stride=self.stride, padding=self.padding, mask=mask_pred,)")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {e}")
            print(x.dtype, self.weight.dtype, self.rotaed_offset.dtype, mask_pred.dtype if mask_pred is not None else None, self.padding, self.stride)
            print('input:', x.shape)
            print('offset:', self.rotaed_offset.shape)
            print('weight:', self.weight.shape)
            print(x.device , self.rotaed_offset.device , self.weight.device)

            raise  # 继续抛出错误，方便调试

        y.to(x.dtype)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1)
            y = y + bias

        if self.stride_in==2:  ###modified now###
            y = self.downsample(y)

        if y.dtype != x.dtype:
            y = y.to(x.dtype)
        return y

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self._deform_conv2d(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(self._deform_conv2d(x)))


def build_polar_grid(
    H, W,
    r_inner_ratio=0.8,
    r_outer_ratio=1.0,
    out_h=32,
    out_w=256,
    device="cuda"
):
    """
    返回用于 grid_sample 的 polar grid
    shape: [1, out_h, out_w, 2]
    """
    cx = (W - 1) / 2
    cy = (H - 1) / 2
    r = min(cx, cy)

    r_inner = r * r_inner_ratio
    r_outer = r * r_outer_ratio

    # rho: [out_h], theta: [out_w]
    rho = torch.linspace(r_inner, r_outer, out_h, device=device)
    theta = torch.linspace(0, 2 * math.pi, out_w, device=device)

    rho, theta = torch.meshgrid(rho, theta, indexing="ij")

    x = cx + rho * torch.cos(theta)
    y = cy + rho * torch.sin(theta)

    # normalize to [-1, 1]
    x = 2 * x / (W - 1) - 1
    y = 2 * y / (H - 1) - 1

    grid = torch.stack([x, y], dim=-1)  # [out_h, out_w, 2]
    return grid.unsqueeze(0)  # [1, out_h, out_w, 2]



class rConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, in_h=1024, in_w=1024, batch_size = 1):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        if type(k) is tuple:
            k = k[0]

        self.weight = nn.Parameter(torch.Tensor(
            c2, c1, k, k))

        self.kernel_size = _pair(k)
        self.stride = _pair(1)   ###modified now###

        self.padding = _pair(autopad(k, p, d))
        self.in_channels = c1
        self.out_channels = c2
        self.stride_in = s

        self.bias = None

        self.rotaed_offset = None
        self.offset_cache = OrderedDict()
        self.max_cache_size = 25
        self.deform_mask = None
        self.valid_index = None

        if s==2:    ###modified now###
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)\


    def tuple_mul(self, t, s):
        return tuple(i * s for i in t)

    def get_valid_kernel_idx(self, in_ch, kernel_size, batch_sz, out_h, out_w, device='cuda'):
        cx, cy = out_w // 2, out_h // 2

        y_grid = torch.arange(out_h, device=device)
        x_grid = torch.arange(out_w, device=device)
        y, x = torch.meshgrid(y_grid, x_grid, indexing='ij')
        dx = x - cx
        dy = y - cy
        dis_center = torch.sqrt(dx.float() ** 2 + dy.float() ** 2).to(torch.int32)

        cnt_down = config.imgsz//out_w

        if cnt_down>8:
            r = out_w // 2 - int(16*out_w/config.imgsz) #这个-16不对啊: 基本不影响
            r1 = r + 3  # # 外圈半径，这个+5没有按比例缩减: 问题不大
            r2 = r*2 // 3
            r3 = 0
        elif cnt_down>4:
            r = out_w // 2 - int(16*out_w/config.imgsz) #这个-16不对啊: 基本不影响
            r1 = r + 3  # # 外圈半径，这个+5没有按比例缩减: 问题不大
            r2 = r*2 // 3
            r3 = 0
        elif cnt_down>2:
            r = out_w // 2 - int(16*out_w/config.imgsz) #这个-16不对啊: 基本不影响
            r1 = r + 3  # # 外圈半径，这个+5没有按比例缩减: 问题不大
            r2 = r // 2
            r3 = 0
        elif cnt_down>1:
            r = out_w // 2 - int(16*out_w/config.imgsz) #这个-16不对啊: 基本不影响
            r1 = r + 3  # # 外圈半径，这个+5没有按比例缩减: 问题不大
            r2 = r // 2
            r3 = 0
        else:
            r = out_w // 2 - int(16*out_w/config.imgsz) #这个-16不对啊: 基本不影响
            r1 = r + 3  # # 外圈半径，这个+5没有按比例缩减: 问题不大
            r2 = 0
            r3 = 0


        if config.Conv_sampling_only_inner:
            valid_mask = torch.zeros((out_h, out_w), dtype=torch.bool, device=device)
            dilation_map = torch.ones((out_h, out_w), dtype=torch.int32, device=device)
            r1 = out_w // 2 + 3
            outer_mask = (dis_center <= r1)
            valid_mask |= outer_mask
        else:

            valid_mask = torch.zeros((out_h, out_w), dtype=torch.bool, device=device)
            dilation_map = torch.ones((out_h, out_w), dtype=torch.int32, device=device)

            # 外圈
            if config.Conv_Sparse_sampling_traing:
                outer_mask = (r2<=dis_center) # & (dis_center <= r1)  #训练的时候，数据增强后，可能会满出外圈
            else:
                outer_mask = (r2 <= dis_center) & (dis_center <= r1)

            # outer_mask = (dis_center >= r2)
            valid_mask |= outer_mask
            # dilation_map[outer_mask] = 1

            if self.stride_in==2:
                # 中圈
                middle_mask = ( r3<= dis_center)&(dis_center < r2)
                # valid_mask |= middle_mask

                # sampled_middle = middle_mask & (x%2==0) & (y%2==0)

                sampled_middle = middle_mask & ((x+y)%2==0)
                valid_mask |= sampled_middle
                # dilation_map[sampled_middle] = 2

                # 内圈
                inner_mask = (dis_center < r3)
                # valid_mask |= inner_mask

                sampled_inner = inner_mask & (x%2==0) & (y%2==0)
                valid_mask |= sampled_inner
                # dilation_map[inner_mask] = 2
            else: #c2f 不稀疏采样
                # 中圈
                middle_mask = (r3 <= dis_center) & (dis_center < r2)
                valid_mask |= middle_mask

                # 内圈
                inner_mask = (dis_center < r3)
                valid_mask |= inner_mask
                # dilation_map[inner_mask] = 2



        # 扩展到 [1, 1, H, W] → 再广播到 [B, k², H, W]
        valid_mask_expand = valid_mask[None, None, :, :]  # shape [1, 1, H, W]
        valid_mask_expand = valid_mask_expand.expand(batch_sz, kernel_size * kernel_size, out_h, out_w).contiguous()  # shape [B, k², H, W]

        # 获取合法索引 [N, 2]
        valid_hw = valid_mask.nonzero(as_tuple=False)  # shape [N, 2], (y, x)
        num_valid = valid_hw.size(0)

        # broadcast in_ch x batch_sz x valid_hw
        in_c_grid = torch.arange(in_ch, device=device).view(-1, 1, 1)
        out_b_grid = torch.arange(batch_sz, device=device).view(1, -1, 1)
        out_y_grid = valid_hw[:, 0].view(1, 1, -1)
        out_x_grid = valid_hw[:, 1].view(1, 1, -1)

        # expand
        in_c_grid = in_c_grid.expand(in_ch, batch_sz, num_valid)
        out_b_grid = out_b_grid.expand(in_ch, batch_sz, num_valid)
        out_y_grid = out_y_grid.expand(in_ch, batch_sz, num_valid)
        out_x_grid = out_x_grid.expand(in_ch, batch_sz, num_valid)

        # flat index
        index = (
                in_c_grid * (batch_sz * out_h * out_w) +
                out_b_grid * (out_h * out_w) +
                out_y_grid * out_w +
                out_x_grid
        ).reshape(-1).to(torch.int32) #(N,)

        return index, dilation_map, valid_mask_expand

    def get_offset_dilation_kernel(self, in_ch, kernel_size, batch_sz, out_h, out_w, in_h, in_w, dilation_map):
        H, W = in_h, in_w
        grid_ori = coords_grid(H, W)
        angles, distances = get_pixel_angle(grid_ori)
        angles = angles.to('cuda')

        y, x = torch.meshgrid(
            torch.arange(kernel_size).float() - kernel_size // 2,
            torch.arange(kernel_size).float() - kernel_size // 2,
            indexing='ij')
        kernel_offset = torch.stack([x, y], dim=-1).to('cuda')  # [k, k, 2]

        # Apply dilation scaling
        if isinstance(dilation_map, torch.Tensor):
            dilation_map = dilation_map[None, None, None, :, :]  # → [1,1,1,H,W] for broadcasting
        kernel_offset = kernel_offset[..., None, None]
        kernel_offset_dilation = kernel_offset * dilation_map  # → [k,k,2,H,W] or scalar

        theta = torch.deg2rad(angles)[None, None]  # [1, 1, H, W]

        offset_x = kernel_offset_dilation[..., 0, :, :] * torch.cos(theta) - kernel_offset_dilation[..., 1, :, :] * torch.sin(theta)
        offset_y = kernel_offset_dilation[..., 0, :, :] * torch.sin(theta) + kernel_offset_dilation[..., 1, :, :] * torch.cos(theta)

        rotated_offset = torch.stack([offset_x, offset_y], dim=2)  # [k,k,2,H,W]
        offset = rotated_offset - kernel_offset  # get final offset

        offset =  torch.flip(offset, dims=[2])  # [k, k, 2, H, W]

        offset = offset.view(-1, H, W)  # 偶数下标存 offset_h、奇数存 offset_w
        # offset = nn.Parameter(torch.Tensor(offset), requires_grad=False)    #它随着模型保存/加载一起保留，而不是每次重建 相当于推理的时候无须再求一次，那两个预处理之后也可以加一下，大点就大点吧
        offset = nn.Parameter(offset.clone().detach(), requires_grad=False)

        return offset

    def _deform_conv2d(self, x):

        if self.rotaed_offset is None or self.rotaed_offset.shape[2:] != x.shape[-2:] or self.rotaed_offset.shape[0] != x.shape[0]:  #//原版
            key = (x.shape[0], *x.shape[-2:])

            if not hasattr(self, 'offset_cache') or self.offset_cache==None:
                self.offset_cache = OrderedDict()
                self.max_cache_size = 10

            if key in self.offset_cache:
                if self.offset_cache[key].device != x.device:
                    self.offset_cache[key] = self.offset_cache[key].to(x.device)
                self.rotaed_offset = self.offset_cache[key]
            else:
                self.rotaed_offset = get_omni_offset(x.shape[-2:], kernel_size=self.kernel_size[0]).to(x.device)[None]
                self.rotaed_offset = self.rotaed_offset.repeat_interleave(x.shape[0], 0)
                self.offset_cache[key] = self.rotaed_offset.clone().cuda()
                # 保证 cache 不超过最大容量
                if len(self.offset_cache) > self.max_cache_size:
                    self.offset_cache.popitem(last=False)  # 移除最早添加的项（最久未使用）

            self.valid_index, self.deform_mask = None, None
            if config.Conv_Sparse_sampling or config.Conv_Sparse_sampling_traing:
                if 1:#x.shape[-1] < config.imgsz: #
                    if config.Conv_Sparse_sampling_traing:
                        _, dilation_map, self.deform_mask = self.get_valid_kernel_idx(self.in_channels, self.kernel_size[0], x.shape[0], x.shape[-2], x.shape[-1], device=x.device)
                    else:
                        self.valid_index, dilation_map, _ = self.get_valid_kernel_idx(self.in_channels, self.kernel_size[0], x.shape[0], x.shape[-2], x.shape[-1], device=x.device)
                    self.rotaed_offset = self.get_offset_dilation_kernel(self.in_channels, self.kernel_size[0], x.shape[0], x.shape[-2], x.shape[-1], x.shape[-2], x.shape[-1],dilation_map).to(x.device)[None]
                    self.rotaed_offset = self.rotaed_offset.repeat_interleave(x.shape[0], 0)
                else:
                    self.valid_index, self.deform_mask = None, None

        if self.rotaed_offset.device != x.device:
            self.rotaed_offset = self.rotaed_offset.to(x.device)

        mask_pred = None

        input_dtype = [x.dtype, self.weight.dtype]
        if not any(dtype==self.rotaed_offset.dtype for dtype in input_dtype):
            self.rotaed_offset = self.rotaed_offset.to(x.dtype)

        if self.deform_mask is not None and self.deform_mask.dtype != x.dtype:
            self.deform_mask = self.deform_mask.to(x.dtype)

        try:
            if self.valid_index is None or config.Conv_Sparse_sampling_traing:
                if config.Conv_Sparse_sampling_traing:
                    y = torchvision.ops.deform_conv2d(x,
                                                      offset=self.rotaed_offset,  # 每像素卷积核偏置
                                                      weight=self.weight,
                                                      bias=None,
                                                      stride=self.stride,
                                                      padding=self.padding,
                                                      mask=self.deform_mask, )
                else:
                    y = torchvision.ops.deform_conv2d(x,
                                                      offset=self.rotaed_offset,  # 每像素卷积核偏置
                                                      weight=self.weight,
                                                      bias=None,
                                                      stride=self.stride,
                                                      padding=self.padding,
                                                      mask=None, )
            else:
                y = RDSConv2d(x,  #自定义可变形卷积
                              offset=self.rotaed_offset,  # 每像素卷积核偏置
                              weight=self.weight,
                              mapping_idx = self.valid_index,
                              bias=None,
                              stride=self.stride,
                              padding=self.padding,
                              mask=None, )

        except Exception as e:
            print("执行以下代码时报错：")
            print("y = torchvision.ops.deform_conv2d(input, offset=offset, weight=self.weight, bias=None, stride=self.stride, padding=self.padding, mask=mask_pred,)")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {e}")
            print(x.dtype, self.weight.dtype, self.rotaed_offset.dtype, mask_pred.dtype if mask_pred is not None else None, self.padding, self.stride)
            print('input:', x.shape)
            print('offset:', self.rotaed_offset.shape)
            print('weight:', self.weight.shape)
            print(x.device , self.rotaed_offset.device , self.weight.device)

            raise  # 继续抛出错误，方便调试

        y.to(x.dtype)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1)
            y = y + bias

        if self.stride_in==2:  ###modified now###
            y = self.downsample(y)

        if y.dtype != x.dtype:
            y = y.to(x.dtype)
        return y

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self._deform_conv2d(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(self._deform_conv2d(x)))


# # rconv 加速推理版
# class rConv(nn.Module):  # class RDSConv(nn.Module): 由于之前训练的预训练权重版本原因，暂时使用名称rConv而不是RDSConv
#     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
#
#     default_act = nn.SiLU()  # default activation
#
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, in_h=1024, in_w=1024, batch_size = 1):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         # self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
#         if type(k) is tuple:
#             k = k[0]
#
#         self.weight = nn.Parameter(torch.Tensor(
#             c2, c1, k, k))
#
#         self.kernel_size = _pair(k)
#         self.stride = _pair(1)   ###modified now###
#         # self.stride = _pair(s)
#
#         self.padding = _pair(autopad(k, p, d))
#         self.in_channels = c1
#         self.out_channels = c2
#         self.stride_in = s
#
#         # self.bias = nn.Parameter(torch.Tensor(c2))  ###modified now###
#         self.bias = None
#
#         self.rotaed_offset = None
#         self.offset_cache = OrderedDict()
#         self.max_cache_size = 25
#         self.deform_mask = None
#         self.valid_index = None
#
#         if s==2:    ###modified now###
#             self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         stdv = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)\
#
#
#     def tuple_mul(self, t, s):
#         return tuple(i * s for i in t)
#
#     def get_valid_kernel_idx(self, in_ch, kernel_size, batch_sz, out_h, out_w, device='cuda'):
#         cx, cy = out_w // 2, out_h // 2
#
#         y_grid = torch.arange(out_h, device=device)
#         x_grid = torch.arange(out_w, device=device)
#         y, x = torch.meshgrid(y_grid, x_grid, indexing='ij')
#         dx = x - cx
#         dy = y - cy
#         dis_center = torch.sqrt(dx.float() ** 2 + dy.float() ** 2).to(torch.int32)
#
#         r = out_w // 2 - int(16*out_w/config.imgsz) #这个-16不对啊
#         r1 = r + 5  #这里在coco预训练和微调的时候可能要改一下，因为某些场景可能超过径向
#         r2 = r * 2 // 4
#         r3 = r * 1 // 4
#
#         ########## 原方法
#         valid_mask = torch.zeros((out_h, out_w), dtype=torch.bool, device=device)
#         dilation_map = torch.ones((out_h, out_w), dtype=torch.int32, device=device)
#         #
#         # # 外圈
#         outer_mask = (r2 <= dis_center) & (dis_center <= r1)
#         # outer_mask = (dis_center >= r2)
#         valid_mask |= outer_mask
#         # # dilation_map[outer_mask] = 1
#         #
#         # # 中圈
#         middle_mask = (r3 <= dis_center) & (dis_center < r2)
#         sampled_middle = middle_mask & ((x + y) % 3!=0)
#         valid_mask |= sampled_middle
#         # dilation_map[middle_mask] = 1.5
#
#         # 内圈
#         inner_mask = (dis_center < r3)
#         sampled_inner = inner_mask & ((x + y) % 3 != 0)  #可以试试%2
#         valid_mask |= sampled_inner
#         # dilation_map[inner_mask] = 2
#
#         ########## 测试
#         # valid_mask = torch.ones((out_h, out_w), dtype=torch.bool, device=device)
#         # dilation_map = torch.ones((out_h, out_w), dtype=torch.int32, device=device)
#         #
#         # # 外圈
#         # outer_mask = (dis_center <= r1) & (dis_center >= r2)
#         # # dilation_map[outer_mask] = 1
#         #
#         # # 中圈
#         # middle_mask = (dis_center < r2) & (dis_center >= r3)
#         # # dilation_map[sampled_middle] = 2
#         #
#         # # 内圈
#         # inner_mask = (dis_center < r2)
#         # dilation_map[inner_mask] = 2
#
#         # dilation_k = 2.0  # 内圈最大膨胀系数
#         #
#         # # 归一化距离：r1 = r - dis_center，注意clip保证非负
#         # normalized = ((r2 - dis_center) / r2).clamp(min=0)
#         #
#         # # dilation连续变化：越靠近中心越大
#         # dilation_map = 1 + normalized ** 2 * (dilation_k - 1)
#         #
#         # # 可选：限制最小最大值
#         # dilation_map = dilation_map.clamp(min=1.0, max=dilation_k)
#         #
#         # outer_mask = (dis_center >= r2)
#         # dilation_map[outer_mask] = 1
#
#
#
#         # 扩展到 [1, 1, H, W] → 再广播到 [B, k², H, W]
#         valid_mask_expand = valid_mask[None, None, :, :]  # shape [1, 1, H, W]
#         valid_mask_expand = valid_mask_expand.expand(batch_sz, kernel_size * kernel_size, out_h, out_w).contiguous()  # shape [B, k², H, W]
#
#         # 获取合法索引 [N, 2]
#         valid_hw = valid_mask.nonzero(as_tuple=False)  # shape [N, 2], (y, x)
#         num_valid = valid_hw.size(0)
#
#         # broadcast in_ch x batch_sz x valid_hw
#         in_c_grid = torch.arange(in_ch, device=device).view(-1, 1, 1)
#         out_b_grid = torch.arange(batch_sz, device=device).view(1, -1, 1)
#         out_y_grid = valid_hw[:, 0].view(1, 1, -1)
#         out_x_grid = valid_hw[:, 1].view(1, 1, -1)
#
#         # expand
#         in_c_grid = in_c_grid.expand(in_ch, batch_sz, num_valid)
#         out_b_grid = out_b_grid.expand(in_ch, batch_sz, num_valid)
#         out_y_grid = out_y_grid.expand(in_ch, batch_sz, num_valid)
#         out_x_grid = out_x_grid.expand(in_ch, batch_sz, num_valid)
#
#         # flat index
#         index = (
#                 in_c_grid * (batch_sz * out_h * out_w) +
#                 out_b_grid * (out_h * out_w) +
#                 out_y_grid * out_w +
#                 out_x_grid
#         ).reshape(-1).to(torch.int32) #(N,)
#
#         return index, dilation_map, valid_mask_expand
#
#     def get_offset_dilation_kernel(self, in_ch, kernel_size, batch_sz, out_h, out_w, in_h, in_w, dilation_map):
#         H, W = in_h, in_w
#         grid_ori = coords_grid(H, W)
#         angles, distances = get_pixel_angle(grid_ori)
#         angles = angles.to('cuda')
#
#         y, x = torch.meshgrid(
#             torch.arange(kernel_size).float() - kernel_size // 2,
#             torch.arange(kernel_size).float() - kernel_size // 2,
#             indexing='ij')
#         kernel_offset = torch.stack([x, y], dim=-1).to('cuda')  # [k, k, 2]
#
#         # Apply dilation scaling
#         if isinstance(dilation_map, torch.Tensor):
#             dilation_map = dilation_map[None, None, None, :, :]  # → [1,1,1,H,W] for broadcasting
#         kernel_offset = kernel_offset[..., None, None]
#         kernel_offset_dilation = kernel_offset * dilation_map  # → [k,k,2,H,W] or scalar
#
#         theta = torch.deg2rad(angles)[None, None]  # [1, 1, H, W]
#
#         offset_x = kernel_offset_dilation[..., 0, :, :] * torch.cos(theta) - kernel_offset_dilation[..., 1, :, :] * torch.sin(theta)
#         offset_y = kernel_offset_dilation[..., 0, :, :] * torch.sin(theta) + kernel_offset_dilation[..., 1, :, :] * torch.cos(theta)
#
#         rotated_offset = torch.stack([offset_x, offset_y], dim=2)  # [k,k,2,H,W]
#         offset = rotated_offset - kernel_offset  # get final offset
#
#         offset =  torch.flip(offset, dims=[2])  # [k, k, 2, H, W]
#
#         offset = offset.view(-1, H, W)  # 偶数下标存 offset_h、奇数存 offset_w
#         # offset = nn.Parameter(torch.Tensor(offset), requires_grad=False)    #它随着模型保存/加载一起保留，而不是每次重建 相当于推理的时候无须再求一次，那两个预处理之后也可以加一下，大点就大点吧
#         offset = nn.Parameter(offset.clone().detach(), requires_grad=False)
#
#         return offset
#
#     def _deform_conv2d(self, x):
#
#         if self.rotaed_offset is None or self.rotaed_offset.shape[2:] != x.shape[-2:] or self.rotaed_offset.shape[0] != x.shape[0]:  #//原版
#             key = (x.shape[0], *x.shape[-2:])
#
#             if not hasattr(self, 'offset_cache'):
#                 self.offset_cache = OrderedDict()
#                 self.max_cache_size = 10
#
#             if key in self.offset_cache:
#                 if self.offset_cache[key].device != x.device:
#                     self.offset_cache[key] = self.offset_cache[key].to(x.device)
#                 self.rotaed_offset = self.offset_cache[key]
#             else:
#                 self.rotaed_offset = get_omni_offset(x.shape[-2:], kernel_size=self.kernel_size[0]).to(x.device)[None]
#                 self.rotaed_offset = self.rotaed_offset.repeat_interleave(x.shape[0], 0)
#                 self.offset_cache[key] = self.rotaed_offset.clone().cuda()
#                 # 保证 cache 不超过最大容量
#                 if len(self.offset_cache) > self.max_cache_size:
#                     self.offset_cache.popitem(last=False)  # 移除最早添加的项（最久未使用）
#
#             self.valid_index, self.deform_mask = None, None
#             if config.Conv_Sparse_sampling:
#                 if x.shape[-1]>128 and self.stride_in==2: #c2f 原本就步长为1，不加速
#                     self.valid_index, dilation_map, self.deform_mask = self.get_valid_kernel_idx(self.in_channels, self.kernel_size[0], x.shape[0], x.shape[-2], x.shape[-1], device=x.device)
#                     self.rotaed_offset = self.get_offset_dilation_kernel(self.in_channels, self.kernel_size[0], x.shape[0], x.shape[-2], x.shape[-1], x.shape[-2], x.shape[-1],dilation_map).to(x.device)[None]
#                     self.rotaed_offset = self.rotaed_offset.repeat_interleave(x.shape[0], 0)
#
#
#         if self.rotaed_offset.device != x.device:
#             self.rotaed_offset = self.rotaed_offset.to(x.device)
#
#         mask_pred = None
#
#         input_dtype = [x.dtype, self.weight.dtype]
#         if not any(dtype==self.rotaed_offset.dtype for dtype in input_dtype):
#             self.rotaed_offset = self.rotaed_offset.to(x.dtype)
#
#         # if self.deform_mask is not None:
#         #     if self.deform_mask.dtype != x.dtype:
#         #         self.deform_mask = self.deform_mask.to(x.dtype)
#
#         try:
#             if self.valid_index is None:
#                 y = torchvision.ops.deform_conv2d(x,
#                                                   offset=self.rotaed_offset,  # 每像素卷积核偏置
#                                                   weight=self.weight,
#                                                   bias=None,
#                                                   stride=self.stride,
#                                                   padding=self.padding,
#                                                   mask=None, )
#                                                   # mask=self.deform_mask, )
#             else:
#                 y = deform_conv2d(x,  #自定义卷积核
#                               offset=self.rotaed_offset,  # 每像素卷积核偏置
#                               weight=self.weight,
#                               mapping_idx = self.valid_index,
#                               bias=None,
#                               stride=self.stride,
#                               padding=self.padding,
#                               mask=None, )
#
#         except Exception as e:
#             print("执行以下代码时报错：")
#             print("y = torchvision.ops.deform_conv2d(input, offset=offset, weight=self.weight, bias=None, stride=self.stride, padding=self.padding, mask=mask_pred,)")
#             print(f"错误类型: {type(e).__name__}")
#             print(f"错误信息: {e}")
#             print(x.dtype, self.weight.dtype, self.rotaed_offset.dtype, mask_pred.dtype if mask_pred is not None else None, self.padding, self.stride)
#             print('input:', x.shape)
#             print('offset:', self.rotaed_offset.shape)
#             print('weight:', self.weight.shape)
#             print(x.device , self.rotaed_offset.device , self.weight.device)
#
#             raise  # 继续抛出错误，方便调试
#
#         y.to(x.dtype)
#
#         if self.bias is not None:
#             bias = self.bias.view(1, self.out_channels, 1, 1)
#             y = y + bias
#
#         if self.stride_in==2:  ###modified now###
#             y = self.downsample(y)
#
#         if y.dtype != x.dtype:
#             y = y.to(x.dtype)
#         return y
#
#     def forward(self, x):
#         """Apply convolution, batch normalization and activation to input tensor."""
#         return self.act(self.bn(self._deform_conv2d(x)))
#
#     def forward_fuse(self, x):
#         """Perform transposed convolution of 2D data."""
#         return self.act(self.conv(self._deform_conv2d(x)))


# rconv 旧版
# class rConv(nn.Module):
#     """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
#
#     default_act = nn.SiLU()  # default activation
#
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, in_h=1024, in_w=1024, batch_size = 1):
#         """Initialize Conv layer with given arguments including activation."""
#         super().__init__()
#         # self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
#         if type(k) is tuple:
#             k = k[0]
#
#         self.weight = nn.Parameter(torch.Tensor(
#             c2, c1, k, k))
#
#         self.kernel_size = _pair(k)
#         self.stride = _pair(1)   ###modified now###
#         # self.stride = _pair(s)
#
#         self.padding = _pair(autopad(k, p, d))
#         self.in_channels = c1
#         self.out_channels = c2
#         self.stride_in = s
#
#         # self.bias = nn.Parameter(torch.Tensor(c2))  ###modified now###
#         self.bias = None
#
#         self.rotaed_offset = None
#         self.offset_cache = OrderedDict()
#         self.max_cache_size = 25
#
#         if s==2:    ###modified now###
#             self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         stdv = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)\
#
#
#     def tuple_mul(self, t, s):
#         return tuple(i * s for i in t)
#
#     def _deform_conv2d(self, x):
#
#         if self.rotaed_offset is None or self.rotaed_offset.shape[2:] != x.shape[-2:] or self.rotaed_offset.shape[0] != x.shape[0]:  #//原版
#             key = (x.shape[0], *x.shape[-2:])
#
#             if not hasattr(self, 'offset_cache'):
#                 self.offset_cache = OrderedDict()
#                 self.max_cache_size = 10
#
#             if key in self.offset_cache:
#                 if self.offset_cache[key].device != x.device:
#                     self.offset_cache[key] = self.offset_cache[key].to(x.device)
#                 self.rotaed_offset = self.offset_cache[key]
#             else:
#                 self.rotaed_offset = get_omni_offset(x.shape[-2:], kernel_size=self.kernel_size[0]).to(x.device)[None]
#                 self.rotaed_offset = self.rotaed_offset.repeat_interleave(x.shape[0], 0)
#                 self.offset_cache[key] = self.rotaed_offset.clone().cuda()
#                 # 保证 cache 不超过最大容量
#                 if len(self.offset_cache) > self.max_cache_size:
#                     self.offset_cache.popitem(last=False)  # 移除最早添加的项（最久未使用）
#
#
#         # if self.rotaed_offset is None or self.rotaed_offset.shape[2:] != x.shape[-2:] or self.rotaed_offset.shape[0] != x.shape[0]:
#         #     self.rotaed_offset = get_omni_offset(x.shape[-2:], kernel_size=self.kernel_size[0]).to(x.device)[None]
#         #     self.rotaed_offset = self.rotaed_offset.repeat_interleave(x.shape[0], 0)
#
#
#         if self.rotaed_offset.device != x.device:
#             self.rotaed_offset = self.rotaed_offset.to(x.device)
#
#         # offset = self.rotaed_offset  #防止重复转变类型，直接使用self.rotaed_offset吧
#         mask_pred = None
#
#         input_dtype = [x.dtype, self.weight.dtype]
#         if not any(dtype==self.rotaed_offset.dtype for dtype in input_dtype):
#             self.rotaed_offset = self.rotaed_offset.to(x.dtype)
#
#         try:
#             y = torchvision.ops.deform_conv2d(x,
#                                               offset=self.rotaed_offset,  # 每像素卷积核偏置
#                                               weight=self.weight,
#                                               bias=None,
#                                               stride=self.stride,
#                                               padding=self.padding,
#                                               mask=mask_pred, )
#         except Exception as e:
#             print("执行以下代码时报错：")
#             print("y = torchvision.ops.deform_conv2d(input, offset=offset, weight=self.weight, bias=None, stride=self.stride, padding=self.padding, mask=mask_pred,)")
#             print(f"错误类型: {type(e).__name__}")
#             print(f"错误信息: {e}")
#             print(x.dtype, self.weight.dtype, self.rotaed_offset.dtype, mask_pred.dtype if mask_pred is not None else None, self.padding, self.stride)
#             print('input:', x.shape)
#             print('offset:', self.rotaed_offset.shape)
#             print('weight:', self.weight.shape)
#             print(x.device , self.rotaed_offset.device , self.weight.device)
#
#             raise  # 继续抛出错误，方便调试
#
#         y.to(x.dtype)
#
#         if self.bias is not None:
#             bias = self.bias.view(1, self.out_channels, 1, 1)
#             y = y + bias
#
#         if self.stride_in==2:  ###modified now###
#             y = self.downsample(y)
#
#         if y.dtype != x.dtype:
#             y = y.to(x.dtype)
#         return y
#
#     def forward(self, x):
#         """Apply convolution, batch normalization and activation to input tensor."""
#         return self.act(self.bn(self._deform_conv2d(x)))
#
#     def forward_fuse(self, x):
#         """Perform transposed convolution of 2D data."""
#         return self.act(self.conv(self._deform_conv2d(x)))


# from custom_conv.deform_conv import deform_conv2d
class r_customConv_test(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, in_h=1024, in_w=1024, batch_size = 1):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        if type(k) is tuple:
            k = k[0]

        self.weight = nn.Parameter(torch.Tensor(
            c2, c1, k, k))

        self.kernel_size = _pair(k)
        self.stride = _pair(s)

        self.padding = _pair(autopad(k, p, d))
        self.in_channels = c1
        self.out_channels = c2
        self.stride_in = s

        self.bias = None  #####LOAF实验记得改回来，或者不改也许可以  ###modified now###
        # self.rotaed_offset = None
        out_size = in_h // self.stride_in
        self.rotaed_offset = torch.randn(batch_size, 2 * 3 * 3, out_size, out_size, device='cuda')  # 默认不反传
        self.mapping_idx = torch.full((c1*k*k*batch_size*out_size*out_size,4), 5, dtype=torch.int32, device='cuda')  ##(in_ch*kernel_h*kernel_w, batch_sz*out_h*out_w)
        self.mapping_wgt = torch.full((c1*k*k*batch_size*out_size*out_size,4), 0.25, dtype=torch.float32, device='cuda')  ##(in_ch*kernel_h*kernel_w, batch_sz*out_h*out_w)
        self.offset_cache = OrderedDict()
        self.max_cache_size = 25

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)\


    def tuple_mul(self, t, s):
        return tuple(i * s for i in t)

    def _deform_conv2d(self, x):

        # if self.rotaed_offset is None or self.tuple_mul(self.rotaed_offset.shape[2:], self.stride_in) != x.shape[-2:] or self.rotaed_offset.shape[0] != x.shape[0]:
        #     key = (x.shape[0], *x.shape[-2:])
        #
        #     if not hasattr(self, 'offset_cache'):
        #         self.offset_cache = OrderedDict()
        #         self.max_cache_size = 10
        #
        #     if key in self.offset_cache:
        #         if self.offset_cache[key].device != x.device:
        #             self.offset_cache[key] = self.offset_cache[key].to(x.device)
        #         self.rotaed_offset = self.offset_cache[key]
        #     else:
        #         if self.stride_in==2:  ###modified now###
        #             h, w = x.shape[-2:]
        #             hw = (h // 2, w // 2)
        #             self.rotaed_offset = get_omni_offset(hw, kernel_size=self.kernel_size[0]).to(x.device)[None]*2
        #         else:
        #             self.rotaed_offset = get_omni_offset(x.shape[-2:], kernel_size=self.kernel_size[0]).to(x.device)[None]
        #         self.rotaed_offset = self.rotaed_offset.repeat_interleave(x.shape[0], 0)
        #         self.offset_cache[key] = self.rotaed_offset.clone().cuda()
        #         # 保证 cache 不超过最大容量
        #         if len(self.offset_cache) > self.max_cache_size:
        #             self.offset_cache.popitem(last=False)  # 移除最早添加的项（最久未使用）
        #
        # if self.rotaed_offset.device != x.device:
        #     self.rotaed_offset = self.rotaed_offset.to(x.device)

        # offset = self.rotaed_offset
        mask_pred = None

        input_dtype = [x.dtype, self.weight.dtype]
        if not any(dtype==self.rotaed_offset.dtype for dtype in input_dtype):
            self.rotaed_offset = self.rotaed_offset.to(x.dtype)

        try:
            # y = torch.zeros(x.shape[0], self.out_channels, x.shape[2]//self.stride[0], x.shape[2]//self.stride[0], device='cuda', requires_grad=True)
            y = deform_conv2d(x,
                          offset=self.rotaed_offset,  # 每像素卷积核偏置
                          weight=self.weight,
                          mapping_idx = self.mapping_idx,
                          # mapping_wgt = self.mapping_wgt,
                          bias=None,
                          stride=self.stride,
                          padding=self.padding,
                          mask=mask_pred, )

            # y = torchvision.ops.deform_conv2d(x,
            #                                   offset=self.rotaed_offset,  # 每像素卷积核偏置
            #                                   weight=self.weight,
            #                                   bias=None,
            #                                   stride=self.stride,
            #                                   padding=self.padding,
            #                                   mask=mask_pred, )
        except Exception as e:
            print("执行以下代码时报错：")
            print("y = torchvision.ops.deform_conv2d(input, offset=offset, weight=self.weight, bias=None, stride=self.stride, padding=self.padding, mask=mask_pred,)")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {e}")
            print(x.dtype, self.weight.dtype, self.rotaed_offset.dtype, mask_pred.dtype if mask_pred is not None else None, self.padding, self.stride)
            print('input:', x.shape)
            print('offset:', self.rotaed_offset.shape)
            print('weight:', self.weight.shape)
            print(x.device , self.rotaed_offset.device , self.weight.device)

            raise  # 继续抛出错误，方便调试

        y.to(x.dtype)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1)
            y = y + bias

        if y.dtype != x.dtype:
            y = y.to(x.dtype)
        return y

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self._deform_conv2d(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(self._deform_conv2d(x)))


class r_customConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, in_h=1024, in_w=1024, batch_size = 1):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        if type(k) is tuple:
            k = k[0]

        self.weight = nn.Parameter(torch.Tensor(
            c2, c1, k, k))

        self.kernel_size = _pair(k)
        self.stride = _pair(1)   ###modified now###
        # self.stride = _pair(s)

        self.padding = _pair(autopad(k, p, d))
        self.in_channels = c1
        self.out_channels = c2
        self.stride_in = s

        # self.bias = nn.Parameter(torch.Tensor(c2))  ###modified now###
        self.bias = None

        self.rotaed_offset = None
        self.mapping_idx = None
        self.valid_index = None
        self.deform_mask = None
        self.offset_cache = OrderedDict()
        self.max_cache_size = 25

        # self.mapping_idx_cache = OrderedDict()

        if s==2:    ###modified now###
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)\


    def tuple_mul(self, t, s):
        return tuple(i * s for i in t)


    def get_valid_kernel_idx(self, in_ch, kernel_size, batch_sz, out_h, out_w, device='cuda'):
        cx, cy = out_w // 2, out_h // 2

        y_grid = torch.arange(out_h, device=device)
        x_grid = torch.arange(out_w, device=device)
        y, x = torch.meshgrid(y_grid, x_grid, indexing='ij')
        dx = x - cx
        dy = y - cy
        dis_center = torch.sqrt(dx.float() ** 2 + dy.float() ** 2).to(torch.int32)

        r = out_w // 2
        r1 = r + 5  #这里在coco预训练和微调的时候可能要改一下，因为某些场景可能超过径向
        r2 = r * 3 // 4
        r3 = r * 2 // 4

        valid_mask = torch.ones((out_h, out_w), dtype=torch.bool, device=device)
        dilation_map = torch.ones((out_h, out_w), dtype=torch.int32, device=device)

        # 外圈
        # outer_mask = (dis_center <= r1) & (dis_center >= r2)
        # valid_mask |= outer_mask
        # # dilation_map[outer_mask] = 1
        #
        # # 中圈
        # middle_mask = (dis_center < r2) & (dis_center >= r3)
        # sampled_middle = middle_mask & ((x + y) % 2 == 0)
        # valid_mask |= sampled_middle
        # # dilation_map[sampled_middle] = 2
        #
        # # 内圈
        # inner_mask = (dis_center < r3)
        # sampled_inner = inner_mask & ((x + y) % 3 == 0)
        # valid_mask |= sampled_inner
        # dilation_map[inner_mask] = 2

        # 扩展到 [1, 1, H, W] → 再广播到 [B, k², H, W]
        valid_mask_expand = valid_mask[None, None, :, :]  # shape [1, 1, H, W]
        valid_mask_expand = valid_mask_expand.expand(batch_sz, kernel_size * kernel_size, out_h, out_w).contiguous()  # shape [B, k², H, W]

        # 获取合法索引 [N, 2]
        valid_hw = valid_mask.nonzero(as_tuple=False)  # shape [N, 2], (y, x)
        num_valid = valid_hw.size(0)

        # broadcast in_ch x batch_sz x valid_hw
        in_c_grid = torch.arange(in_ch, device=device).view(-1, 1, 1)
        out_b_grid = torch.arange(batch_sz, device=device).view(1, -1, 1)
        out_y_grid = valid_hw[:, 0].view(1, 1, -1)
        out_x_grid = valid_hw[:, 1].view(1, 1, -1)

        # expand
        in_c_grid = in_c_grid.expand(in_ch, batch_sz, num_valid)
        out_b_grid = out_b_grid.expand(in_ch, batch_sz, num_valid)
        out_y_grid = out_y_grid.expand(in_ch, batch_sz, num_valid)
        out_x_grid = out_x_grid.expand(in_ch, batch_sz, num_valid)

        # flat index
        index = (
                in_c_grid * (batch_sz * out_h * out_w) +
                out_b_grid * (out_h * out_w) +
                out_y_grid * out_w +
                out_x_grid
        ).reshape(-1).to(torch.int32) #(N,)

        return index, dilation_map, valid_mask_expand


    def get_offset_dilation_kernel(self, in_ch, kernel_size, batch_sz, out_h, out_w, in_h, in_w, dilation_map):
        H, W = in_h, in_w
        grid_ori = coords_grid(H, W)
        angles, distances = get_pixel_angle(grid_ori)
        angles = angles.to('cuda')

        y, x = torch.meshgrid(
            torch.arange(kernel_size).float() - kernel_size // 2,
            torch.arange(kernel_size).float() - kernel_size // 2,
            indexing='ij')
        kernel_offset = torch.stack([x, y], dim=-1).to('cuda')  # [k, k, 2]

        # Apply dilation scaling
        if isinstance(dilation_map, torch.Tensor):
            dilation_map = dilation_map[None, None, None, :, :]  # → [1,1,1,H,W] for broadcasting
        kernel_offset = kernel_offset[..., None, None]
        kernel_offset_dilation = kernel_offset * dilation_map  # → [k,k,2,H,W] or scalar

        theta = torch.deg2rad(angles)[None, None]  # [1, 1, H, W]

        offset_x = kernel_offset_dilation[..., 0, :, :] * torch.cos(theta) - kernel_offset_dilation[..., 1, :, :] * torch.sin(theta)
        offset_y = kernel_offset_dilation[..., 0, :, :] * torch.sin(theta) + kernel_offset_dilation[..., 1, :, :] * torch.cos(theta)

        rotated_offset = torch.stack([offset_x, offset_y], dim=2)  # [k,k,2,H,W]
        offset = rotated_offset - kernel_offset  # get final offset

        offset =  torch.flip(offset, dims=[2])  # [k, k, 2, H, W]

        offset = offset.view(-1, H, W)  # 偶数下标存 offset_h、奇数存 offset_w
        # offset = nn.Parameter(torch.Tensor(offset), requires_grad=False)    #它随着模型保存/加载一起保留，而不是每次重建 相当于推理的时候无须再求一次，那两个预处理之后也可以加一下，大点就大点吧
        offset = nn.Parameter(offset.clone().detach(), requires_grad=False)

        return offset


    def _deform_conv2d(self, x):

        if self.rotaed_offset is None or self.rotaed_offset.shape[2:] != x.shape[-2:] or self.rotaed_offset.shape[0] != x.shape[0]:  #不能cache
            # self.rotaed_offset = get_omni_offset(x.shape[-2:], kernel_size=self.kernel_size[0]).to(x.device)[None]
            # self.rotaed_offset = self.rotaed_offset.repeat_interleave(x.shape[0], 0)
            # self.mapping_idx = self.get_mapping_idx(self.rotaed_offset, self.in_channels, self.kernel_size[0], x.shape[0], x.shape[-2], x.shape[-1], x.shape[-2], x.shape[-1])


            self.valid_index, dilation_map, self.deform_mask = self.get_valid_kernel_idx(self.in_channels, self.kernel_size[0], x.shape[0], x.shape[-2], x.shape[-1])
            self.rotaed_offset = self.get_offset_dilation_kernel(self.in_channels, self.kernel_size[0], x.shape[0],  x.shape[-2], x.shape[-1], x.shape[-2], x.shape[-1], dilation_map).to(x.device)[None]
            self.rotaed_offset = self.rotaed_offset.repeat_interleave(x.shape[0], 0)


        if self.rotaed_offset.device != x.device:
            self.rotaed_offset = self.rotaed_offset.to(x.device)

        if self.valid_index.device != x.device:
            self.valid_index = self.valid_index.to(x.device)

        if self.deform_mask.device != x.device:
            self.deform_mask = self.deform_mask.to(x.device)

        # if self.dilation_map.device != x.device:
        #     self.dilation_map = self.dilation_map.to(x.device)

        # if self.mapping_idx.device != x.device:
        #     self.mapping_idx = self.mapping_idx.to(x.device)

        # offset = self.rotaed_offset  #防止重复转变类型，直接使用self.rotaed_offset吧
        mask_pred = None

        target_dtype = x.dtype

        # if self.weight.dtype != target_dtype:
        #     self.weight = self.weight.to(target_dtype)

        if self.rotaed_offset.dtype != target_dtype:
            self.rotaed_offset = self.rotaed_offset.to(target_dtype)
        if self.deform_mask.dtype != target_dtype:
            self.deform_mask = self.deform_mask.to(target_dtype)
        try:
            y = deform_conv2d(x,
                          offset=self.rotaed_offset,  # 每像素卷积核偏置
                          weight=self.weight,
                          mapping_idx = self.valid_index,
                          # dilation_map = self.dilation_map,
                          # mapping_idx = self.mapping_idx,
                          # mapping_wgt = self.mapping_wgt,
                          bias=None,
                          stride=self.stride,
                          padding=self.padding,
                          # mask=self.deform_mask, )
                          mask=None, )

        except Exception as e:
            print("执行以下代码时报错：")
            print("y = torchvision.ops.deform_conv2d(input, offset=offset, weight=self.weight, bias=None, stride=self.stride, padding=self.padding, mask=mask_pred,)")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {e}")
            print(x.dtype, self.weight.dtype, self.rotaed_offset.dtype, mask_pred.dtype if mask_pred is not None else None, self.padding, self.stride)
            print('input:', x.shape)
            print('offset:', self.rotaed_offset.shape)
            print('weight:', self.weight.shape)
            print(x.device , self.rotaed_offset.device , self.weight.device)

            raise  # 继续抛出错误，方便调试

        y.to(x.dtype)

        if self.bias is not None:
            bias = self.bias.view(1, self.out_channels, 1, 1)
            y = y + bias

        if self.stride_in==2:  ###modified now###
            y = self.downsample(y)

        if y.dtype != x.dtype:
            y = y.to(x.dtype)
        return y

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self._deform_conv2d(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(self._deform_conv2d(x)))

        # mapping_idx = torch.full((in_ch * kernel_size * kernel_size * batch_sz * out_h * out_w, ), 5, dtype=torch.int32, device='cuda')
        #[out_ch, out_b, out_h, out_w] 原来的index顺序  #随着网络层数加深，这个数值不断/2 因为in_ch*2 但是out_h/2 且 out_w/2 所以后面的层可以全加，速度不会差多少
        # 我需要的 index 顺序是 [in_ch, kernel_h, kernel_w, batch_sz, out_h, out_w]
        # kernel_num = in_ch * batch_sz * out_h * out_w
        #
        # for index in range(kernel_num):
        #     out_x = index % out_w
        #     out_y = (index // out_w) % out_h
        #     out_b = (index // (out_w * out_h)) % batch_sz
        #     in_c = index // (out_w * out_h * batch_sz)
        #     out_c = in_c * kernel_size * kernel_size
        #
        #     columns_ptr = (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) + out_y * out_w + out_x)
        #     input_ptr = (out_b * (in_ch * in_h * in_w) + in_c * (in_h * in_w))
        #     offset_ptr = (out_b * 1 + 0) * 2 * kernel_size * kernel_size * out_h * out_w
        #
        #     for k_i in range(kernel_size):
        #         for k_j in range(kernel_size):
        #             # print(f"out_c: {out_c}, columns_ptr: {columns_ptr}, input_ptr: {input_ptr}, offset_ptr: {offset_ptr}")
        #             mask_idx = k_i * kernel_size + k_j
        #             offset_idx = 2 * mask_idx
        #             # offset_h = offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x]
        #             offset_h = offset[out_b][offset_idx][out_y][out_x]
        #             offset_w = offset[out_b][offset_idx + 1][out_y][out_x]
        #             input_y = (out_y * stride - padding) + k_i * dilation + offset_h
        #             input_x = (out_x * stride - padding) + k_j * dilation + offset_w
        #
        #             int_input_x = int(input_x)
        #             int_input_y = int(input_y)
        #
        #             if int_input_x < 0: int_input_x = 0
        #             if int_input_x >= in_w: int_input_x = in_w - 1
        #             if int_input_y < 0: int_input_y = 0
        #             if int_input_y >= in_h: int_input_y = in_h - 1
        #
        #             input_index = input_ptr + int_input_y * in_w + int_input_x
        #
        #             mapping_idx[columns_ptr] = input_index
        #
        #             columns_ptr += batch_sz * out_h * out_w


    # def get_mapping_idx(self, offset, in_ch, kernel_size, batch_sz, out_h, out_w, in_h, in_w, stride = 1, padding = 1,  dilation = 1):
    #     """Generate mapping index for deformable convolution."""
    #
    #     # 参数
    #     b, c, in_h, in_w = batch_sz, in_ch, in_h, in_w
    #     # _, _, out_h, out_w = offset.shape
    #     k = kernel_size
    #     # stride, padding, dilation = self.stride, self.padding, self.dilation  # 假设你有这些值
    #
    #     # 1. 基础的 output feature grid 坐标
    #     grid_y, grid_x = torch.meshgrid(torch.arange(out_h), torch.arange(out_w), indexing='ij')
    #     grid_y = grid_y.to('cpu')
    #     grid_x = grid_x.to('cpu')
    #
    #     # 2. 展开为 [1, 1, out_h, out_w] 并乘 stride、加 padding
    #     base_y = grid_y * stride - padding
    #     base_x = grid_x * stride - padding
    #     base_y = base_y.view(1, 1, out_h, out_w).expand(b, k * k, out_h, out_w)
    #     base_x = base_x.view(1, 1, out_h, out_w).expand(b, k * k, out_h, out_w)
    #
    #     # # 3. 获取 offset
    #
    #     # 加入 kernel center 偏移
    #     y, x = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size), indexing='ij')  # [k, k]
    #     y = y * dilation  # dilation
    #     x = x * dilation
    #
    #     kernel_offset = torch.stack([x, y], dim=-1).to('cpu')  # [k, k, 2]
    #     kernel_offset = kernel_offset.view(-1, 2)  # [k*k, 2]
    #     dy = kernel_offset[:, 1].view(1, -1, 1, 1)  # [1, k*k, 1, 1]
    #     dx = kernel_offset[:, 0].view(1, -1, 1, 1)
    #
    #     # 加入自定义偏移，正确写法（交叉拆分）
    #     offset_y = self.rotaed_offset[:, 0::2, :, :].to('cpu')  # 偶数下标 → Y  self.rotaed_offset.shape=[b,2*k*k,out_h,out_w]  为了计算对应的offset
    #     offset_x = self.rotaed_offset[:, 1::2, :, :].to('cpu')  # 奇数下标 → X
    #
    #     # 4. 加上偏移
    #     sample_y = base_y + dy + offset_y
    #     sample_x = base_x + dx + offset_x
    #
    #     # 6. 转为整数索引（nearest neighbor）  向0取整，这样很不好！！！应该就近取整，再想想
    #     sample_y_int = torch.round(sample_y)  # [b,k*k,out_h,out_w]  在input：[b,in_ch,in_h,in_w] 对应的最后两维的offset已计算好
    #     sample_x_int = torch.round(sample_x)
    #
    #     # 5. clamp 到合法图像区域（避免超出）
    #     sample_y_int = sample_y_int.clamp(0, in_h - 1).to(torch.int32)
    #     sample_x_int = sample_x_int.clamp(0, in_w - 1).to(torch.int32)
    #
    #     # 7. 映射为 input 中的扁平索引（[b, c, in_h, in_w] => flatten index）
    #     # 每个 index = b * c * H * W + c * H * W + y * W + x
    #     # 为此，我们需要 meshgrid 扩展 batch/c/channel
    #     sample_b = torch.arange(b, device='cpu').view(b, 1, 1, 1).expand(b, k * k, out_h, out_w)
    #     sample_c = torch.arange(c, device='cpu').view(1, c, 1, 1, 1)
    #
    #     # input index = b * c * in_h * in_w + c * in_h * in_w + y * in_w + x
    #     # 这里我们构建为 [b, c, k*k, out_h, out_w]
    #     sample_y_int = sample_y_int.unsqueeze(1).expand(b, c, k * k, out_h, out_w)
    #     sample_x_int = sample_x_int.unsqueeze(1).expand(b, c, k * k, out_h, out_w)
    #     sample_b = sample_b.unsqueeze(1).expand(b, c, k * k, out_h, out_w)
    #
    #     mapping_idx = (
    #             sample_b * (c * in_h * in_w) +
    #             torch.arange(c, device='cpu').view(1, c, 1, 1, 1) * (in_h * in_w) +
    #             sample_y_int * in_w +
    #             sample_x_int
    #     )  # shape = [b, in_ch, k*k, out_h, out_w]  在input：[b,in_ch,in_h,in_w] 中对应的全局索引已计算好，但是维度顺序不对
    #
    #     mapping_idx = mapping_idx.view(b, c, k, k, out_h, out_w)  # 假设 k*k 可 reshape 成 (k,k)
    #     mapping_idx = mapping_idx.permute(1, 2, 3, 0, 4, 5).contiguous()  # [c, k, k, b, out_h, out_w]
    #     mapping_idx = mapping_idx.view(-1).to(torch.int32).to('cuda').detach()
    #
    #     return mapping_idx



class FW_Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, 1, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.stride_in=s
        if s==2:    ###modified now###
            self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        y = self.conv(x)
        if self.stride_in==2:  ###modified now###
            y = self.downsample(y)
        return self.act(self.bn(y))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
