# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Convolution modules
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# from .block import C2f

__all__ = ('Conv', 'Conv2', 'LightConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus', 'GhostConv',
           'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'RepConv', 'GAM_Attention', 'ChannelAttention2',
           'CPCA', 'DropPath', 'LSKblock', 'EMA', 'MSLGhostAttention', 'FA', 'MSLGhostAttention_v2', 'MSLAP',
           'MPConv', 'MLCA', 'MLCASA', 'Conv_ATT', 'ADown', 'InceptionBottleneck', 'MomentCA', 'EAN', 'Cat')


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv_ATT(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, ATT_TYPE='', p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        if ATT_TYPE == 'CBAM':
            self.ATT = CBAM(c2)
        elif ATT_TYPE == 'CPCA':
            self.ATT = CPCA(c2, c2)
        elif ATT_TYPE == 'GAM':
            self.ATT = GAM_Attention(c2, c2)
        elif ATT_TYPE == 'LSK':
            self.ATT = LSKblock(c2)
        elif ATT_TYPE == 'EMA':
            self.ATT = EMA(c2)
        # elif ATT_TYPE == 'DLKA':
        #     self.ATT = deformable_LKA(c2)
        elif ATT_TYPE == 'MSLAG_s':
            self.ATT = MSLGhostAttention(c2, k=(3, 5, 7, 9))
        elif ATT_TYPE == 'MSLAG_l':
            self.ATT = MSLGhostAttention(c2, k=(5, 7, 9, 11))
        elif ATT_TYPE == 'FA':
            self.ATT = FA(c2)
        elif ATT_TYPE == 'MSLAG_s2':
            self.ATT = MSLGhostAttention_v2(c2)
        elif ATT_TYPE == 'MSLAP':
            self.ATT = MSLAP(c2)
        # elif ATT_TYPE == 'MCA':
        #     self.ATT = MCALayer(c2)
        elif ATT_TYPE == 'MLCA':
            self.ATT = MLCA(c2)
        elif ATT_TYPE == 'MLCASA':
            self.ATT = MLCASA(c2)
        else:
            self.ATT = None

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        if self.ATT is not None:
            return self.ATT(self.act(self.bn(self.conv(x))))
        else:
            return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        if self.ATT is not None:
            return self.ATT(self.act(self.conv(x)))
        else:
            return self.act(self.conv(x))


class Cat(nn.Module):
    def __init__(self, num_in, dimension=1, epsilon=1e-4):
        super(Cat, self).__init__()
        self.num_in = num_in
        self.d = dimension
        self.epsilon = epsilon
        if self.num_in == 2:
            self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        else:
            self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.w_relu = nn.ReLU()

    def forward(self, x):
        if self.num_in == 2:
            low, high = x
        else:
            low, mid, high = x
        wt = self.w_relu(self.w)
        wt1 = wt / (torch.sum(wt, dim=0) + self.epsilon)

        return torch.cat([wt1[0] * low, wt1[1] * high], self.d) if self.num_in == 2 else torch.cat([wt1[0] * low, wt1[1]*mid, wt1[2] * high], self.d)


class MPConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=2):
        super(MPConv, self).__init__()
        self.c = c1 // 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1 = Conv(c1 // 2, c1 // 2, k, s)
        self.conv2 = Conv(c1, c1, k, s=1)
        self.conv = Conv(c2, c2, k=1, s=1)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.c, self.c], 1)
        x1 = self.pool(x1)
        x2 = self.conv1(x2)
        x3 = self.conv2(torch.cat([x1, x2], 1))
        return self.conv(torch.cat([x1, x2, x3], 1))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]:i[0] + 1, i[1]:i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__('cv2')
        self.forward = self.forward_fuse


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LightConv(nn.Module):
    """Light convolution with args(ch_in, ch_out, kernel).
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 3, 1, None, c_, act=act)  # K=5   3

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status. This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process"""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process"""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class GAM_Attention(nn.Module):
    def __init__(self, in_channels, c2, rate=4):
        super(GAM_Attention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2).sigmoid()
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out


class ChannelAttention2(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention2, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x


class CPCA(nn.Module):

    def __init__(self, in_channels, out_channels,
                 channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        self.ca = ChannelAttention2(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)  # 1x1卷积
        inputs = self.act(inputs)  # GELU激活

        channel_att_vec = self.ca(inputs)  # 通道注意力
        inputs = channel_att_vec * inputs  # 通道注意力

        x_init = self.dconv5_5(inputs)  # 5x5 组卷积，depth-wise
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)  # 1x1
        out = spatial_att * inputs
        out = self.conv(out)
        return out


class LSKblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)  # b, , h ,1
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2) # b, , w, 1
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))# b, , h+w, 1
        x_h, x_w = torch.split(hw, [h, w], dim=2)  #
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class GEMA(nn.Module):
    def __init__(self, channels, factor=8, anyatt='CBAM'):
        super(GEMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        # self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        if anyatt == 'CBAM':
            self.att = CBAM(channels)
        else:
            self.att = None


    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x1 = self.gn(self.att(group_x))
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)
 # Generalized

class EAN(nn.Module):
    def __init__(self, c1, g=8, eps: float = 1e-5, device=None, dtype=None):
        super(EAN, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.groups = g
        self.eps = eps
        self.gn = nn.GroupNorm(c1 // self.groups, c1 // self.groups)
        self.alpha = Parameter(torch.empty(1, **factory_kwargs))
        self.sigma = Parameter(torch.empty(1, **factory_kwargs))
        nn.init.ones_(self.alpha)
        nn.init.zeros_(self.sigma)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        gn = self.gn(group_x)
        # print(self.gn.weight.size())
        omega = self.gn.weight / torch.sqrt(torch.sum(self.gn.weight ** 2 + self.eps))
        att = (self.alpha * omega * gn.permute(0, 2, 3, 1) + self.sigma).permute(0, 3, 1, 2).sigmoid()
        # print(self.alpha.dtype)
        # print(omega.shape)
        # print(gn.shape)
        # gn = omega*gn.permute(0,2,3,1)
        # gn = self.alpha*gn + self.sigma
        # att = gn.sigmoid()
        x = group_x * att
        return x.reshape(b, c, h, w)


class MomentCA(nn.Module):
    def __init__(self, inc, strategy='E', device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.strategy = strategy
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(1, 1, (3, 2), 1, (1, 0), bias=True)
        self.act = nn.Sigmoid()
        self.alpha = Parameter(torch.empty(1, **factory_kwargs))
        self.sigma = Parameter(torch.empty(1, **factory_kwargs))
        nn.init.ones_(self.alpha)
        nn.init.zeros_(self.sigma)

    def forward(self, x):
        b, c, h, w = x.size()
        M1 = self.pool(x)  # [b, c, 1, 1]
        if self.strategy == 'E':
            M2 = self.pool((x - M1) ** 2)
        else:
            M2 = self.pool((x - M1) ** 3)
        M = torch.cat([M1.permute(0, 2, 1, 3) * self.alpha, M2.permute(0, 2, 1, 3) * self.sigma], 1)  # b,2,c,1
        att = self.fc(M.permute(0, 3, 2, 1)).permute(0, 2, 1, 3).sigmoid()
        return x * att


class MSLGhostAttention(nn.Module):
    # Multi-Scale lightweight AM based on GhostConv
    def __init__(self, cin, k=(3, 5, 7, 9)):  # (5,7,9,11)
        super(MSLGhostAttention, self).__init__()
        assert cin % 4 == 0, 'GhostAttention: input_channel must divided by 4!'
        self.c = cin
        self.c2 = int(cin / 2)
        self.cp = int(self.c2 / 4)
        self.conv1 = nn.Conv2d(self.c, self.c2, 1)
        self.conv_3 = nn.Conv2d(self.cp, self.cp, kernel_size=k[0], padding=(k[0] - 1) // 2, groups=self.cp)
        self.conv_5 = nn.Conv2d(self.cp, self.cp, kernel_size=k[1], padding=(k[1] - 1) // 2, groups=self.cp)
        self.conv_7 = nn.Conv2d(self.cp, self.cp, kernel_size=k[2], padding=(k[2] - 1) // 2, groups=self.cp)
        self.conv_9 = nn.Conv2d(self.cp, self.cp, kernel_size=k[3], padding=(k[3] - 1) // 2, groups=self.cp)
        self.conv2 = nn.Conv2d(self.c, self.c, 1)

    def forward(self, x):
        x0 = self.conv1(x)
        x1, x2, x3, x4 = x0.split((self.cp, self.cp, self.cp, self.cp), 1)
        x1 = self.conv_3(x1)
        x2 = self.conv_5(x2)
        x3 = self.conv_7(x3)
        x4 = self.conv_9(x4)
        attn = self.conv2(torch.cat([x0, x1, x2, x3, x4], 1)).sigmoid()

        return x * attn


class MSLGhostAttention_v2(nn.Module):
    # Multi-Scale lightweight AM based on GhostConv
    def __init__(self, cin, k=(3, 5, 7, 9)):  # (5,7,9,11)
        super(MSLGhostAttention_v2, self).__init__()
        assert cin % 4 == 0, 'GhostAttention: input_channel must divided by 4!'
        self.c = cin
        self.c2 = int(cin / 2)
        self.cp = int(self.c2 / 4)
        self.conv1 = nn.Conv2d(self.c, self.c2, 1)
        self.conv_3 = nn.Sequential(
            nn.Conv2d(self.cp, self.cp, kernel_size=(1, k[0]), padding=(0, (k[0] - 1) // 2), groups=self.cp),
            nn.Conv2d(self.cp, self.cp, kernel_size=(k[0], 1), padding=((k[0] - 1) // 2, 0), groups=self.cp)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(self.cp, self.cp, kernel_size=(1, k[1]), padding=(0, (k[1] - 1) // 2), groups=self.cp),
            nn.Conv2d(self.cp, self.cp, kernel_size=(k[1], 1), padding=((k[1] - 1) // 2, 0), groups=self.cp)
        )
        self.conv_7 = nn.Sequential(
            nn.Conv2d(self.cp, self.cp, kernel_size=(1, k[2]), padding=(0, (k[2] - 1) // 2), groups=self.cp),
            nn.Conv2d(self.cp, self.cp, kernel_size=(k[2], 1), padding=((k[2] - 1) // 2, 0), groups=self.cp)
        )
        self.conv_9 = nn.Sequential(
            nn.Conv2d(self.cp, self.cp, kernel_size=(1, k[3]), padding=(0, (k[3] - 1) // 2), groups=self.cp),
            nn.Conv2d(self.cp, self.cp, kernel_size=(k[3], 1), padding=((k[3] - 1) // 2, 0), groups=self.cp)
        )
        # self.conv_3 = nn.Conv2d(self.cp, self.cp, kernel_size=k[0], padding=(k[0] - 1) // 2, groups=self.cp)
        # self.conv_5 = nn.Conv2d(self.cp, self.cp, kernel_size=k[1], padding=(k[1] - 1) // 2, groups=self.cp)
        # self.conv_7 = nn.Conv2d(self.cp, self.cp, kernel_size=k[2], padding=(k[2] - 1) // 2, groups=self.cp)
        # self.conv_9 = nn.Conv2d(self.cp, self.cp, kernel_size=k[3], padding=(k[3] - 1) // 2, groups=self.cp)
        self.conv2 = nn.Conv2d(self.c, self.c, 1)

    def forward(self, x):
        x0 = self.conv1(x)
        x1, x2, x3, x4 = x0.split((self.cp, self.cp, self.cp, self.cp), 1)
        x1 = self.conv_3(x1)
        x2 = self.conv_5(x2)
        x3 = self.conv_7(x3)
        x4 = self.conv_9(x4)
        attn = self.conv2(torch.cat([x0, x1, x2, x3, x4], 1)).sigmoid()

        return x * attn


class FA(nn.Module):
    def __init__(self, cin):
        super(FA, self).__init__()
        self.focus = Focus(cin, cin)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(cin, cin, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x1 = self.focus(x)
        return x * self.act(self.fc(self.pool(x1)))


# class MSLAP(nn.Module):
#     def __init__(self, cin, k=(3, 5, 7, 9), div_n=0):
#         super(MSLAP, self).__init__()
#         self.div_n = div_n
#         self.c = cin
#         self.cp = int(cin / 4)
#         self.conv_3 = nn.Conv2d(self.cp, self.cp, kernel_size=k[0], padding=(k[0] - 1) // 2, groups=self.cp)
#         self.conv_5 = nn.Conv2d(self.cp, self.cp, kernel_size=k[1], padding=(k[1] - 1) // 2, groups=self.cp)
#         self.conv_7 = nn.Conv2d(self.cp, self.cp, kernel_size=k[2], padding=(k[2] - 1) // 2, groups=self.cp)
#         self.conv_9 = nn.Conv2d(self.cp, self.cp, kernel_size=k[3], padding=(k[3] - 1) // 2, groups=self.cp)
#         # self.conv_3 = nn.Sequential(
#         #     nn.Conv2d(self.cp, self.cp, kernel_size=(1, k[0]), padding=(0, (k[0] - 1) // 2), groups=self.cp),
#         #     nn.Conv2d(self.cp, self.cp, kernel_size=(k[0], 1), padding=((k[0] - 1) // 2, 0), groups=self.cp)
#         # )
#         # self.conv_5 = nn.Sequential(
#         #     nn.Conv2d(self.cp, self.cp, kernel_size=(1, k[1]), padding=(0, (k[1] - 1) // 2), groups=self.cp),
#         #     nn.Conv2d(self.cp, self.cp, kernel_size=(k[1], 1), padding=((k[1] - 1) // 2, 0), groups=self.cp)
#         # )
#         # self.conv_7 = nn.Sequential(
#         #     nn.Conv2d(self.cp, self.cp, kernel_size=(1, k[2]), padding=(0, (k[2] - 1) // 2), groups=self.cp),
#         #     nn.Conv2d(self.cp, self.cp, kernel_size=(k[2], 1), padding=((k[2] - 1) // 2, 0), groups=self.cp)
#         # )
#         # self.conv_9 = nn.Sequential(
#         #     nn.Conv2d(self.cp, self.cp, kernel_size=(1, k[3]), padding=(0, (k[3] - 1) // 2), groups=self.cp),
#         #     nn.Conv2d(self.cp, self.cp, kernel_size=(k[3], 1), padding=((k[3] - 1) // 2, 0), groups=self.cp)
#         # )
#         self.conv2 = nn.Conv2d(self.c, self.c, 1)
#     def forward(self, x):
#         x1, x2, x3, x4 = x.split((self.cp, self.cp, self.cp, self.cp), 1)
#         x1 = self.conv_3(x1)
#         x2 = self.conv_5(x2)
#         x3 = self.conv_7(x3)
#         x4 = self.conv_9(x4)
#         attn = self.conv2(torch.cat([x1, x2, x3, x4], 1)).sigmoid()
#         return x*attn


class MSLAP(nn.Module):
    def __init__(self, cin, k=(3, 5, 7, 9)):
        super(MSLAP, self).__init__()
        self.cp = int(cin / 8)
        self.conv_3 = nn.Conv2d(self.cp, self.cp, kernel_size=k[0], padding=(k[0] - 1) // 2, groups=self.cp)
        self.conv_5 = nn.Conv2d(self.cp, self.cp, kernel_size=k[1], padding=(k[1] - 1) // 2, groups=self.cp)
        self.conv_7 = nn.Conv2d(self.cp, self.cp, kernel_size=k[2], padding=(k[2] - 1) // 2, groups=self.cp)
        self.conv_9 = nn.Conv2d(self.cp, self.cp, kernel_size=k[3], padding=(k[3] - 1) // 2, groups=self.cp)
        # self.conv_3 = nn.Sequential(
        #     nn.Conv2d(self.cp, self.cp, kernel_size=(1, k[0]), padding=(0, (k[0] - 1) // 2), groups=self.cp),
        #     nn.Conv2d(self.cp, self.cp, kernel_size=(k[0], 1), padding=((k[0] - 1) // 2, 0), groups=self.cp)
        # )
        # self.conv_5 = nn.Sequential(
        #     nn.Conv2d(self.cp, self.cp, kernel_size=(1, k[1]), padding=(0, (k[1] - 1) // 2), groups=self.cp),
        #     nn.Conv2d(self.cp, self.cp, kernel_size=(k[1], 1), padding=((k[1] - 1) // 2, 0), groups=self.cp)
        # )
        # self.conv_7 = nn.Sequential(
        #     nn.Conv2d(self.cp, self.cp, kernel_size=(1, k[2]), padding=(0, (k[2] - 1) // 2), groups=self.cp),
        #     nn.Conv2d(self.cp, self.cp, kernel_size=(k[2], 1), padding=((k[2] - 1) // 2, 0), groups=self.cp)
        # )
        # self.conv_9 = nn.Sequential(
        #     nn.Conv2d(self.cp, self.cp, kernel_size=(1, k[3]), padding=(0, (k[3] - 1) // 2), groups=self.cp),
        #     nn.Conv2d(self.cp, self.cp, kernel_size=(k[3], 1), padding=((k[3] - 1) // 2, 0), groups=self.cp)
        # )
        self.conv2 = nn.Conv2d(cin, cin, 1)

    def forward(self, x):
        x1, x2, x3, x4, x5 = x.split((self.cp, self.cp, self.cp, self.cp, 4 * self.cp), 1)
        x1 = self.conv_3(x1)
        x2 = self.conv_5(x2)
        x3 = self.conv_7(x3)
        x4 = self.conv_9(x4)
        attn = self.conv2(torch.cat([x1, x2, x3, x4, x5], 1)).sigmoid()

        return x * attn


class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma=2, b=1, local_weight=0.5):
        super(MLCA, self).__init__()

        # ECA 计算方法
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)  # eca  gamma=2
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_weight = local_weight

        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        local_arv = self.local_arv_pool(x)
        global_arv = self.global_arv_pool(local_arv)

        b, c, m, n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape

        # (b,c,local_size,local_size) -> (b,c,local_size*local_size)-> (b,local_size*local_size,c)-> (b,1,local_size*local_size*c)
        temp_local = local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)

        # (b,c,local_size,local_size) <- (b,c,local_size*local_size)<-(b,local_size*local_size,c) <- (b,1,local_size*local_size*c)
        y_local_transpose = y_local.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(b, c,
                                                                                                            self.local_size,
                                                                                                            self.local_size)
        y_global_transpose = y_global.view(b, -1).transpose(-1, -2).unsqueeze(-1)

        # 反池化
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(), [self.local_size, self.local_size])
        att_all = F.adaptive_avg_pool2d(att_global * (1 - self.local_weight) + (att_local * self.local_weight), [m, n])

        x = x * att_all
        return x


class MLCASA(nn.Module):
    def __init__(self, c):
        super(MLCASA, self).__init__()
        self.mlca = MLCA(c)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.mlca(x))


class CAA(nn.Module):
    """Context Anchor Attention"""

    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11
    ):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(channels, channels, 1, 1, 0
                          )
        self.h_conv = nn.Conv2d(channels, channels, (1, h_kernel_size), 1,
                                (0, h_kernel_size // 2), groups=channels
                                )
        self.v_conv = nn.Conv2d(channels, channels, (v_kernel_size, 1), 1,
                                (v_kernel_size // 2, 0), groups=channels
                                )
        self.conv2 = Conv(channels, channels, 1, 1, 0,
                          )
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor


def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number to the nearest value that can be
    divisible by the divisor. It is taken from the original tf repo. It ensures
    that all layers have a channel number that is divisible by divisor. It can
    be seen here: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py  # noqa

    Args:
        value (int): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel number to
            the original channel number. Default: 0.9.

    Returns:
        int: The modified output channel number.
    """

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


class InceptionBottleneck(nn.Module):
    """Bottleneck with Inception module"""

    def __init__(
            self,
            in_channels: int,
            out_channels=None,
            kernel_sizes=(3, 5, 7, 9, 11),
            dilations=(1, 1, 1, 1, 1),
            expansion: float = 1.0,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11,

    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.pre_conv = Conv(in_channels, hidden_channels, 1, 1, 0, 1,
                             )

        self.dw_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_sizes[0], 1,
                                 autopad(kernel_sizes[0], None, dilations[0]), dilations[0],
                                 groups=hidden_channels)
        self.dw_conv1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_sizes[1], 1,
                                  autopad(kernel_sizes[1], None, dilations[1]), dilations[1],
                                  groups=hidden_channels)
        self.dw_conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_sizes[2], 1,
                                  autopad(kernel_sizes[2], None, dilations[2]), dilations[2],
                                  groups=hidden_channels)
        self.dw_conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_sizes[3], 1,
                                  autopad(kernel_sizes[3], None, dilations[3]), dilations[3],
                                  groups=hidden_channels)
        self.dw_conv4 = nn.Conv2d(hidden_channels, hidden_channels, kernel_sizes[4], 1,
                                  autopad(kernel_sizes[4], None, dilations[4]), dilations[4],
                                  groups=hidden_channels)
        self.pw_conv = Conv(hidden_channels, hidden_channels, 1, 1, 0, 1
                            )

        if with_caa:
            self.caa_factor = CAA(hidden_channels, caa_kernel_size, caa_kernel_size)
        else:
            self.caa_factor = None

        self.add_identity = add_identity and in_channels == out_channels

        self.post_conv = Conv(hidden_channels, out_channels, 1, 1, 0, 1
                              )

    def forward(self, x):
        x = self.pre_conv(x)

        y = x  # if there is an inplace operation of x, use y = x.clone() instead of y = x
        x = self.dw_conv(x)
        x = x + self.dw_conv1(x) + self.dw_conv2(x) + self.dw_conv3(x) + self.dw_conv4(x)
        x = self.pw_conv(x)
        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y

        x = self.post_conv(x)
        return x

# class PKIBlock(nn.Module):
#     """Poly Kernel Inception Block"""
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels= None,
#             kernel_sizes= (3, 5, 7, 9, 11),
#             dilations= (1, 1, 1, 1, 1),
#             with_caa: bool = True,
#             caa_kernel_size: int = 11,
#             expansion: float = 1.0,
#             ffn_scale: float = 4.0,
#             ffn_kernel_size: int = 3,
#             dropout_rate: float = 0.,
#             drop_path_rate: float = 0.,
#             layer_scale= 1.0,
#             add_identity: bool = True,
#
#     ):
#         super().__init__()
#         out_channels = out_channels or in_channels
#         hidden_channels = make_divisible(int(out_channels * expansion), 8)
#
#         if norm_cfg is not None:
#             self.norm1 = build_norm_layer(norm_cfg, in_channels)[1]
#             self.norm2 = build_norm_layer(norm_cfg, hidden_channels)[1]
#         else:
#             self.norm1 = nn.BatchNorm2d(in_channels)
#             self.norm2 = nn.BatchNorm2d(hidden_channels)
#
#         self.block = InceptionBottleneck(in_channels, hidden_channels, kernel_sizes, dilations,
#                                          expansion=1.0, add_identity=True,
#                                          with_caa=with_caa, caa_kernel_size=caa_kernel_size,
#                                          )
#         self.ffn = ConvFFN(hidden_channels, out_channels, ffn_scale, ffn_kernel_size, dropout_rate, add_identity=False,
#                            norm_cfg=None, act_cfg=None)
#         self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
#
#         self.layer_scale = layer_scale
#         if self.layer_scale:
#             self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_channels), requires_grad=True)
#             self.gamma2 = nn.Parameter(layer_scale * torch.ones(out_channels), requires_grad=True)
#         self.add_identity = add_identity and in_channels == out_channels
#
#     def forward(self, x):
#         if self.layer_scale:
#             if self.add_identity:
#                 x = x + self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
#                 x = x + self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
#             else:
#                 x = self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
#                 x = self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
#         else:
#             if self.add_identity:
#                 x = x + self.drop_path(self.block(self.norm1(x)))
#                 x = x + self.drop_path(self.ffn(self.norm2(x)))
#             else:
#                 x = self.drop_path(self.block(self.norm1(x)))
#                 x = self.drop_path(self.ffn(self.norm2(x)))
#         return x
