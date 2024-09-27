import torch 
from torch import nn
import torchvision
import os 
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)




class ConvBNSiLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ObjBottleneck(nn.Module):
    """
    Parameters:
        in_channels (int): number of channel of the input tensor
        out_channels (int): number of channel of the output tensor
        width_multiple (float): it controls the number of channels (and weights)
                                of all the convolutions beside the
                                first and last one. If closer to 0,
                                the simpler the modelIf closer to 1,
                                the model becomes more complex
    """

    def __init__(self, in_channels, out_channels, width_multiple=1):
        super(ObjBottleneck, self).__init__()
        c_ = int(width_multiple*in_channels)
        self.c1 = ConvBNSiLU(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c2 = ConvBNSiLU(c_, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.c2(self.c1(x)) + x

class GlobalChannelAttention(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        assert (kernel_size % 2 == 1), "Kernel size must be odd"

        padding = (kernel_size - 1) // 2
        self.conv_q = nn.Conv1d(1, 1, kernel_size, padding=padding)
        self.conv_k = nn.Conv1d(1, 1, kernel_size, padding=padding)
        self.GAP = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        N, C, H, W = x.shape

        avg_pool = self.GAP(x).view(N, 1, C)
        query = self.conv_q(avg_pool).sigmoid()
        key = self.conv_k(avg_pool).sigmoid().transpose(1, 2)
        query_key = torch.bmm(key, query).softmax(-1).view(N, C, C)

        value = x.view(N, C, -1)
        att = torch.bmm(query_key, value).view(N, C, H, W)
        return x * att

class GlobalSpatialAttention(nn.Module):
    def __init__(self, in_channels, num_reduced_channels):
        super().__init__()
        self.conv1x1_q = nn.Conv2d(in_channels, num_reduced_channels, 1)
        self.conv1x1_k = nn.Conv2d(in_channels, num_reduced_channels, 1)
        self.conv1x1_v = nn.Conv2d(in_channels, num_reduced_channels, 1)
        self.conv1x1_att = nn.Conv2d(num_reduced_channels, in_channels, 1)

    def forward(self, feature_maps):
        query = self.conv1x1_q(feature_maps)
        N, C, H, W = query.shape
        query = query.view(N, C, -1)
        
        key = self.conv1x1_k(feature_maps).view(N, C, -1)
        query_key = torch.bmm(key.permute(0, 2, 1), query).softmax(dim=-1).view(N, H * W, H * W)
        
        value = self.conv1x1_v(feature_maps).view(N, C, -1)
        att = torch.bmm(value, query_key).view(N, C, H, W)
        att = self.conv1x1_att(att)
        
        return att


class GlobalAttention(nn.Module):
    def __init__(self, in_channels, num_reduced_channels, kernel_size):
        super().__init__()
        self.global_channel_attention = GlobalChannelAttention(kernel_size)
        self.global_spatial_attention = GlobalSpatialAttention(in_channels, num_reduced_channels)

    def forward(self, x):
        global_channel_output = self.global_channel_attention(x)
        global_attention_output = self.global_spatial_attention(x)
        return global_attention_output * global_channel_output + global_channel_output
class C2GA(nn.Module):
    def __init__(self, in_channels, out_channels,feature_map_size, width_multiple=1, depth=1, backbone=True):
        super(C2GA, self).__init__()
        c_ = int(width_multiple*in_channels)

        self.c1 = ConvBNSiLU(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.globalatt = GlobalAttention(in_channels,in_channels//4,kernel_size=5)

        if backbone:
            self.seq = nn.Sequential(
                *[ObjBottleneck(c_, c_, width_multiple=1) for _ in range(depth)]
            )
        else:
            self.seq = nn.Sequential(
                *[nn.Sequential(
                    ConvBNSiLU(c_, c_, 1, 1, 0),
                    ConvBNSiLU(c_, c_, 3, 1, 1)
                ) for _ in range(depth)]
            )
        self.c_out = ConvBNSiLU(c_ * 2, out_channels,
                         kernel_size=1, stride=1, padding=0)
                         

    def forward(self, x):
        x = torch.cat([self.seq(self.c1(x)), self.globalatt(x)], dim=1)
        return self.c_out(x)

class C3(nn.Module):
    """
    Parameters:
        in_channels (int): number of channel of the input tensor
        out_channels (int): number of channel of the output tensor
        width_multiple (float): it controls the number of channels (and weights)
                                of all the convolutions beside the
                                first and last one. If closer to 0,
                                the simpler the modelIf closer to 1,
                                the model becomes more complex
        depth (int): it controls the number of times the bottleneck (residual block)
                        is repeated within the C3 block
        backbone (bool): if True, self.seq will be composed by bottlenecks 1, if False
                            it will be composed by bottlenecks 2 (check in the image linked below)
    """

    def __init__(self, in_channels, out_channels, width_multiple=1, depth=1, backbone=True):
        super(C3, self).__init__()
        c_ = int(width_multiple*in_channels)

        self.c1 = ConvBNSiLU(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c_skipped = ConvBNSiLU(
            in_channels,  c_, kernel_size=1, stride=1, padding=0)
        if backbone:
            self.seq = nn.Sequential(
                *[ObjBottleneck(c_, c_, width_multiple=1) for _ in range(depth)]
            )
        else:
            self.seq = nn.Sequential(
                *[nn.Sequential(
                    ConvBNSiLU(c_, c_, 1, 1, 0),
                    ConvBNSiLU(c_, c_, 3, 1, 1)
                ) for _ in range(depth)]
            )
        self.c_out = ConvBNSiLU(c_ * 2, out_channels,
                         kernel_size=1, stride=1, padding=0)
                         
        
    def forward(self, x):
        x = torch.cat([self.seq(self.c1(x)), self.c_skipped(x)], dim=1)
        return self.c_out(x)


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

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """

        super().__init__()
        self.c = int(out_channels * e)  # hidden channels
        self.cv1 = Conv(in_channels, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, out_channels, 1)  # optional act=FReLU(c2)
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

class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
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

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False):
        super(DeformableConv2d, self).__init__()

        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size * kernel_size,
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)
    
    def forward(self, x):

        h, w = x.shape[2:]
        max_offset = max(h, w)/4.

        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator
                                          )
        return x

