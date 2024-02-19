import torch 
from torch import nn
import numpy as np

import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from seg_utils.Parameters import Parameters
from Models.utils import *

#######################################################

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


#######################################################


class ConvBNReLU(nn.Sequential):
    """
    A sequential module consisting of a convolutional layer, batch normalization, and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        n_filters (int): Number of filters in the convolutional layer.
        k_size (int): Size of the convolutional kernel.
        padding (int): Padding added to the input during convolution.
        stride (int): Stride used for the convolution operation.
        bias (bool, optional): Whether to include bias in the convolutional layer. Default is True.
    """
    def __init__(self, in_channels, n_filters, k_size, padding, stride, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, n_filters, k_size, padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True),
        )

class DownsampleConv(nn.Sequential):
    """
    A sequential module for downsampling consisting of ConvBNReLU blocks.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downsample_ratio (int, optional): Ratio by which to downsample. Default is 2.
    """
    def __init__(self, in_channels, out_channels, downsample_ratio=2):
        super().__init__()
        ratio_log2 = int(torch.log2(torch.tensor(downsample_ratio)).item())
        for i in range(ratio_log2):
            temp_channels = in_channels if i == 0 else out_channels
            self.add_module(f'Conv2d_{i}', ConvBNReLU(in_channels=temp_channels, n_filters=out_channels, k_size=3, padding=1, stride=2))
        self.add_module(f'ConvBNReLU_{0}', ConvBNReLU(in_channels=out_channels, n_filters=out_channels, k_size=1, padding=0, stride=1))

class BilinearUpsampleConv1x1(nn.Sequential):
    """
    A sequential module for bilinear upsampling followed by ConvBNReLU.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        upsample_ratio (int, optional): Ratio by which to upsample. Default is 2.
    """
    def __init__(self, in_channels, out_channels, upsample_ratio=2):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, padding=0, stride=1)
        )
        # for i in range(int(torch.log2(torch.tensor(upsample_ratio)).item())):
        self.add_module(f'Upsample_{0}', nn.Upsample(scale_factor=upsample_ratio, mode='bilinear', align_corners=True))
        self.add_module(f'BatchNorm_{0}', nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        self.add_module(f'ReLU_{0}', nn.ReLU(inplace=False))

class UpsampleConvTranspose(nn.Sequential):
    """
    A sequential module for transpose convolutional upsampling followed by ConvBNReLU.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        upsample_ratio (int, optional): Ratio by which to upsample. Default is 2.
    """
    def __init__(self, in_channels, out_channels, upsample_ratio=2):
        super().__init__()
        ratio_log2 = int(torch.log2(torch.tensor(upsample_ratio)).item())
        for i in range(ratio_log2):
            if i == 0 :
                self.add_module(f'ConvTranspose2d_{i}', nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, 1))
            else:
                self.add_module(f'ConvTranspose2d_{i}', nn.ConvTranspose2d(out_channels, out_channels, 3, 2, 1, 1))

        self.add_module(f'Conv1x1_{0}', ConvBNReLU(in_channels=out_channels, n_filters=out_channels, k_size=1, padding=0, stride=1))

class MixedUpsample(nn.Module):
    """
    A module that performs mixed upsampling using both bilinear upsampling and transpose convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        upsample_ratio (int, optional): Ratio by which to upsample. Default is 2.
    """
    def __init__(self, in_channels, out_channels, upsample_ratio=2):
        super(MixedUpsample, self).__init__()
        # self.upsamplelinear = BilinearUpsampleConv1x1(in_channels, out_channels, upsample_ratio)
        self.upsampleconv = UpsampleConvTranspose(in_channels, out_channels, upsample_ratio)

    def forward(self, x):
        """
        Forward pass of the mixed upsampling module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        # return torch.add(self.upsampleconv(x) , self.upsamplelinear(x))
        return self.upsampleconv(x)
class Output(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Output, self).__init__(

        ConvBNReLU(in_channels, in_channels//4, 3, 1, 1),
        ConvBNReLU(in_channels//4, in_channels//8, 3, 1, 1),
        nn.Conv2d(in_channels//8, out_channels, kernel_size=1, padding=0, stride=1, bias=False),
        )

class LaneSegHead(nn.Module):
    def __init__(self, out_channels_list ,w,r,d ):
        super(LaneSegHead,self).__init__()
        
        self.MixUpsample_1 = MixedUpsample(in_channels = out_channels_list[2]//w , out_channels = out_channels_list[2]//w)
        self.c2f_1 = C2f(in_channels = int(1.5*out_channels_list[2]//w ), out_channels= int(1.5*out_channels_list[2]//w ))
        self.MixUpsample_2 = MixedUpsample(in_channels = int(1.5*out_channels_list[2]//w ) , out_channels = int(1.5*out_channels_list[2]//w ))
        self.c2f_2 = C2f(in_channels =int(1.5*out_channels_list[2]//w ) , out_channels= out_channels_list[2]//w)
        self.MixUpsample_3 = MixedUpsample(in_channels = out_channels_list[2]//w , out_channels = out_channels_list[2]//w)
        self.out =   Output(out_channels_list[2]//w , 1 )


    def forward(self,Quarter,Octant):
        x = self.c2f_1(torch.cat((self.MixUpsample_1(Octant),Quarter),1))
        x = self.c2f_2(self.MixUpsample_2(x))

        return self.out(x)

class Area_FE_Head(nn.Module):
    def __init__(self, out_channels_list, w,r,d , out_fe):
        super(Area_FE_Head,self).__init__()

        self.MixUpsample_1 = MixedUpsample(in_channels = out_channels_list[2]//w , out_channels = out_channels_list[2]//w)
        self.c2f_1_d = C2f(in_channels = out_channels_list[2]//w , out_channels= out_channels_list[2]//w)
        self.MixUpsample_2 = MixedUpsample(in_channels = out_channels_list[2]//w , out_channels = out_channels_list[2]//w)
        self.c2f_2_d = C2f(in_channels = out_channels_list[2]//w , out_channels= out_channels_list[2]//w)
        self.MixUpsample_3 = MixedUpsample(in_channels = out_channels_list[2]//w , out_channels = out_channels_list[2]//w)

        self.c2f_1_f = C2f(in_channels =3*out_channels_list[2]//w , out_channels= 3*out_channels_list[2]//w)
        # self.c2f_2_f = C2f(in_channels = 2*out_channels_list[2]//w , out_channels= 2*out_channels_list[2]//w)
        # self.c2f_3_f = C2f(in_channels = 3*out_channels_list[2]//w , out_channels= 3*out_channels_list[2]//w)

        self.DownsampleConv_1 = DownsampleConv(in_channels = out_channels_list[2]//w , out_channels= 3*out_channels_list[2]//w,downsample_ratio=2)
        # self.DownsampleConv_2 = DownsampleConv(in_channels = out_channels_list[2]//w , out_channels= out_channels_list[2]//w,downsample_ratio=4)
        # self.DownsampleConv_3 = DownsampleConv(in_channels = 3*out_channels_list[2]//w , out_channels= 3*out_channels_list[2]//w,downsample_ratio=2)

        self.out_d =   Output(out_channels_list[2]//w , 1 )
        self.out_f =   Output(3*out_channels_list[2]//w, out_fe)

    
    def forward(self,Octant):
        out_d = self.c2f_1_d(self.MixUpsample_1(Octant))
        out_f = self.c2f_1_f(self.DownsampleConv_1(out_d))
        out_d = self.c2f_2_d(self.MixUpsample_2(out_d))
        # out_f = self.out_f(self.DownsampleConv_3(self.c2f_3_f(torch.cat((out_f,self.DownsampleConv_2(out_d)),1))))
        out_d = self.out_d(out_d)
        out_f = self.out_f(out_f)

        return [out_d,out_f]


class SegHead(nn.Module):
    def __init__(self, out_channels_list ,w,r,d , class_number):
        super(SegHead,self).__init__()

        self.p = Parameters()

        self.lane = LaneSegHead(out_channels_list ,w,r,d )
        self.area_fe = Area_FE_Head(out_channels_list, w,r,d , self.p.feature_size)
        
    def forward(self, Half,Quarter,Octant,One_sixteenth):
        area_fe_out = self.area_fe(Octant)
        Out_Confidence = torch.cat((self.lane(Quarter,Octant),area_fe_out[0]),1)

        return [Out_Confidence, area_fe_out[1] ]  

