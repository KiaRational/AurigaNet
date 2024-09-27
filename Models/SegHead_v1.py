import torch 
from torch import nn
import numpy as np

import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from seg_utils.Parameters import Parameters

from Models.utils import *
######################################################

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

#######################################################

class ConvBNSiLU(nn.Sequential):
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
    A sequential module for downsampling consisting of ConvBNSiLU blocks.

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
            self.add_module(f'Conv2d_{i}', ConvBNSiLU(in_channels=temp_channels, n_filters=out_channels, k_size=3, padding=1, stride=2))
        self.add_module(f'ConvBNSiLU_{0}', ConvBNSiLU(in_channels=out_channels, n_filters=out_channels, k_size=1, padding=0, stride=1))

class BilinearUpsampleConv1x1(nn.Sequential):
    """
    A sequential module for bilinear upsampling followed by ConvBNSiLU.

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
        self.add_module(f'SiLU_{0}', nn.SiLU(inplace=False))

class UpsampleConvTranspose(nn.Sequential):
    """
    A sequential module for transpose convolutional upsampling followed by ConvBNSiLU.

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
                self.add_module(f'SiLU_{0}', nn.SiLU(inplace=False))

            else:
                self.add_module(f'ConvTranspose2d_{i}', nn.ConvTranspose2d(out_channels, out_channels, 3, 2, 1, 1))
                self.add_module(f'SiLU_{0}', nn.SiLU(inplace=False))


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

        ConvBNSiLU(in_channels, in_channels//4, 3, 1, 1),
        ConvBNSiLU(in_channels//4, in_channels//8, 3, 1, 1),
        nn.Conv2d(in_channels//8, out_channels, kernel_size=1, padding=0, stride=1, bias=False),
        )

class Output_deform(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Output_deform, self).__init__(

        DeformableConv2d(in_channels, in_channels//4, 3, 1, 1),
        DeformableConv2d(in_channels//4, in_channels//8, 3, 1, 1),
        DeformableConv2d(in_channels//8, out_channels, kernel_size=1, stride=1, padding=0,bias=False),
        )

class LaneSegHead(nn.Module):
    def __init__(self, out_channels_list ,w,r,d ):
        super(LaneSegHead,self).__init__()
        
        self.MixUpsample_1 = MixedUpsample(in_channels = out_channels_list[1]//w , out_channels = out_channels_list[1]//w)
        self.c1 = ConvBNSiLU(out_channels_list[2]//w , out_channels_list[1]//w , 1, 0, 1)
        self.ga = C2GA(out_channels_list[1]//w,out_channels_list[1]//w,feature_map_size=80,depth=3)
        self.out =   Output(out_channels_list[1]//w , 1 )

    def forward(self,Quarter,Octant):
        x = self.MixUpsample_1(self.ga(self.c1(Octant)))
        return self.out(x)

class Area_FE_Head(nn.Module):
    def __init__(self, out_channels_list, w,r,d , out_fe):
        super(Area_FE_Head,self).__init__()
        self.c1 = ConvBNSiLU(out_channels_list[2]//w , out_channels_list[1]//w , 1, 0, 1)
        self.MixUpsample_1 = MixedUpsample(in_channels = out_channels_list[1]//w , out_channels = out_channels_list[1]//w)
        self.ga = C2GA(out_channels_list[1]//w,out_channels_list[1]//w,feature_map_size=80,depth=3)
        self.def_conv_1_f = DeformableConv2d(in_channels =(out_channels_list[1])//w , out_channels= (out_channels_list[1])//w)
        self.DownsampleConv_1 = DownsampleConv(in_channels = (out_channels_list[1])//w , out_channels= (out_channels_list[1])//w,downsample_ratio=2)
        self.def_conv_2_f = DeformableConv2d(in_channels = (out_channels_list[1])//w , out_channels= (out_channels_list[1]//2)//w,kernel_size=3,stride=1,padding=1)
        self.out_d =   Output((out_channels_list[1])//w , 1 )
        self.def_conv_3_1 =   DeformableConv2d((out_channels_list[1]//2)//w,(out_channels_list[1]//2)//w)
        self.out_f =   DeformableConv2d((out_channels_list[1]//2)//w, out_fe)

    def forward(self,Quarter,Octant):

        out_d = self.MixUpsample_1(self.ga(self.c1(Octant)))
        out_f = self.def_conv_2_f(self.def_conv_1_f(self.DownsampleConv_1(out_d)))
        out_d = self.out_d(out_d)
        out_f = self.out_f(self.def_conv_3_1(out_f))
        return [out_d,out_f]

class SegHead(nn.Module):
    def __init__(self, out_channels_list ,w,r,d , class_number):
        super(SegHead,self).__init__()

        self.p = Parameters()

        self.lane = LaneSegHead(out_channels_list ,w,r,d )
        self.area_fe = Area_FE_Head(out_channels_list, w,r,d , self.p.feature_size)

    def forward(self, Half,Quarter,Octant,One_sixteenth):
        area_fe_out = self.area_fe(Quarter,Octant)
        Out_Confidence = torch.cat((area_fe_out[0],self.lane(Quarter,Octant)),1)

        return [Out_Confidence, area_fe_out[1]]  

