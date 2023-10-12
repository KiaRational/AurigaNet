import torch 
from torch import nn
import numpy as np

import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from seg_utils.Parameters import Parameters



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
            nn.ReLU(inplace=True),
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
            self.add_module(f'Conv2d_{i}', ConvBNReLU(in_channels=temp_channels, n_filters=out_channels//2, k_size=3, padding=1, stride=2))
            self.add_module(f'ConvBNReLU_{i}', ConvBNReLU(in_channels=out_channels//2, n_filters=out_channels, k_size=1, padding=0, stride=1))

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
        for i in range(int(torch.log2(torch.tensor(upsample_ratio)).item())):
            self.add_module(f'Upsample_{i}', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            self.add_module(f'BatchNorm_{i}', nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            self.add_module(f'ReLU_{i}', nn.ReLU(inplace=False))

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
            self.add_module(f'ConvTranspose2d_{i}', nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, 1))
            self.add_module(f'Conv1x1_{i}', ConvBNReLU(in_channels=out_channels, n_filters=out_channels, k_size=1, padding=0, stride=1))

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
        self.upsamplelinear = BilinearUpsampleConv1x1(in_channels, out_channels, upsample_ratio)
        self.upsampleconv = UpsampleConvTranspose(in_channels, out_channels, upsample_ratio)

    def forward(self, x):
        """
        Forward pass of the mixed upsampling module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.upsampleconv(x) + self.upsamplelinear(x)

class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBottleneckBlock, self).__init__()
        self.layer1 = ConvBNReLU(in_channels=in_channels, n_filters=out_channels, k_size=1, padding=0, stride=1)
        self.layer2 = ConvBNReLU(in_channels=out_channels, n_filters=out_channels, k_size=kernel_size, padding=padding, stride=stride)
        self.layer3 = ConvBNReLU(in_channels=out_channels, n_filters=out_channels, k_size=1, padding=0, stride=1)
        self.res_conv = None
        if in_channels != out_channels:
            self.res_conv = ConvBNReLU(in_channels=in_channels, n_filters=out_channels, k_size=1, padding=0, stride=1)

    def forward(self, x):
        res = x if self.res_conv is None else self.res_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x + res

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, repeat_num = 3):
        super(BasicBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(repeat_num):
            if i == 0:
                self.layers.append(ResidualBottleneckBlock(in_channels, out_channels, kernel_size, padding, stride))
            else:
                self.layers.append(ResidualBottleneckBlock(out_channels, out_channels, kernel_size, padding, stride))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Output(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Output, self).__init__()
        self.conv1 = ConvBNReLU(in_channels, in_channels//4, 3, 1, 1)
        self.conv2 = ConvBNReLU(in_channels//4, in_channels//8, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels//8, out_channels, kernel_size=1, padding=0, stride=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class SegHead(nn.Module):
    def __init__(self, out_channels_list ,w,r,d , class_number):
        super(SegHead,self).__init__()

        self.p = Parameters()
        # BasicBlockB_S B is number of branch and S is number of stage
        self.BasicBlock1_1 = BasicBlock(in_channels = out_channels_list[1]//w , out_channels = out_channels_list[1]//w)
        self.BasicBlock1_2 = BasicBlock(in_channels = out_channels_list[1]//w , out_channels = out_channels_list[1]//w)
        self.BasicBlock2_1 = BasicBlock(in_channels = out_channels_list[2]//w , out_channels = out_channels_list[2]//w)
        self.BasicBlock2_2 = BasicBlock(in_channels = out_channels_list[2]//w , out_channels = out_channels_list[2]//w)
        self.BasicBlock3_2 = BasicBlock(in_channels = out_channels_list[3]//w , out_channels = out_channels_list[3]//w)
        self.BasicBlock4_3 = BasicBlock(in_channels = out_channels_list[0]//w , out_channels = out_channels_list[0]*r//w,repeat_num=5)
        self.BasicBlock2_3 = BasicBlock(in_channels = out_channels_list[2]//w , out_channels = out_channels_list[2]//w)
        # self.BasicBlock3_3 = BasicBlock(in_channels = out_channels_list[3]//w , out_channels = out_channels_list[3]//(w*r))
        self.Downsample1_1_2 = DownsampleConv(in_channels = out_channels_list[1]//w , out_channels = out_channels_list[2]//w , downsample_ratio=2)
        self.Downsample1_2_2 = DownsampleConv(in_channels = out_channels_list[1]//w , out_channels = out_channels_list[2]//w , downsample_ratio=2)
        self.Downsample1_2_4 = DownsampleConv(in_channels = out_channels_list[1]//w , out_channels = out_channels_list[3]//w , downsample_ratio=4)
        self.Downsample2_2_2 = DownsampleConv(in_channels = out_channels_list[2]//w , out_channels = out_channels_list[3]//w , downsample_ratio=2)
        # self.Downsample1_3_4 = DownsampleConv(in_channels = out_channels_list[1]//w , out_channels = out_channels_list[2]//w , downsample_ratio=4)
        # self.Downsample2_3_2 = DownsampleConv(in_channels = out_channels_list[2]//w , out_channels = out_channels_list[3]//w , downsample_ratio=2)
        self.Upsample2_1_2 = BilinearUpsampleConv1x1(in_channels = out_channels_list[2]//w , out_channels = out_channels_list[1]//w , upsample_ratio=2)
        self.Upsample2_2_2 = BilinearUpsampleConv1x1(in_channels = out_channels_list[2]//w , out_channels = out_channels_list[1]//w , upsample_ratio=2)
        self.Upsample3_2_4 = BilinearUpsampleConv1x1(in_channels = out_channels_list[3]//w , out_channels = out_channels_list[1]//w , upsample_ratio=4)
        self.Upsample3_2_2 = BilinearUpsampleConv1x1(in_channels = out_channels_list[3]//w , out_channels = out_channels_list[2]//w , upsample_ratio=2)

        self.Upsample1_3_2 = BilinearUpsampleConv1x1(in_channels = out_channels_list[1]//w , out_channels = out_channels_list[1]//w , upsample_ratio=2)
        self.Upsample2_3_4= BilinearUpsampleConv1x1(in_channels = out_channels_list[2]//w , out_channels = out_channels_list[2]//w , upsample_ratio=4)
        self.Upsample3_3_8 = BilinearUpsampleConv1x1(in_channels = out_channels_list[3]//w , out_channels = out_channels_list[3]//w , upsample_ratio=8)     

        self.mixUpsample1_3_2 = MixedUpsample(in_channels = out_channels_list[1]//w , out_channels = out_channels_list[1]//w , upsample_ratio=2)
        self.mixUpsample2_3_4= MixedUpsample(in_channels = out_channels_list[2]//w , out_channels = out_channels_list[2]//w , upsample_ratio=4)
        self.mixUpsample3_3_8 = MixedUpsample(in_channels = out_channels_list[3]//w , out_channels = out_channels_list[3]//w , upsample_ratio=8)



        self.relu = nn.ReLU(inplace=False)

        self.Output_Confidence_Lane  = Output(in_channels=256, out_channels= 1 )
        self.Output_Confidence_Area  = Output(in_channels=256, out_channels= 1 )  

        self.Output_EmbeddingFeatureArea   = Output(in_channels=256, out_channels = self.p.feature_size)

    def forward(self, Half,Quarter,Octant,One_sixteenth):

        # -----------------------Stage 1------------------------

        Branch1 =      self.BasicBlock1_1(self.relu(Quarter))

        Branch2 =      self.BasicBlock2_1(self.relu(Octant))

        Branch1_1_2 = self.Downsample1_1_2(Branch1)
        Branch2_1_2 = self.Upsample2_1_2(Branch2)

        Branch1 = Branch1 + Branch2_1_2

        Branch2 = Branch2 + Branch1_1_2

        # -----------------------Stage 2------------------------

        Branch1 = self.BasicBlock1_2(Branch1)
        Branch2 = self.BasicBlock2_2(Branch2)
        Branch3 = self.BasicBlock3_2(self.relu(One_sixteenth))

        Branch1_2_2 = self.Downsample1_2_2(Branch1)
        Branch1_2_4 = self.Downsample1_2_4(Branch1)
        BranchD2_2_2 = self.Downsample2_2_2(Branch2)
        BranchU2_2_2 = self.Upsample2_2_2(Branch2)
        Branch3_2_2 = self.Upsample3_2_2(Branch3)
        Branch3_2_4 = self.Upsample3_2_4(Branch3)

        Branch1 = Branch1 + BranchU2_2_2 + Branch3_2_4
        Branch2 = Branch2 + Branch1_2_2 + Branch3_2_2
        Branch3 = Branch3 + Branch1_2_4 + BranchD2_2_2
        
        # ------------------------Mix Stage------------------------

        Branch4 = self.BasicBlock4_3(Half)

        # Branch1_i = self.Downsample1_3_4(Branch1)

        # Branch2_I = self.Downsample2_3_2(Branch2)

        # Branch3_I = self.BasicBlock3_3(Branch3)

        Branch1_a = self.Upsample1_3_2(Branch1)
        Branch2_a = self.Upsample2_3_4(Branch2)
        Branch3_a = self.Upsample3_3_8(Branch3)

        Branch1_l = self.mixUpsample1_3_2(Branch1)

        Branch2_l = self.mixUpsample2_3_4(Branch2)

        Branch3_l = self.mixUpsample3_3_8(Branch3)


        Out_Confidence_l = torch.cat((Branch1_l,Branch2_l,Branch3_l ,Branch4),dim=1)
        Out_Confidence_a = torch.cat((Branch1_a,Branch2_a,Branch3_a ,Branch4),dim=1)

        # Out_Instance   = torch.cat((Branch1_i,Branch2_I,Branch3_I),dim=1)

        Out_Confidence_lane = self.Output_Confidence_Lane(Out_Confidence_l)
        Out_Confidence_area = self.Output_Confidence_Area(Out_Confidence_a)

        Out_Confidence = torch.cat((Out_Confidence_area,Out_Confidence_lane),dim=1)
        # Out_EmbeddingFeature_Area   = self.Output_EmbeddingFeatureArea(Out_Instance)
        Out_EmbeddingFeature_Area = []
        return [Out_Confidence, Out_EmbeddingFeature_Area ]  

