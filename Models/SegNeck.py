import torch 
from torch import nn
import numpy as np

import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.Parameters import Parameters


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
    def __init__(self, in_channels, n_filters, k_size, padding, stride, bias=True):
        super().__init__(
            nn.Conv2d(in_channels, n_filters, k_size, padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
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
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, repeat_num = 4):
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
        self.conv3 = ConvBNReLU(in_channels//8, out_channels, 1, 0, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class FusionLayer(nn.Module):
    """
    A module for performing feature fusion across multiple branches.

    Args:
        in_branches (int): Number of input branches.
        out_branches (int): Number of output branches.
        c (int): Number of channels in the convolutional layers.
    """
    def __init__(self, in_branches , out_branches, c):
        super(FusionLayer, self).__init__()

        self.out_branches = out_branches
        self.in_branches = in_branches
        self.fuse_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.out_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.in_branches):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                      bias=False),
                            nn.BatchNorm2d(c * (2 ** j), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=False),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=False)


    def forward(self, branches):
        assert len(self.fuse_layers) == len(branches)

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](branches[0]))

                else:
                    # print(self.fuse_layers[i][j](branches[j]).shape,x_fused[i].shape)
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](branches[j]).clone()


        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i].clone())

        return x_fused

class SegNeck(nn.Module):
    """
    A segmentation network neck module.

    Args:
        Resize (int, optional): The factor by which the feature maps are resized. Default is 1.
    """

    def __init__(self, out_channels_list , class_number):
        super(SegNeck,self).__init__()

        self.p = Parameters()
        # BasicBlockB_S B is number of branch and S is number of stage
        self.BasicBlock1_1 = BasicBlock(in_channels = out_channels_list[0] , out_channels = out_channels_list[0])
        self.BasicBlock1_4 = BasicBlock(in_channels = out_channels_list[0] , out_channels = out_channels_list[0])
        self.BasicBlock1_2 = BasicBlock(in_channels = out_channels_list[0] , out_channels = out_channels_list[0])
        self.BasicBlock1_3 = BasicBlock(in_channels = out_channels_list[0] , out_channels = out_channels_list[0])
        self.BasicBlock2_2 = BasicBlock(in_channels = out_channels_list[1] , out_channels = out_channels_list[1])
        self.BasicBlock2_3 = BasicBlock(in_channels = out_channels_list[1] , out_channels = out_channels_list[1])
        self.BasicBlock2_4 = BasicBlock(in_channels = out_channels_list[1] , out_channels = out_channels_list[1])
        self.BasicBlock3_3 = BasicBlock(in_channels = out_channels_list[2] , out_channels = out_channels_list[2])
        self.BasicBlock3_4 = BasicBlock(in_channels = out_channels_list[2] , out_channels = out_channels_list[2])
        self.BasicBlock4_4 = BasicBlock(in_channels = out_channels_list[3] , out_channels = out_channels_list[3])
        self.SecondBasicBlock1_4 = BasicBlock(in_channels = out_channels_list[0] , out_channels = out_channels_list[0])
        self.mixUpsample2_4_2 = MixedUpsample(in_channels = out_channels_list[1] , out_channels = out_channels_list[1] , upsample_ratio=2)
        self.mixUpsample3_4_4 = MixedUpsample(in_channels = out_channels_list[2] , out_channels = out_channels_list[2] , upsample_ratio=4)
        self.mixUpsample4_4_8 = MixedUpsample(in_channels = out_channels_list[3] , out_channels = out_channels_list[3] , upsample_ratio=8)
        
        self.SecondmixUpsample2_4_2 = MixedUpsample(in_channels = out_channels_list[1] , out_channels = out_channels_list[1] , upsample_ratio=2)
        self.SecondmixUpsample3_4_4 = MixedUpsample(in_channels = out_channels_list[2] , out_channels = out_channels_list[2] , upsample_ratio=4)
        self.SecondmixUpsample4_4_8 = MixedUpsample(in_channels = out_channels_list[3] , out_channels = out_channels_list[3] , upsample_ratio=8)

        self.relu = nn.ReLU(inplace=False)
        self.Fusing2 = FusionLayer(2,2,out_channels_list[0])
        self.Fusing3 = FusionLayer(3,3,out_channels_list[0])
        self.Fusing4 = FusionLayer(4,4,out_channels_list[0])

        self.Output_Confidence  = Output(in_channels=np.sum(out_channels_list), out_channels= class_number )

        self.Output_EmbeddingFeatureLane   = Output(in_channels=np.sum(out_channels_list), out_channels = self.p.feature_size)
        self.Output_EmbeddingFeatureArea   = Output(in_channels=np.sum(out_channels_list), out_channels = self.p.feature_size)

    def forward(self, Half,Quarter,Octant,One_sixteenth):
        # -----------------------Stage 1------------------------

        Branch1 =      self.BasicBlock1_1(self.relu(Half))


        # -----------------------Stage 2------------------------

        Branch2 =      self.BasicBlock2_2(self.relu(Quarter))

        Branch1 , Branch2 = self.Fusing2([Branch1,Branch2])


        # -----------------------Stage 3------------------------

        Branch1 = self.BasicBlock1_3(Branch1)
        Branch2 = self.BasicBlock2_3(Branch2)
        Branch3 = self.BasicBlock3_3(self.relu(Octant))

        Branch1 , Branch2 , Branch3 = self.Fusing3([Branch1 , Branch2 , Branch3])

        # -----------------------Stage 4------------------------

        Branch1 = self.BasicBlock1_4(Branch1)
        Branch2 = self.BasicBlock2_4(Branch2)
        Branch3 = self.BasicBlock3_4(Branch3)
        Branch4 = self.BasicBlock4_4(self.relu(One_sixteenth))

        Branch1 , Branch2 , Branch3 , Branch4 = self.Fusing4([Branch1 , Branch2 , Branch3 , Branch4])

        # ------------------------Output------------------------

        Branch1 = self.BasicBlock1_4(Branch1)
        Branch2_1 = self.mixUpsample2_4_2(Branch2)
        Branch3_1 = self.mixUpsample3_4_4(Branch3)
        Branch4_1 = self.mixUpsample4_4_8(Branch4)
        
        SecondBranch1 = self.SecondBasicBlock1_4(Branch1)
        SecondBranch2_1 = self.SecondmixUpsample2_4_2(Branch2)
        SecondBranch3_1 = self.SecondmixUpsample3_4_4(Branch3)
        SecondBranch4_1 = self.SecondmixUpsample4_4_8(Branch4)

        
        Out_Confidence = torch.cat((Branch1,Branch2_1,Branch3_1 , Branch4_1),dim=1)
        Out_Instance   = torch.cat((SecondBranch1,SecondBranch2_1,SecondBranch3_1 , SecondBranch4_1),dim=1)

        Out_Confidence = self.Output_Confidence(Out_Confidence)

        Out_EmbeddingFeatureLane   = self.Output_EmbeddingFeatureLane(Out_Instance)

        Out_EmbeddingFeatureArea   = self.Output_EmbeddingFeatureArea(Out_Instance)


        return [Out_Confidence, torch.cat((Out_EmbeddingFeatureArea,Out_EmbeddingFeatureLane),dim=1)]  

