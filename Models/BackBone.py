from torch import nn
import torch
import os 
import sys
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(project_root)
from .utils import *
from Models.glam import GLAM

class ConvBNSiLU(nn.Sequential):
    """
    A sequential module consisting of a convolutional layer, batch normalization, and SiLU activation.
    
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
            nn.BatchNorm2d(n_filters),
            nn.SiLU(inplace=True),
        )

class BackBone(nn.Module):
    def __init__(self, in_channels, out_channels_list, w, d, r):
        super(BackBone, self).__init__()

        # Define the layers
        self.conv1 = ConvBNSiLU(in_channels=in_channels, n_filters=out_channels_list[0]//w, k_size=3, padding=1, stride=2)
        self.conv2 = ConvBNSiLU(in_channels=out_channels_list[0]//w, n_filters=out_channels_list[1]//w, k_size=3, padding=1, stride=2)
        self.c3_1 = C3(in_channels=out_channels_list[1]//w, out_channels=out_channels_list[1]//w, depth=3//d)
        self.conv3 = ConvBNSiLU(in_channels=out_channels_list[1]//w, n_filters=out_channels_list[2]//w, k_size=3, padding=1, stride=2)
        # self.glam_1 = GLAM(in_channels=out_channels_list[2]//w, num_reduced_channels=out_channels_list[2]//w, feature_map_size=80, kernel_size=5)

        self.c3_2 = C3(in_channels=out_channels_list[2]//w, out_channels=out_channels_list[2]//w, depth=6//d)

        self.conv4 = ConvBNSiLU(in_channels=out_channels_list[2]//w, n_filters=out_channels_list[3]//w, k_size=3, padding=1, stride=2)
        # self.glam_2 = GLAM(in_channels=out_channels_list[3]//w, num_reduced_channels=out_channels_list[3]//w, feature_map_size=40, kernel_size=5)

        self.c3_3 = C3(in_channels=out_channels_list[3]//w, out_channels=out_channels_list[3]//w, depth=6//d)


    def forward(self, x):
        x = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.c3_1(out2)
        x = self.conv3(out3)
        # res_1 = self.glam_1(x)
        out5 = self.c3_2(x)# + res_1
        x = self.conv4(out5)
        # res_2 = self.glam_2(x)
        out7 = self.c3_3(x)# + res_2

        
        return out2 ,out3 , out5, out7
