import torch 
from torch import nn

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


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()

        c_ = int(in_channels//2)

        self.c1 = ConvBNSiLU(in_channels, c_, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.c_out = ConvBNSiLU(c_ * 4, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.c1(x)
        pool1 = self.pool(x)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)

        return self.c_out(torch.cat([x, pool1, pool2, pool3], dim=1))



class Neck(nn.Module):
    def __init__(self, in_channels , w=4,r=2,d=3):
        super(Neck , self).__init__()
        self.c1 = nn.Sequential(
            ConvBNSiLU(in_channels[3]//w,in_channels[3]*r//w,kernel_size=3,stride=2,padding=1),
            C3(in_channels[3]*r//w,in_channels[3]*r//w,width_multiple=1,depth=1),
            SPPF(in_channels[3]*r//w,in_channels[3]*r//w))
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.c3_1 = C3(in_channels[3]*(r+1)//w,in_channels[3]//w,depth=d//3)

        self.cbl_1 = ConvBNSiLU(in_channels[3]//w,in_channels[3]//w,kernel_size=1,stride=1,padding=1)
        
        self.c3_2 = C3((in_channels[3]+in_channels[2])//w,in_channels[2]//w,depth=d//3)

    def forward(self, Octant,One_sixteenth):

        One_thirty_second_out = self.c1(One_sixteenth)

        One_sixteenth_up = self.up(One_thirty_second_out)

        One_sixteenth_out = self.c3_1(torch.cat((One_sixteenth,One_sixteenth_up),dim=1))

        Octant_up = self.up(One_sixteenth_out)

        Octant_out = self.c3_2(torch.cat((Octant,Octant_up),dim=1))

        return Octant_out , One_sixteenth_out , One_thirty_second_out

